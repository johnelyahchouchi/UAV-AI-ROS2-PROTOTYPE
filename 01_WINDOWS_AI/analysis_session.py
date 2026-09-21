"""Local session journal connecting analysis workers to the desktop workspace."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import tempfile
import time
from typing import Any


def mean_metric(targets: list[dict], key: str) -> float | None:
    """Average defined finite values; missing observations remain undefined."""
    values = [item.get(key) for item in targets]
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def extraction_record(view: Any, state: Any, *, started_at: float) -> dict:
    """Preserve all analysis fields and every sample detection, without image pixels."""
    analysis = view.analysis.to_dict()
    samples = []
    for index, sample in enumerate(view.analysis.samples):
        detections = sample.detections if state.method == "v1" else sample
        samples.append({
            "index": index,
            "family": sample.family if state.method == "v1" else "mc_dropout",
            "parameters": dict(sample.parameters) if state.method == "v1" else {},
            "detections": [asdict(item) for item in detections],
        })
    targets = analysis.get("targets", [])
    return {
        "schema_version": 1, "method": state.method, "frame": state.frame_number,
        "captured_utc": state.sampled_utc,
        "elapsed_seconds": max(0.0, state.sampled_at - started_at),
        "duration_ms": state.duration_ms, "state": state.state, "status": state.status,
        "analysis": analysis, "samples": samples,
        "metrics": {
            "persistence": mean_metric(targets, "detection_persistence" if state.method == "v1" else "persistence"),
            "agreement": mean_metric(targets, "class_agreement" if state.method == "v1" else "winner_class_agreement"),
            "iou": mean_metric(targets, "mean_iou_to_reference" if state.method == "v1" else "mean_reference_iou"),
            "target_count": len(targets),
        },
    }


class AnalysisSession:
    """Transactional local journal; each operation owns its SQLite connection."""

    def __init__(self, directory: Path, *, create: bool = False) -> None:
        self.directory = Path(directory).resolve()
        self.path = self.directory / "analysis.sqlite3"
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS results (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS commands (seq INTEGER PRIMARY KEY, key INTEGER NOT NULL);
                """)
        elif not self.path.is_file():
            raise ValueError("Select a session folder containing analysis.sqlite3")

    @contextmanager
    def _connect(self, *, readonly: bool = False):
        """Commit/rollback and explicitly close connections on every path."""
        if readonly:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=0.1)
        else:
            connection = sqlite3.connect(self.path, timeout=2)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def set_metadata(self, **values: Any) -> None:
        """Store session provenance and explicit availability/error messages."""
        with self._connect() as db:
            db.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                           [(key, json.dumps(value, allow_nan=False)) for key, value in values.items()])

    def metadata(self) -> dict:
        """Read provenance without loading any model or making network requests."""
        with self._connect(readonly=True) as db:
            return {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}

    def append(self, record: dict) -> None:
        """Commit one complete result; readers cannot see half a record."""
        payload = json.dumps(record, allow_nan=False, separators=(",", ":"))
        with self._connect() as db:
            db.execute("INSERT INTO results(payload) VALUES (?)", (payload,))

    def read_after(self, sequence: int, *, limit: int = 32) -> list[tuple[int, dict]]:
        """Read bounded batches in observation order."""
        with self._connect(readonly=True) as db:
            rows = db.execute("SELECT seq,payload FROM results WHERE seq>? ORDER BY seq LIMIT ?", (sequence, limit)).fetchall()
        return [(seq, json.loads(payload)) for seq, payload in rows]

    def read_record(self, sequence: int) -> dict:
        """Fetch detail on selection, keeping the graph's in-memory history small."""
        with self._connect(readonly=True) as db:
            row = db.execute("SELECT payload FROM results WHERE seq=?", (sequence,)).fetchone()
        if row is None:
            raise ValueError("Extraction record is no longer available")
        return json.loads(row[0])

    def command(self, key: str) -> None:
        """Queue only the supported embedded-view controls."""
        if key not in ("p", "z", "s", "q"):
            raise ValueError("Unsupported workspace control")
        with self._connect() as db:
            db.execute("INSERT INTO commands(key) VALUES (?)", (ord(key),))

    def next_command(self, after: int) -> tuple[int, int]:
        """Return one pending key or an unchanged cursor and no-key sentinel."""
        with self._connect(readonly=True) as db:
            row = db.execute("SELECT seq,key FROM commands WHERE seq>? ORDER BY seq LIMIT 1", (after,)).fetchone()
        return tuple(row) if row else (after, -1)

    def export_json(self, destination: Path) -> None:
        """Atomically export through a fixed record ID without holding a long read lock."""
        destination = Path(destination).resolve()
        if destination == self.path or destination.suffix.lower() != ".json":
            raise ValueError("Choose a .json export file, not the session database")
        metadata = self.metadata()
        with self._connect(readonly=True) as db:
            maximum = db.execute("SELECT COALESCE(MAX(seq),0) FROM results").fetchone()[0]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=".analysis-export-", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write('{"metadata":' + json.dumps(metadata) + ',"records":[')
                cursor, separator = 0, ""
                while cursor < maximum:
                    with self._connect(readonly=True) as db:
                        rows = db.execute("SELECT seq,payload FROM results WHERE seq>? AND seq<=? ORDER BY seq LIMIT 8", (cursor, maximum)).fetchall()
                    if not rows:
                        raise ValueError("Session changed during export; retry with the original session")
                    for cursor, payload in rows:
                        output.write(separator + payload)
                        separator = ","
                output.write("]}")
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


class EmbeddedPreview:
    """OpenCV-compatible display sink with a one-frame background preview mailbox."""

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 1

    def __init__(self, cv2_module: Any, session: AnalysisSession) -> None:
        self.cv2, self.session = cv2_module, session
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._frame = None
        self._last = 0.0
        self._cursor = 0
        self._worker = threading.Thread(target=self._publish, daemon=True, name="workspace-preview")
        self._worker.start()

    def __getattr__(self, name: str):
        return getattr(self.cv2, name)

    def namedWindow(self, *args):
        """The host dashboard owns the window."""

    resizeWindow = moveWindow = namedWindow

    def getWindowProperty(self, *args):
        return 1

    def imshow(self, name: str, frame: Any) -> None:
        """Copy at most four dashboard frames per second; discard older previews."""
        now = time.monotonic()
        if now - self._last < 0.25:
            return
        self._last = now
        with self._lock:
            self._frame = frame.copy()
        self._wake.set()

    def waitKey(self, delay: int) -> int:
        if delay > 1:
            time.sleep(min(delay, 30) / 1000)
        try:
            self._cursor, key = self.session.next_command(self._cursor)
            return key
        except sqlite3.Error:
            return -1

    def _publish(self) -> None:
        while not self._closed:
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                ok, encoded = self.cv2.imencode(".jpg", frame)
                if not ok:
                    raise OSError("Preview JPEG encoding failed")
                temporary = self.session.directory / "preview.tmp"
                temporary.write_bytes(encoded.tobytes())
                os.replace(temporary, self.session.directory / "preview.jpg")
            except OSError as error:
                print(f"Workspace preview unavailable: {error}", flush=True)

    def destroyAllWindows(self) -> None:
        self._closed = True
        self._wake.set()
        self._worker.join(timeout=1)
