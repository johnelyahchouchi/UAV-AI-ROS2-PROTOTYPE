"""Native live, uncertainty-history, and recorded-extraction workspace."""

from __future__ import annotations

from collections import deque
from io import BytesIO
import json
from pathlib import Path
import queue
import sqlite3
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from PIL import Image, ImageTk

from analysis_session import AnalysisSession
from .configuration import PROJECT_ROOT


BG = "#101c2d"
TEXT = "#ecf4ff"
MUTED = "#93a7bd"
CYAN = "#29d3e2"
AMBER = "#ffbd59"
HISTORY_LIMIT = 300


def read_preview(path: Path) -> Image.Image:
    """Release the shared Windows file before decoding its image bytes."""
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as source:
        return source.copy()


def metric_series(records: list[dict], metric: str) -> list[tuple[float, float | None]]:
    """Keep undefined/failure points as gaps rather than turning them into zeros."""
    return [(float(row["elapsed_seconds"]), row.get("metrics", {}).get(metric)) for row in records]


class HistoryChart(tk.Canvas):
    """Small fixed 0-1 line chart without smoothing or probability claims."""

    def __init__(self, parent, title: str, keys: tuple[tuple[str, str, str], ...]):
        super().__init__(parent, background=BG, highlightthickness=0, height=175)
        self.title, self.keys, self.records = title, keys, []
        self.bind("<Configure>", lambda event: self.redraw())

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(200, self.winfo_width()), max(140, self.winfo_height())
        left, top, right, bottom = 42, 48, width - 15, height - 30
        self.create_text(12, 12, text=self.title, fill=TEXT, anchor="nw", font=("Segoe UI", 10, "bold"))
        for index, (_, label, color) in enumerate(self.keys):
            self.create_text(12 + index * 180, 32, text=label, fill=color, anchor="w", font=("Segoe UI", 9))
        for value in (0, .5, 1):
            y = bottom - value * (bottom - top)
            self.create_line(left, y, right, y, fill="#263b55")
            self.create_text(left - 8, y, text=f"{value:g}", fill=MUTED, anchor="e")
        self.create_text(width // 2, height - 10, text="Sample capture time (session seconds)", fill=MUTED, font=("Segoe UI", 8))
        if not self.records:
            self.create_text(width // 2, (top + bottom) // 2, text="Waiting for measured results", fill=MUTED)
            return
        start = min(row["elapsed_seconds"] for row in self.records)
        end = max(row["elapsed_seconds"] for row in self.records)
        span = max(1, end - start)
        for value, anchor, x in ((start, "nw", left), (end, "ne", right)):
            self.create_text(x, bottom + 2, text=f"{value:.1f}", anchor=anchor, fill=MUTED, font=("Segoe UI", 8))
        for key, _, color in self.keys:
            previous = None
            for seconds, value in metric_series(self.records, key):
                if value is None:
                    previous = None
                    continue
                point = (left + (seconds - start) / span * (right - left), bottom - value * (bottom - top))
                if previous:
                    self.create_line(*previous, *point, fill=color, width=2)
                self.create_oval(point[0]-2, point[1]-2, point[0]+2, point[1]+2, fill=color, outline="")
                previous = point


def text_area(parent) -> tk.Text:
    """Create a scrollable read-only text surface for full extraction fields."""
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    text = tk.Text(frame, bg=BG, fg=TEXT, insertbackground=TEXT, wrap="word", height=7,
                   font=("Consolas", 9), relief="flat", state="disabled")
    scroll = ttk.Scrollbar(frame, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    return text


def set_text(widget: tk.Text, value: str) -> None:
    """Replace a read-only text widget's contents."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("end", value)
    widget.configure(state="disabled")


def table_area(parent, columns: tuple[str, ...], *, height: int = 4, expand: bool = False) -> ttk.Treeview:
    """Create a table with visible vertical and horizontal navigation."""
    container = ttk.Frame(parent)
    container.pack(fill="both" if expand else "x", expand=expand)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(0, weight=1)
    tree = ttk.Treeview(container, columns=columns, show="headings", height=height, selectmode="browse")
    vertical = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    return tree


class MethodPanel(ttk.Frame):
    """Two trend graphs and every extracted field for selected sampled frames."""

    def __init__(self, parent, method: str):
        super().__init__(parent, padding=8)
        self.method = method
        self.load_record = None
        self.records: deque[tuple[int, dict]] = deque(maxlen=HISTORY_LIMIT)
        self.identity = tk.StringVar(value=f"{method.upper()}: waiting for a session")
        ttk.Label(self, textvariable=self.identity).pack(anchor="w")
        ttk.Label(self, text="Frame-level means across observed objects; identities are not tracked across samples. Undefined values leave gaps.",
                  style="Muted.TLabel").pack(anchor="w", pady=(3, 6))
        graphs = ttk.Frame(self)
        graphs.pack(fill="x")
        graphs.columnconfigure((0, 1), weight=1)
        self.charts = (
            HistoryChart(graphs, "Detection and class consistency", (("persistence", "Persistence", CYAN), ("agreement", "Class agreement", AMBER))),
            HistoryChart(graphs, "Bounding-box consistency", (("iou", "Mean reference IoU", CYAN),)),
        )
        for index, chart in enumerate(self.charts):
            chart.grid(row=0, column=index, sticky="ew", padx=(0, 8))
        ttk.Label(self, text="Recent sampled frames (latest 300 per method). Select a row for all samples and metrics.",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 3))
        self.tree = table_area(self, ("frame", "time", "objects", "status"), height=3)
        for key, label, width in (("frame", "Frame", 70), ("time", "Captured UTC", 235), ("objects", "Objects", 70), ("status", "Result", 400)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width)
        self.tree.bind("<<TreeviewSelect>>", self.select)
        self.details = ttk.Notebook(self)
        self.details.pack(fill="both", expand=True, pady=(6, 0))
        samples_tab, metrics_tab, raw_tab = (ttk.Frame(self.details) for _ in range(3))
        for panel, label in ((samples_tab, "Sample inputs"), (metrics_tab, "Object metrics"), (raw_tab, "Complete JSON")):
            self.details.add(panel, text=label)
        self.sample_table = table_area(samples_tab, ("sample", "transformation", "parameters", "detections"), expand=True)
        for key, label in (("sample", "Sample / pass"), ("transformation", "Transformation"), ("parameters", "Actual parameters"), ("detections", "Detections")):
            self.sample_table.heading(key, text=label)
            self.sample_table.column(key, width=85 if key in ("sample", "detections") else 250)
        self.metric_table = table_area(metrics_tab, ("id", "class", "persistence", "agreement", "iou", "confidence_std"), expand=True)
        for key, label in (("id", "Object (this frame)"), ("class", "Dominant class"), ("persistence", "Persistence"), ("agreement", "Class agreement"), ("iou", "Reference IoU"), ("confidence_std", "Confidence std")):
            self.metric_table.heading(key, text=label)
            self.metric_table.column(key, width=130)
        ttk.Label(metrics_tab, text="Persistence: fraction of inputs/passes detected. Agreement: dominant class fraction when detected. IoU: box overlap consistency.",
                  style="Muted.TLabel", wraplength=1000).pack(anchor="w")
        self.detail = text_area(raw_tab)
        set_text(self.detail, "Select a completed extraction. No synthetic data is shown here.")

    def reset(self) -> None:
        self.records.clear()
        self.tree.delete(*self.tree.get_children())
        self.sample_table.delete(*self.sample_table.get_children())
        self.metric_table.delete(*self.metric_table.get_children())
        set_text(self.detail, "Waiting for a sampled frame.")
        for chart in self.charts:
            chart.records = []
            chart.redraw()

    def add(self, sequence: int, record: dict) -> None:
        selected = self.tree.selection()
        last = str(self.records[-1][0]) if self.records else None
        follow = not selected or selected[0] == last
        if len(self.records) == HISTORY_LIMIT:
            self.tree.delete(str(self.records[0][0]))
        summary = {key: value for key, value in record.items() if key not in ("samples", "analysis")}
        self.records.append((sequence, summary))
        self.tree.insert("", "end", iid=str(sequence), values=(record["frame"], record.get("captured_utc"),
                         record.get("metrics", {}).get("target_count", "N/A"), record["status"]))
        for chart in self.charts:
            chart.records = [row for _, row in self.records]
            chart.redraw()
        if follow:
            self.tree.selection_set(str(sequence))
            self.tree.see(str(sequence))
            self.select()

    def select(self, event=None) -> None:
        selected = self.tree.selection()
        if selected:
            for seq, record in self.records:
                if str(seq) == selected[0]:
                    if self.load_record is not None:
                        record = self.load_record(seq)
                    samples = record.get("samples", [])
                    self.sample_table.delete(*self.sample_table.get_children())
                    for sample in samples:
                        self.sample_table.insert("", "end", values=(sample["index"], sample["family"],
                            ", ".join(f"{key}={value:.3g}" for key, value in sample["parameters"].items()) or "Unchanged pixels",
                            len(sample["detections"])))
                    self.metric_table.delete(*self.metric_table.get_children())
                    def metric(target, key):
                        value = target.get(key)
                        return "N/A" if value is None else f"{value:.3f}"
                    for target in record.get("analysis", {}).get("targets", []):
                        self.metric_table.insert("", "end", values=(target["target_id"],
                            target.get("dominant_class", target.get("dominant_winner_class", "N/A")),
                            metric(target, "detection_persistence" if self.method == "v1" else "persistence"),
                            metric(target, "class_agreement" if self.method == "v1" else "winner_class_agreement"),
                            metric(target, "mean_iou_to_reference" if self.method == "v1" else "mean_reference_iou"),
                            metric(target, "confidence_std" if self.method == "v1" else "winner_confidence_std")))
                    explanation = (
                        "V1: clean input plus named perturbations; parameters describe the actual image transformations.\n"
                        if self.method == "v1" else
                        "V2: unchanged input across stochastic model passes; no input blur or zoom is applied.\n"
                    )
                    summary = "\n".join(f"Sample {row['index']}: {row['family']} | {len(row['detections'])} detections | {row['parameters']}" for row in samples)
                    set_text(self.detail, explanation + "Neither method estimates correctness probability.\n\n" + summary +
                             "\n\nALL EXTRACTED FIELDS (boxes are source-image pixels):\n" + json.dumps(record, indent=2))
                    break


class AnalysisWorkspace(ttk.Frame):
    """One host-branded surface for live and recorded results."""

    def __init__(self, parent, *, launch: Callable, settings: Callable, open_path: Callable):
        super().__init__(parent)
        self.launch, self.settings, self.open_path = launch, settings, open_path
        self.session = None
        self.sequence = 0
        self._image = None
        self._image_stamp = None
        self._metadata = None
        self._closed = False
        self._job = None
        self._cancel = threading.Event()
        self._controller = None
        self._progress = (0.0, "Ready")
        self._results = queue.Queue()
        self._outputs = None
        self._notifications = queue.Queue()
        self._export_job = None
        self._live_controls = False
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 6))
        for label, command in (("Start live session", launch), ("Open saved session", self.open_session), ("Export full JSON", self.export)):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Dashboard catalog", command=lambda: self.open_path(
            PROJECT_ROOT / "07_DOCUMENTATION/DASHBOARD_QUICK_CATALOG.md"
        )).pack(side="left")
        self.status = tk.StringVar(value="Configure models in Live Tester, then start a session. Recorded extraction is available below.")
        ttk.Label(self, textvariable=self.status, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(0, 6))
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.live = ttk.Frame(self.tabs)
        self.tabs.add(self.live, text="Live view")
        controls = ttk.Frame(self.live)
        controls.pack(fill="x")
        for label, key in (("Pause / resume", "p"), ("2x zoom", "z"), ("Save screenshot", "s"), ("Stop session", "q")):
            ttk.Button(controls, text=label, command=lambda value=key: self.send_key(value)).pack(side="left", padx=(0, 8))
        self.preview_status = tk.StringVar(value="Waiting for a live preview")
        ttk.Label(self.live, textvariable=self.preview_status, style="Muted.TLabel").pack(anchor="w", pady=5)
        self.preview = tk.Label(self.live, bg="#08111f", fg=TEXT, text="Your live video and explanation panels appear here.")
        self.preview.pack(fill="both", expand=True)
        self.methods = {method: MethodPanel(self.tabs, method) for method in ("v1", "v2")}
        for method, panel in self.methods.items():
            self.tabs.add(panel, text=f"{method.upper()} extraction + graphs")
        self.recorded = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.recorded, text="Recorded extraction")
        self._build_recorded()
        self.after(250, self.poll)

    def attach(self, session: AnalysisSession, *, live_controls: bool = False) -> None:
        """Select a session; saved-session review never queues live controls."""
        self._live_controls = live_controls
        self.session, self.sequence, self._image_stamp, self._metadata = session, 0, None, None
        self._image = None
        self.preview.configure(image="", text="Waiting for this session's preview")
        for panel in self.methods.values():
            panel.load_record = session.read_record
            panel.reset()
        self.status.set(f"Session: {session.directory.name} | Full results are saved locally.")

    def open_session(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Open an analysis session", initialdir=PROJECT_ROOT / "08_OUTPUTS/analysis_sessions")
        if folder:
            try:
                self.attach(AnalysisSession(Path(folder)))
            except (ValueError, sqlite3.Error) as error:
                messagebox.showerror("Cannot open session", str(error), parent=self)

    def send_key(self, key: str) -> None:
        """Send a control only to the currently attached live session."""
        if self.session and self._live_controls and (self._metadata or {}).get("status") != "Stopped":
            try:
                self.session.command(key)
            except sqlite3.Error as error:
                self.status.set(f"Control unavailable: {error}")
        else:
            self.status.set("This is a saved or stopped session. Start a live session to use playback controls.")

    def export(self) -> None:
        """Export in a worker so video previews and controls stay responsive."""
        if self._export_job and self._export_job.is_alive():
            self.status.set("An export is already running.")
            return
        if not self.session:
            self.status.set("Start or open a session before exporting.")
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if path:
            session = self.session
            def write_export():
                try:
                    session.export_json(Path(path))
                    self._notifications.put(f"Exported committed results to {Path(path).name}")
                except (OSError, sqlite3.Error, ValueError) as error:
                    self._notifications.put(f"Export failed: {error}")
            self.status.set("Exporting committed results...")
            self._export_job = threading.Thread(target=write_export, daemon=True, name="analysis-export")
            self._export_job.start()

    def poll(self) -> None:
        if self._closed:
            return
        try:
            self.status.set(self._notifications.get_nowait())
        except queue.Empty:
            pass
        if self.session:
            try:
                metadata = self.session.metadata()
                if metadata != self._metadata:
                    self._metadata = metadata
                    for method, panel in self.methods.items():
                        identity = metadata.get(f"{method}_model") or "Unavailable"
                        reason = metadata.get("v2_availability", "Waiting") if method == "v2" else "Input perturbation robustness"
                        if metadata.get("uncertainty_enabled") is False:
                            reason = "Disabled: enable continuous uncertainty in Live Tester"
                        panel.identity.set(f"{method.upper()} | {identity} | {reason}")
                for sequence, record in self.session.read_after(self.sequence, limit=8):
                    if record.get("method") in self.methods:
                        self.methods[record["method"]].add(sequence, record)
                    self.sequence = sequence
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
                self.status.set(f"Session read unavailable: {error}")
            self.refresh_preview()
        self.recorded_progress["value"] = self._progress[0] * 100
        self.recorded_status.set(self._progress[1])
        try:
            kind, value = self._results.get_nowait()
            if kind == "result":
                self._outputs = value.outputs
                self.rows = value.detection_rows
                self.page = 0
                self.show_page()
                set_text(self.recorded_detail, json.dumps(value.summary, indent=2) + "\n\n" + (value.model_warning or ""))
            else:
                set_text(self.recorded_detail, str(value))
        except queue.Empty:
            pass
        self.after(250, self.poll)

    def refresh_preview(self) -> None:
        """Retry preview file contention independently of extraction/journal reads."""
        try:
            preview_path = self.session.directory / "preview.jpg"
            stamp = preview_path.stat().st_mtime_ns
            age = max(0, time.time() - stamp / 1e9)
            size = (max(320, self.preview.winfo_width()), max(180, self.preview.winfo_height()))
            signature = (stamp, size)
            if signature != self._image_stamp:
                image = read_preview(preview_path)
                image.thumbnail(size)
                self._image = ImageTk.PhotoImage(image)
                self.preview.configure(image=self._image, text="")
                self._image_stamp = signature
            self.preview_status.set(f"Dashboard preview up to 4 FPS | received {age:.1f}s ago" +
                                    (" | STALE / stopped or paused" if age > 3 else ""))
        except FileNotFoundError:
            self.preview_status.set("Waiting for this session's preview")
        except OSError as error:
            self.preview_status.set(f"Preview unavailable; retrying automatically. Last image retained. {error}")

    def _build_recorded(self) -> None:
        ttk.Label(self.recorded, text="Recorded video extraction", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(self.recorded, text="Uses the base model and inference settings from Live Tester. Choose an MP4, then run. V2 analysis remains in its own tab.",
                  style="Muted.TLabel", wraplength=1000).pack(anchor="w", pady=4)
        self.video = tk.StringVar()
        paths = ttk.Frame(self.recorded)
        paths.pack(fill="x")
        ttk.Entry(paths, textvariable=self.video).pack(side="left", fill="x", expand=True)
        ttk.Button(paths, text="Choose MP4", command=self.choose_video).pack(side="left", padx=6)
        actions = ttk.Frame(self.recorded)
        actions.pack(fill="x", pady=6)
        for label, action in (("Start extraction", self.start_recorded), ("Cancel", self.cancel_recorded),
                              ("Open CSV", lambda: self.open_result("csv_report")),
                              ("Open annotated video", lambda: self.open_result("annotated_video")),
                              ("Output folder", lambda: self.open_result("directory"))):
            ttk.Button(actions, text=label, command=action).pack(side="left", padx=(0, 6))
        self.recorded_status = tk.StringVar(value="Ready")
        ttk.Label(self.recorded, textvariable=self.recorded_status, style="Muted.TLabel").pack(anchor="w")
        self.recorded_progress = ttk.Progressbar(self.recorded, maximum=100)
        self.recorded_progress.pack(fill="x", pady=5)
        columns = ("frame", "seconds", "track", "class_id", "class", "confidence", "x1", "y1", "x2", "y2")
        self.table = table_area(self.recorded, columns, height=5)
        for column in columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=75 if column != "class" else 150)
        pager = ttk.Frame(self.recorded)
        pager.pack(fill="x", pady=4)
        ttk.Button(pager, text="Previous 100", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(pager, text="Next 100", command=lambda: self.change_page(1)).pack(side="left", padx=8)
        self.page_label = tk.StringVar(value="No extracted rows yet")
        ttk.Label(pager, textvariable=self.page_label, style="Muted.TLabel").pack(side="left")
        self.rows, self.page = [], 0
        self.recorded_detail = text_area(self.recorded)
        set_text(self.recorded_detail, "Pipeline: validate input and model hash -> decode frames -> infer -> extract rows -> annotate -> encode MP4 -> validate outputs.\n\nCSV includes every detection: source frame/time, optional track ID, class, confidence, and pixel box coordinates. This workspace uses detection-only mode.\n\nThe table shows up to 10,000 rows, 100 per page. The CSV contains the full extraction.")

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(parent=self, filetypes=(("MP4", "*.mp4"),))
        if path:
            self.video.set(path)

    def change_page(self, delta: int) -> None:
        self.page = max(0, min(max(0, (len(self.rows)-1)//100), self.page + delta))
        self.show_page()

    def show_page(self) -> None:
        self.table.delete(*self.table.get_children())
        start = self.page * 100
        for row in self.rows[start:start+100]:
            self.table.insert("", "end", values=row)
        self.page_label.set(f"Rows {min(start+1, len(self.rows))}-{min(start+100, len(self.rows))} of {len(self.rows)} loaded; full CSV available")

    def open_result(self, field: str) -> None:
        if self._outputs:
            self.open_path(getattr(self._outputs, field))

    def start_recorded(self) -> None:
        if self._job and self._job.is_alive():
            return
        settings = self.settings()
        video = self.video.get() or settings.video_path
        if not video or not settings.model_path:
            self._progress = (0, "Choose a video and configure the base model in Live Tester.")
            return
        self._cancel.clear()
        # Consume an undelivered result from the preceding job before resetting its view.
        while not self._results.empty():
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
        self._outputs = None
        self.rows, self.page = [], 0
        self.show_page()
        set_text(self.recorded_detail, "Validating configured model and video; preparing extraction...")
        self._progress = (0, "Preparing extraction...")
        self._job = threading.Thread(target=self._run_recorded, args=(settings, video), daemon=True)
        self._job.start()

    def _run_recorded(self, settings: Any, video: str) -> None:
        try:
            source = str(PROJECT_ROOT / "01_WINDOWS_AI/model_test_dashboard/src")
            if source not in sys.path:
                sys.path.insert(0, source)
            from uav_model_dashboard.configuration import ProcessingSettings, validate_video_path, validate_model_path
            from uav_model_dashboard.model_manager import ModelManager
            from uav_model_dashboard.output_manager import OutputManager
            from uav_model_dashboard.processing_control import ProcessingController
            from uav_model_dashboard.video_processor import VideoProcessor, ProcessingRequest
            from uav_security.model_integrity import verify_trusted_model
            registry = Path(settings.registry_path) if settings.registry_path else None
            device = {"auto": "Auto", "cpu": "CPU", "0": "GPU 0", "cuda:0": "GPU 0", "cuda": "GPU 0"}.get(settings.device, settings.device)
            configuration = ProcessingSettings.from_values(settings.confidence, settings.iou, settings.image_size, device, "Detection only")
            manager = ModelManager(verifier=lambda path: verify_trusted_model(path, registry))
            self._controller = ProcessingController()
            processor = VideoProcessor(manager, OutputManager(PROJECT_ROOT / "08_OUTPUTS/recorded_extraction"), self._controller)
            if self._cancel.is_set():
                self._progress = (0, "Cancelled before processing")
                return

            def progress(fraction, *, desc):
                self._progress = (fraction, desc)
                if self._cancel.is_set():
                    self._controller.request_cancel()

            result = processor.process(ProcessingRequest(validate_video_path(video), validate_model_path(settings.model_path), configuration), progress=progress)
            self._results.put(("result", result))
            self._progress = (1, "Complete: CSV and annotated video are ready")
        except Exception as error:
            self._progress = (0, "Cancelled" if self._cancel.is_set() else "Extraction failed")
            message = error.user_message() if callable(getattr(error, "user_message", None)) else str(error)
            self._results.put(("error", message))
        finally:
            self._controller = None

    def cancel_recorded(self) -> None:
        self._cancel.set()
        if self._controller:
            self._controller.request_cancel()

    def close(self) -> None:
        self._closed = True
        self.cancel_recorded()

    @property
    def busy(self) -> bool:
        """Report background work that must finish before orderly application shutdown."""
        return any(job is not None and job.is_alive() for job in (self._job, self._export_job))
