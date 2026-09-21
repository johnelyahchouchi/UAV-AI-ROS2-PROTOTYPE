"""Cross-process journal and dashboard reporting without weights or hardware."""

from pathlib import Path
import json
import queue
import sys
import threading
import sqlite3
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "01_WINDOWS_AI", ROOT / "08_MODEL_UNCERTAINTY/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analysis_session import AnalysisSession, EmbeddedPreview, extraction_record
from control_center.analysis_workspace import metric_series
from live_tester.continuous_uncertainty import ContinuousUncertaintyController, MethodSnapshot
from live_tester.uncertainty_adapter import RobustnessInspector
from uav_uncertainty.analysis import analyze_image
from uav_uncertainty.detection import Detection
from uav_uncertainty.mc_dropout_v2 import MCDOV2Config, run_mcdo_frame


FRAME = np.full((24, 32, 3), 70, dtype=np.uint8)
BOX = Detection(0, "box", .8, (3, 4, 20, 22))


def make_record(method="v1", empty=False):
    detector = SimpleNamespace(detect=lambda pixels: [] if empty else [BOX])
    if method == "v1":
        analysis = analyze_image(FRAME, detector, sample_count=5)
    else:
        runner = SimpleNamespace(prepare=lambda pixels: SimpleNamespace(detect_pass=lambda: [] if empty else [BOX]))
        analysis = run_mcdo_frame(FRAME, runner, MCDOV2Config(sample_count=4))
    view = SimpleNamespace(analysis=analysis)
    state = MethodSnapshot(method, method.upper(), "scope", "READY", "READY", frame_number=12,
                           sampled_at=15, sampled_utc="2026-09-20T10:00:00+00:00", duration_ms=123)
    return extraction_record(view, state, started_at=10)


@pytest.mark.parametrize("method", ["v1", "v2"])
def test_extraction_preserves_every_sample_detection_and_complete_analysis(method):
    record = make_record(method)
    assert record["elapsed_seconds"] == 5
    assert record["duration_ms"] == 123
    assert len(record["samples"]) == (6 if method == "v1" else 4)
    assert all(sample["detections"][0]["bbox"] == BOX.bbox for sample in record["samples"])
    assert record["metrics"]["persistence"] == 1
    assert record["metrics"]["agreement"] == 1
    assert record["metrics"]["iou"] == 1
    if method == "v1":
        assert record["samples"][3]["family"] == "gaussian_blur"
        assert "sigma" in record["samples"][3]["parameters"]
    else:
        assert all(sample["parameters"] == {} for sample in record["samples"])


@pytest.mark.parametrize("method", ["v1", "v2"])
def test_no_detections_are_undefined_graph_values_not_perfect_or_zero_scores(method):
    record = make_record(method, empty=True)
    assert record["metrics"] == {"target_count": 0, "persistence": None, "agreement": None, "iou": None}
    assert metric_series([record], "persistence") == [(5, None)]
    assert metric_series([{**record, "metrics": {}}], "iou") == [(5, None)]


def test_session_readers_see_complete_records_and_export_all_fields(tmp_path):
    writer = AnalysisSession(tmp_path / "session", create=True)
    reader = AnalysisSession(writer.directory)
    writer.set_metadata(v1_model="fixture.pt", v2_availability="unavailable")
    expected = make_record()
    errors = []

    def produce():
        try:
            for index in range(20):
                writer.append({**expected, "frame": index})
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=produce)
    worker.start()
    sequence, results = 0, []
    while worker.is_alive():
        batch = reader.read_after(sequence, limit=3)
        if batch:
            sequence = batch[-1][0]
            results.extend(batch)
    worker.join()
    results.extend(reader.read_after(sequence, limit=100))
    assert not errors
    assert [record["frame"] for _, record in results] == list(range(20))
    assert reader.read_record(1)["samples"][0]["detections"][0]["class_name"] == "box"
    exported = tmp_path / "full.json"
    reader.export_json(exported)
    data = json.loads(exported.read_text())
    assert len(data["records"]) == 20
    assert data["metadata"]["v1_model"] == "fixture.pt"
    with pytest.raises(ValueError):
        writer.append({"bad": float("nan")})
    assert len(reader.read_after(0, limit=100)) == 20


def test_commands_are_ordered_restricted_and_not_replayed(tmp_path):
    session = AnalysisSession(tmp_path, create=True)
    assert session.next_command(0) == (0, -1)
    session.command("z")
    session.command("p")
    assert session.next_command(0) == (1, ord("z"))
    assert session.next_command(1) == (2, ord("p"))
    assert session.next_command(2) == (2, -1)
    with pytest.raises(ValueError):
        session.command("arbitrary command")


def test_embedded_preview_publishes_owned_frame_and_accepts_zoom(tmp_path):
    session = AnalysisSession(tmp_path, create=True)
    preview = EmbeddedPreview(cv2, session)
    frame = FRAME.copy()
    try:
        preview.imshow("fixture", frame)
        frame[:] = 200
        deadline = time.monotonic() + 3
        while not (tmp_path / "preview.jpg").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        pixels = cv2.imread(str(tmp_path / "preview.jpg"))
        assert pixels is not None and np.all(pixels == 70)
        session.command("z")
        assert preview.waitKey(1) == ord("z")
        assert preview.waitKey(1) == -1
    finally:
        preview.destroyAllWindows()
    assert not preview._worker.is_alive()


def test_journal_failure_does_not_discard_scientific_result():
    inspector = RobustnessInspector(SimpleNamespace(detect=lambda pixels: []), cv2_module=cv2, sample_count=1)

    def failed_sink(view, state):
        raise OSError("disk full fixture")

    controller = ContinuousUncertaintyController(inspector, None, interval_seconds=1, result_sink=failed_sink)
    try:
        controller.submit(FRAME, frame_number=1)
        controller._thread.join(3)
        state = controller.snapshot().v1
        assert state.state == "READY" and "EXPORT FAILED" in state.status
        assert state.lines and state.completed == 2
    finally:
        controller.close()


def test_opening_missing_session_does_not_create_files(tmp_path):
    with pytest.raises(ValueError):
        AnalysisSession(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_connections_close_on_success_and_error(tmp_path, monkeypatch):
    connections = []
    original = sqlite3.connect

    class Tracked(sqlite3.Connection):
        was_closed = False

        def close(self):
            self.was_closed = True
            return super().close()

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs, factory=Tracked)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    session = AnalysisSession(tmp_path, create=True)
    session.append(make_record())
    session.read_after(0)
    with pytest.raises(RuntimeError):
        with session._connect() as connection:
            raise RuntimeError("fixture")
    assert connections and all(connection.was_closed for connection in connections)


def test_failed_export_preserves_previous_file_and_cleans_its_temporary(tmp_path, monkeypatch):
    import analysis_session
    session = AnalysisSession(tmp_path / "session", create=True)
    session.append(make_record())
    output = tmp_path / "report.json"
    output.write_text("previous report")
    monkeypatch.setattr(analysis_session.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("fixture disk error")))
    with pytest.raises(OSError):
        session.export_json(output)
    assert output.read_text() == "previous report"
    assert not list(tmp_path.glob(".analysis-export-*"))
    with pytest.raises(ValueError):
        session.export_json(session.path)
    assert len(session.read_after(0)) == 1


def test_empty_model_launch_guides_user_before_opening_video_dialog(monkeypatch):
    from control_center.app import UAVPrototypeControlCenter
    from control_center.configuration import LiveTesterSettings
    import control_center.app as app
    selected, messages = [], []
    fake = SimpleNamespace(processes={}, current_live_settings=lambda: LiveTesterSettings(model_path=""),
        notebook=SimpleNamespace(select=selected.append), live_tab="live", root=None)
    monkeypatch.setattr(app.messagebox, "showinfo", lambda title, message, **kw: messages.append(message))
    monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **kw: pytest.fail("Must select model first"))
    UAVPrototypeControlCenter.launch_live(fake)
    assert selected == ["live"]
    assert "Base detector (.pt)" in messages[0] and "MP4" in messages[0]


def test_recorded_workspace_reuses_backend_settings_and_registry(tmp_path, monkeypatch):
    dashboard_source = ROOT / "01_WINDOWS_AI/model_test_dashboard/src"
    monkeypatch.syspath_prepend(str(dashboard_source))
    from control_center.analysis_workspace import AnalysisWorkspace
    from control_center.configuration import LiveTesterSettings
    from uav_model_dashboard import video_processor, model_manager, output_manager
    import uav_security.model_integrity as integrity
    video, model, registry = (tmp_path / name for name in ("fixture.mp4", "fixture.pt", "registry.csv"))
    for path in (video, model, registry):
        path.touch()
    seen = []
    monkeypatch.setattr(integrity, "verify_trusted_model", lambda path, reg: seen.append((path, reg)) or "a"*64)

    class Processor:
        def __init__(self, manager, output, controller):
            self.manager = manager

        def process(self, request, *, progress):
            self.manager._verifier(request.model_path)
            assert request.settings.device.value == "GPU 0"
            assert request.settings.confidence == .4
            assert request.video_path == video
            progress(.5, desc="Frame 1 of 2")
            return "fixture result"

    monkeypatch.setattr(video_processor, "VideoProcessor", Processor)
    monkeypatch.setattr(output_manager, "OutputManager", lambda root: object())
    fake = SimpleNamespace(_cancel=threading.Event(), _controller=None, _results=queue.Queue())
    settings = LiveTesterSettings(model_path=str(model), registry_path=str(registry), device="cuda:0", confidence="0.4")
    AnalysisWorkspace._run_recorded(fake, settings, str(video))
    assert seen == [(model, registry)]
    assert fake._results.get_nowait() == ("result", "fixture result")
    assert fake._controller is None and fake._progress[0] == 1


def test_preview_read_releases_file_before_decode(tmp_path, monkeypatch):
    from control_center import analysis_workspace as workspace
    path = tmp_path / "preview.jpg"
    workspace.Image.new("RGB", (20, 10), "red").save(path)
    original = workspace.Image.open

    def open_image(source):
        # The shared file can be removed before Pillow starts decoding.
        path.unlink()
        return original(source)

    monkeypatch.setattr(workspace.Image, "open", open_image)
    result = workspace.read_preview(path)
    assert result.size == (20, 10) and result.getpixel((0, 0))[0] > 240


def test_preview_lock_retries_without_changing_session_status(tmp_path, monkeypatch):
    from control_center import analysis_workspace as workspace
    path = tmp_path / "preview.jpg"
    workspace.Image.new("RGB", (20, 10)).save(path)
    messages, configured = [], []
    fake = SimpleNamespace(session=SimpleNamespace(directory=tmp_path), _image_stamp=None,
        _image="previous", preview_status=SimpleNamespace(set=messages.append),
        preview=SimpleNamespace(winfo_width=lambda: 640, winfo_height=lambda: 360,
                                configure=lambda **kwargs: configured.append(kwargs)))
    original = workspace.read_preview
    monkeypatch.setattr(workspace, "read_preview", lambda path: (_ for _ in ()).throw(PermissionError("locked")))
    workspace.AnalysisWorkspace.refresh_preview(fake)
    assert fake._image == "previous" and fake._image_stamp is None
    assert "retrying" in messages[-1] and not configured
    monkeypatch.setattr(workspace, "read_preview", original)
    monkeypatch.setattr(workspace.ImageTk, "PhotoImage", lambda image: "recovered")
    workspace.AnalysisWorkspace.refresh_preview(fake)
    assert fake._image == "recovered" and configured
    assert "retrying" not in messages[-1]
