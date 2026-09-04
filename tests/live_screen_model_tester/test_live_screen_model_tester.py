from __future__ import annotations

from datetime import datetime
import importlib.util
import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "01_WINDOWS_AI" / "apps" / "live_screen_model_tester.py"
SPEC = importlib.util.spec_from_file_location("live_screen_model_tester", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tester = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tester
SPEC.loader.exec_module(tester)


def test_validate_manual_region_accepts_region_inside_monitor() -> None:
    monitor = tester.CaptureRegion(0, 0, 1920, 1080)
    region = tester.CaptureRegion(100, 200, 640, 480)

    assert tester.validate_manual_region(region, [monitor]) == region


@pytest.mark.parametrize(
    "region",
    [
        tester.CaptureRegion(-1, 0, 10, 10),
        tester.CaptureRegion(0, -1, 10, 10),
        tester.CaptureRegion(0, 0, 0, 10),
        tester.CaptureRegion(0, 0, 10, -1),
        tester.CaptureRegion(1800, 900, 200, 200),
    ],
)
def test_validate_manual_region_rejects_invalid_bounds(region) -> None:
    with pytest.raises(tester.TesterError):
        tester.validate_manual_region(region, [tester.CaptureRegion(0, 0, 1920, 1080)])


def test_monitor_selection_is_one_based() -> None:
    monitors = (
        tester.CaptureRegion(0, 0, 1920, 1080),
        tester.CaptureRegion(1920, 0, 2560, 1440),
    )

    assert tester.monitor_by_number(monitors, 2) == monitors[1]
    with pytest.raises(tester.TesterError, match="1 through 2"):
        tester.monitor_by_number(monitors, 3)


def test_class_filter_supports_all_multiple_and_tank_only() -> None:
    assert tester.parse_class_filter("all") is None
    assert tester.parse_class_filter("military_tank, Cargo Truck") == frozenset(
        {"military_tank", "cargo_truck"}
    )
    assert tester.parse_class_filter("all", tank_only=True) == frozenset(
        {"military_tank"}
    )

    detections = (
        tester.DetectionResult(0, "military_tank", 0.9, (1, 1, 5, 5)),
        tester.DetectionResult(1, "cargo-truck", 0.8, (2, 2, 6, 6)),
    )
    filtered = tester.filter_detections(detections, frozenset({"cargo_truck"}))
    assert [item.class_name for item in filtered] == ["cargo-truck"]


def test_resolve_model_path_uses_cli_then_environment(tmp_path: Path) -> None:
    cli_model = tmp_path / "cli.pt"
    env_model = tmp_path / "env.pt"
    cli_model.write_bytes(b"cli")
    env_model.write_bytes(b"env")

    assert (
        tester.resolve_model_path(str(cli_model), {"UAV_MODEL_PATH": str(env_model)})
        == cli_model
    )
    assert tester.resolve_model_path(None, {"UAV_MODEL_PATH": str(env_model)}) == env_model
    with pytest.raises(tester.TesterError, match="Provide --model"):
        tester.resolve_model_path(None, {})


def test_discover_first_mp4_prefers_newest_file_in_first_matching_directory(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    older = first_directory / "older.mp4"
    newer = first_directory / "newer.MP4"
    ignored = first_directory / "newest.mov"
    later_directory_video = second_directory / "latest.mp4"
    for path in (older, newer, ignored, later_directory_video):
        path.touch()
    os.utime(older, ns=(100, 100))
    os.utime(newer, ns=(200, 200))
    os.utime(ignored, ns=(400, 400))
    os.utime(later_directory_video, ns=(300, 300))

    selected = tester.discover_first_mp4([first_directory, second_directory])

    assert selected == newer.resolve()


def test_resolve_video_path_supports_environment_and_auto_detection(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured.mp4"
    automatic = tmp_path / "automatic.mp4"
    configured.touch()
    automatic.touch()

    assert tester.resolve_video_path(
        None,
        auto_detect=True,
        environment={"UAV_VIDEO_PATH": str(configured)},
        search_directories=[tmp_path],
    ) == configured.resolve()
    assert tester.resolve_video_path(
        None,
        auto_detect=True,
        environment={},
        search_directories=[tmp_path],
    ) == automatic.resolve()


def test_video_picker_returns_selected_mp4_and_destroys_root(tmp_path: Path) -> None:
    video = tmp_path / "selected video.mp4"
    video.touch()

    class FakeRoot:
        withdrawn = False
        topmost = False
        destroyed = False

        def withdraw(self):
            self.withdrawn = True

        def attributes(self, name, value):
            assert name == "-topmost"
            self.topmost = value

        def destroy(self):
            self.destroyed = True

    root = FakeRoot()

    def choose(**options):
        assert options["parent"] is root
        assert options["filetypes"] == (("MP4 video", "*.mp4"),)
        return str(video)

    selected = tester.select_video_with_dialog(
        tk_factory=lambda: root, askopenfilename=choose
    )

    assert selected == video.resolve()
    assert root.withdrawn is True
    assert root.topmost is True
    assert root.destroyed is True


def test_video_picker_cancel_is_a_clear_error() -> None:
    class FakeRoot:
        def withdraw(self):
            return None

        def attributes(self, *args):
            return None

        def destroy(self):
            return None

    with pytest.raises(tester.TesterError, match="cancelled"):
        tester.select_video_with_dialog(
            tk_factory=FakeRoot, askopenfilename=lambda **options: ""
        )


def test_video_source_restarts_at_eof_when_looping(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.touch()

    class FakeCapture:
        def __init__(self):
            self.index = 0
            self.frames = ["frame-1", "frame-2"]
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def set(self, key, value):
            self.index = int(value)
            return True

        def release(self):
            self.released = True

    capture = FakeCapture()

    class FakeCV2:
        CAP_PROP_POS_FRAMES = 1

        @staticmethod
        def VideoCapture(path):
            assert Path(path) == video_path
            return capture

    metadata = tester.VideoMetadata(video_path, 100, 50, 30.0, 2)
    with tester.VideoFileSource(metadata, loop=True, cv2_module=FakeCV2) as source:
        assert [source.read(), source.read(), source.read()] == [
            "frame-1",
            "frame-2",
            "frame-1",
        ]

    assert capture.released is True


def test_video_mode_rejects_screen_region_options() -> None:
    args = tester.build_parser().parse_args(["--auto-video", "--select-region"])

    with pytest.raises(tester.TesterError, match="screen-region"):
        tester.validate_numeric_options(args)


def test_video_preview_processes_frame_and_quits_cleanly(tmp_path: Path) -> None:
    class FakeFrame:
        def copy(self):
            return FakeFrame()

    class FakeSource:
        metadata = tester.VideoMetadata(tmp_path / "video.mp4", 100, 50, 30.0, 1)
        opened = False
        closed = False

        def __enter__(self):
            self.opened = True
            return self

        def read(self):
            return FakeFrame()

        def __exit__(self, *args):
            self.closed = True

    class FakeProcessor:
        region = tester.CaptureRegion(0, 0, 100, 50)
        renderer = None
        tracker = tester.PerformanceTracker()
        model_name = "best.pt"
        device = "CPU"
        source_label = "video.mp4"

        def process(self, frame, **kwargs):
            return tester.FrameOutcome(frame, (), tester.FrameMetrics())

    class FakeCV2:
        WINDOW_NORMAL = 0
        WND_PROP_VISIBLE = 1
        shown = 0
        destroyed = False

        @classmethod
        def namedWindow(cls, *args):
            return None

        @classmethod
        def resizeWindow(cls, *args):
            return None

        @classmethod
        def imshow(cls, *args):
            cls.shown += 1

        @classmethod
        def waitKey(cls, *args):
            return ord("q")

        @classmethod
        def getWindowProperty(cls, *args):
            return 1

        @classmethod
        def destroyAllWindows(cls):
            cls.destroyed = True

    timestamps = iter([1.0, 1.0, 1.001])
    source = FakeSource()
    tester.run_video_preview(
        source,
        FakeProcessor(),
        tester.ScreenshotStore(tmp_path),
        max_fps=0,
        cv2_module=FakeCV2,
        clock=lambda: next(timestamps),
    )

    assert source.opened is True
    assert source.closed is True
    assert FakeCV2.shown == 1
    assert FakeCV2.destroyed is True


def test_resolve_device_auto_falls_back_and_explicit_cuda_fails_cleanly() -> None:
    assert tester.resolve_device("auto", cuda_available=False, cuda_device_count=0) == "cpu"
    assert tester.resolve_device("auto", cuda_available=True, cuda_device_count=1) == "0"
    assert tester.resolve_device("cuda:0", cuda_available=True, cuda_device_count=1) == "0"
    with pytest.raises(tester.TesterError, match="CUDA was requested"):
        tester.resolve_device("0", cuda_available=False, cuda_device_count=0)
    assert tester.device_display_name("cpu") == "CPU"
    assert tester.device_display_name("1") == "CUDA:1"


def test_discover_monitors_excludes_virtual_aggregate() -> None:
    class FakeGrabber:
        monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monitors = tester.discover_monitors(FakeGrabber)

    assert monitors == (
        tester.CaptureRegion(0, 0, 1920, 1080),
        tester.CaptureRegion(1920, 0, 1920, 1080),
    )


def test_performance_tracker_uses_rolling_averages_and_displayed_interval() -> None:
    tracker = tester.PerformanceTracker(window_size=2)
    tracker.record(capture_ms=2, inference_ms=8, total_ms=12, frame_timestamp=1.0)
    tracker.record(capture_ms=4, inference_ms=12, total_ms=18, frame_timestamp=1.1)
    metrics = tracker.record(
        capture_ms=6, inference_ms=16, total_ms=24, frame_timestamp=1.3
    )

    assert metrics.capture_ms == pytest.approx(5.0)
    assert metrics.inference_ms == pytest.approx(14.0)
    assert metrics.total_ms == pytest.approx(21.0)
    assert metrics.fps == pytest.approx(1 / 0.15)


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ((10, 20, 90, 80), (10, 20, 90, 80)),
        ((-10, -20, 120, 130), (0, 0, 99, 99)),
        ((10, 10, 5, 20), None),
        ((10, 10, float("nan"), 20), None),
        ((-20, -20, -1, -1), None),
        ((1, 2, 3), None),
    ],
)
def test_clip_bbox_is_safe(coordinates, expected) -> None:
    assert tester.clip_bbox(coordinates, 100, 100) == expected


def test_screenshot_paths_are_created_and_never_reused(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "screenshots"
    store = tester.ScreenshotStore(output)
    fixed_time = datetime(2026, 9, 4, 12, 34, 56, 123456)
    first = store.next_path(fixed_time)
    output.mkdir(parents=True)
    first.write_bytes(b"existing")
    second = store.next_path(fixed_time)

    assert first.name == "live_screen_20260904_123456_123456.png"
    assert second.name == "live_screen_20260904_123456_123456_1.png"
    assert second != first


def test_frame_processor_accepts_fake_frame_detector_and_renderer() -> None:
    class FakeDetector:
        def detect(self, frame):
            assert frame == "fake-frame"
            return [tester.DetectionResult(0, "military_tank", 0.95, (1, 1, 4, 4))]

    class FakeRenderer:
        def draw_detections(self, frame, detections):
            return {"frame": frame, "count": len(detections)}

        def draw_hud(self, frame, **kwargs):
            frame["hud"] = kwargs["detection_count"]
            return frame

    times = iter([10.005, 10.015, 10.020])
    processor = tester.FrameProcessor(
        FakeDetector(),
        FakeRenderer(),
        allowed_classes=frozenset({"military_tank"}),
        model_name="best.pt",
        region=tester.CaptureRegion(0, 0, 100, 100),
        device="cpu",
        clock=lambda: next(times),
    )
    outcome = processor.process(
        "fake-frame",
        capture_ms=5.0,
        frame_started_at=10.0,
        frame_timestamp=10.0,
        hud_visible=True,
    )

    assert len(outcome.detections) == 1
    assert outcome.annotated_frame == {"frame": "fake-frame", "count": 1, "hud": 1}
    assert outcome.metrics.inference_ms == pytest.approx(10.0)
    assert outcome.metrics.total_ms == pytest.approx(20.0)


def test_yolo_detector_verifies_before_loading(tmp_path: Path) -> None:
    calls = []
    model_path = tmp_path / "trusted.pt"
    model_path.write_bytes(b"placeholder")

    class FakeModel:
        names = {}

    def verifier(path, registry):
        calls.append(("verify", path, registry))
        return "a" * 64

    def loader(path, *, registry_path):
        calls.append(("load", path, registry_path))
        return FakeModel()

    detector = tester.YoloDetector(
        model_path,
        confidence=0.25,
        iou=0.45,
        image_size=640,
        device="cpu",
        verifier=verifier,
        model_loader=loader,
    )

    assert [call[0] for call in calls] == ["verify", "load"]
    assert detector.model_sha256 == "a" * 64


def test_preview_position_prefers_another_monitor() -> None:
    captured = tester.CaptureRegion(0, 0, 1920, 1080)
    second = tester.CaptureRegion(1920, 0, 1920, 1080)

    left, top, overlaps = tester.choose_preview_position(captured, [captured, second], 800, 600)

    assert (left, top) == (1940, 20)
    assert overlaps is False


def test_batch_launcher_is_dynamic_and_opens_video_picker() -> None:
    launcher = (
        PROJECT_ROOT
        / "01_WINDOWS_AI"
        / "launchers"
        / "Start_Live_Screen_Tester.bat"
    )
    content = launcher.read_text(encoding="utf-8")

    assert "%~dp0..\\apps\\live_screen_model_tester.py" in content
    assert "--select-video --loop-video" in content
    assert "UAV_YOLO_PYTHON" in content
    assert "UAV_MODEL_PATH" in content
    assert "..\\UAV_YOLO_ENV\\Scripts\\python.exe" in content
    assert (
        "05_TRAINING\\detection_runs\\military_kaggle_yolov8s_v1"
        "\\weights\\best.pt"
    ) in content
    assert "powershell.exe" not in content.lower()
    assert "-ExecutionPolicy Bypass" not in content
    assert "C:\\Users\\" not in content
    assert "OneDrive" not in content
