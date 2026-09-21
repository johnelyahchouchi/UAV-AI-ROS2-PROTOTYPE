"""Explainability provenance and non-blocking playback, without model weights."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "01_WINDOWS_AI", ROOT / "08_MODEL_UNCERTAINTY/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from live_tester.continuous_uncertainty import ContinuousUncertaintyController, ContinuousWorkspaceRenderer
from live_tester.domain import CaptureRegion, DetectionResult, FrameMetrics, FrameOutcome, VideoMetadata
from live_tester.explanation import MAX_VISUAL_SAMPLES, frame_preview
from live_tester.mc_dropout_adapter import MCDOV2LiveInspector
from live_tester.presentation_ui import PillowCanvas
from live_tester.runtime import PerformanceTracker, ScreenshotStore, run_video_preview
from live_tester.uncertainty_adapter import RobustnessInspector
from uav_uncertainty.analysis import analyze_image
from uav_uncertainty.detection import Detection
from uav_uncertainty.mc_dropout_v2 import MCDOV2Config, run_mcdo_frame
from uav_uncertainty.observations import SampleObservation


FRAME = np.random.default_rng(7).integers(0, 256, (61, 95, 3), dtype=np.uint8)
BOX = Detection(0, "box", 0.8, (20, 15, 70, 45))


class Runner:
    """Deterministic test double, never used by the application's real V2 loader."""

    def __init__(self, count=4):
        self.count = count
        self.inputs = []

    def prepare(self, frame):
        self.inputs.append(frame.copy())
        self.remaining = self.count
        return self

    def detect_pass(self):
        self.remaining -= 1
        return [BOX] if self.remaining % 2 else []


def inspectors():
    detector = SimpleNamespace(detect=lambda frame: [DetectionResult(0, "box", 0.8, (20, 15, 70, 45))])
    return (
        RobustnessInspector(detector, cv2_module=object(), sample_count=5),
        MCDOV2LiveInspector(Runner(), cv2_module=object(), config=MCDOV2Config(sample_count=4)),
    )


def finish(controller):
    controller._thread.join(timeout=5)
    assert not controller.snapshot().running, "worker did not finish"
    return controller.snapshot()


def test_v1_observer_reports_exact_inputs_outputs_and_parameters_without_changing_results():
    inputs, observed = [], []

    def detect(image):
        inputs.append(image.copy())
        return [BOX]

    detector = SimpleNamespace(detect=detect)
    analysis = analyze_image(FRAME, detector, sample_count=5, observer=observed.append)
    assert len(observed) == len(inputs) == 6
    for index, (sample, pixels) in enumerate(zip(observed, inputs)):
        assert sample.sample_index == index
        assert sample.total == 6
        assert sample.detections == analysis.samples[index].detections
        assert dict(sample.parameters) == analysis.samples[index].parameters
        assert np.array_equal(sample.image, pixels)
        assert not sample.image.flags.writeable
    assert observed[3].family == "gaussian_blur"
    assert not np.array_equal(observed[0].image, observed[3].image)
    # Presentation callbacks must not change the scientific result.
    assert analysis.to_dict() == analyze_image(FRAME, detector, sample_count=5).to_dict()


def test_v2_observer_emits_only_completed_passes_on_identical_pixels():
    observed = []
    analysis = run_mcdo_frame(FRAME, Runner(), MCDOV2Config(sample_count=4), observer=observed.append)
    assert len(observed) == 4
    assert [sample.detections for sample in observed] == list(analysis.samples)
    assert all(np.array_equal(sample.image, FRAME) for sample in observed)
    assert all(not sample.image.flags.writeable for sample in observed)
    assert all(sample.parameters == () and sample.family == "mc_dropout" for sample in observed)
    assert analysis.to_dict() == run_mcdo_frame(FRAME, Runner(), MCDOV2Config(sample_count=4)).to_dict()


def test_preview_pixels_are_bounded_readonly_and_do_not_mutate_source():
    source = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    preview = frame_preview(source)
    source[:] = 90
    for pixels in (preview.overview, preview.zoom):
        assert pixels.shape[:2] == (270, 480)
        assert np.all(pixels == 60)
        assert not pixels.flags.writeable
    odd = frame_preview(FRAME)
    assert odd.crop == (24 / 95, 15 / 61, 47 / 95, 30 / 61)


def test_controller_publishes_real_progress_and_clears_old_cycle_metadata():
    v1, v2 = inspectors()
    controller = ContinuousUncertaintyController(v1, v2, interval_seconds=1)
    assert controller.submit(FRAME, frame_number=8, now=10, captured_at=9.5, captured_utc="2026-09-18T12:00:00.000+00:00")
    first = finish(controller)
    assert first.v1.completed == first.v1.total == 6
    assert first.v2.completed == first.v2.total == 4
    assert first.v1.sampled_at == first.v2.sampled_at == 9.5
    assert first.v1.previews[3].family == "gaussian blur"

    entered, release = threading.Event(), threading.Event()

    class Blocked:
        def analyze(self, frame):
            entered.set()
            assert release.wait(5)
            return v1.analyze(frame)

    controller.v1_inspector = Blocked()
    try:
        assert controller.submit(FRAME, frame_number=9, now=12)
        assert entered.wait(2)
        state = controller.snapshot()
        for method in (state.v1, state.v2):
            assert method.frame_number == 9
            assert method.lines == method.previews == ()
            assert method.updated_at is None and method.duration_ms is None
            assert method.completed == 0
        assert not controller.submit(FRAME, frame_number=10, now=14)
    finally:
        release.set()
        finish(controller)
        controller.close()


def test_partial_failure_is_visible_and_does_not_prevent_v2():
    v1, v2 = inspectors()

    class Fails:
        def analyze_with_observer(self, frame, *, observer):
            observer(SampleObservation(frame, 0, 6, "clean_baseline", (), (BOX,)))
            raise RuntimeError("fixture inference failed")

    controller = ContinuousUncertaintyController(Fails(), v2, interval_seconds=1)
    try:
        controller.submit(FRAME, frame_number=1)
        state = finish(controller)
        assert state.v1.state == "FAILED" and state.v1.completed == 1
        assert "fixture inference failed" in state.v1.lines[0]
        assert state.v2.state == "READY"
    finally:
        controller.close()


def test_long_analysis_retains_bounded_visual_history_but_full_progress():
    v1, _ = inspectors()
    v2 = MCDOV2LiveInspector(Runner(40), cv2_module=object(), config=MCDOV2Config(sample_count=40))
    controller = ContinuousUncertaintyController(v1, v2, interval_seconds=1)
    try:
        controller.submit(FRAME, frame_number=1)
        state = finish(controller).v2
        assert state.completed == state.total == 40
        assert len(state.previews) == MAX_VISUAL_SAMPLES
        assert state.previews[0].sample_index == 8
    finally:
        controller.close()


@pytest.mark.parametrize("shape", [(360, 640), (900, 1600), (1080, 1920), (640, 360)])
@pytest.mark.parametrize("zoomed", [False, True])
def test_renderer_labels_real_sample_age_replay_and_view_only_zoom(monkeypatch, shape, zoomed):
    v1, v2 = inspectors()
    controller = ContinuousUncertaintyController(v1, v2, interval_seconds=1)
    controller.submit(FRAME, frame_number=8, now=10, captured_at=9.5, captured_utc="2026-09-18T12:00:00.000+00:00")
    snapshot = finish(controller)
    controller.close()
    text = []
    original = PillowCanvas.text

    def record(self, value, *args, **kwargs):
        text.append(str(value))
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(PillowCanvas, "text", record)
    renderer = ContinuousWorkspaceRenderer()
    renderer.zoomed = zoomed
    rendered = renderer.render(
        np.full((*shape, 3), 60, dtype=np.uint8), snapshot,
        metrics=FrameMetrics(), model_name="test fixture", device="CPU",
        source_label="synthetic fixture", detection_count=1, interval_seconds=2,
        paused=False, now=13, exact_inspection_available=True,
    )
    assert rendered.dtype == np.uint8
    assert any("Frame 8 | 2026-09-18 12:00:00.000 UTC" == line for line in text)
    assert any("Source age 3.5s" in line for line in text)
    assert any("Replay completed sample" in line for line in text)
    assert any("Completed 6/6" in line for line in text)
    assert any("Completed 4/4" in line for line in text)
    assert any("2x center / view only" in line for line in text) == zoomed


def test_video_keeps_advancing_and_zoom_works_while_analysis_is_blocked(tmp_path):
    entered, release = threading.Event(), threading.Event()
    v1, _ = inspectors()

    class Blocked:
        def analyze(self, frame):
            entered.set()
            assert release.wait(5)
            return v1.analyze(frame)

    class Source:
        metadata = VideoMetadata(tmp_path / "fixture.mp4", 95, 61, 30, 3)
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            self.reads += 1
            return FRAME.copy()

    class Processor:
        region = CaptureRegion(0, 0, 95, 61)
        model_name, device, source_label = "fixture", "CPU", "fixture"
        tracker = PerformanceTracker()

        def process(self, frame, **kwargs):
            return FrameOutcome(frame, (), FrameMetrics())

    class UI:
        WINDOW_NORMAL, WND_PROP_VISIBLE = 0, 1
        keys = iter((ord("z"), -1, ord("q")))
        window_sizes = []
        namedWindow = imshow = staticmethod(lambda *args: None)
        getWindowProperty = staticmethod(lambda *args: 1)
        destroyAllWindows = staticmethod(lambda: None)

        @classmethod
        def resizeWindow(cls, name, width, height):
            cls.window_sizes.append((width, height))

        @classmethod
        def waitKey(cls, *args):
            assert entered.wait(2)
            assert not release.is_set()
            return next(cls.keys)

    controller = ContinuousUncertaintyController(Blocked(), None, interval_seconds=1)
    renderer = ContinuousWorkspaceRenderer()
    source = Source()
    try:
        run_video_preview(
            source, Processor(), ScreenshotStore(tmp_path), max_fps=0, cv2_module=UI,
            continuous_controller=controller, continuous_renderer=renderer,
            sleeper=lambda delay: None,
        )
        assert source.reads == 3
        assert renderer.zoomed is True
        assert UI.window_sizes == [(1280, 720)]
        assert controller.snapshot().v2.state == "UNAVAILABLE"
    finally:
        release.set()
        finish(controller)
