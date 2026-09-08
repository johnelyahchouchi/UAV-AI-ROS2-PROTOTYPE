from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = PROJECT_ROOT / "01_WINDOWS_AI"
UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
for search_path in (WINDOWS_AI_ROOT, UNCERTAINTY_SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from live_tester import (  # noqa: E402
    CaptureRegion,
    DetectionResult,
    FontResolver,
    FrameMetrics,
    FrameOutcome,
    MCDOV2LiveInspector,
    PerformanceTracker,
    PillowCanvas,
    Rect,
    RobustnessInspector,
    ScreenshotStore,
    VideoMetadata,
    calculate_presentation_layout,
    mcdo_frame_summary,
    presentation_canvas_size,
    render_method_selector,
    run_video_preview,
)
from uav_uncertainty.detection import Detection  # noqa: E402
from uav_uncertainty.mc_dropout_v2 import (  # noqa: E402
    MCDOTargetCluster,
    MCDOV2Config,
    _interpret_dimensions,
    calculate_mcdo_target,
)


def _capture_text(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []
    original = PillowCanvas.text

    def recording_text(self, value, *args, **kwargs):
        captured.append(str(value))
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(PillowCanvas, "text", recording_text)
    return captured


def _mcdo_target(
    pass_indices: tuple[int, ...],
    *,
    sample_count: int = 4,
    class_name: str = "military_tank",
):
    cluster = MCDOTargetCluster(cluster_id=1)
    for index in pass_indices:
        cluster.add(index, Detection(0, class_name, 0.85, (10.0, 12.0, 60.0, 52.0)))
    return calculate_mcdo_target(cluster, sample_count)


class _SequenceSession:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def detect_pass(self):
        return next(self.outputs)


class _SequenceRunner:
    model_sha256 = "a" * 64

    def __init__(self, outputs):
        self.outputs = outputs

    def prepare(self, exact_frame):
        return _SequenceSession(self.outputs)


def test_font_resolver_falls_back_when_no_truetype_font_loads() -> None:
    fallback = ImageFont.load_default()

    def unavailable(*args, **kwargs):
        raise OSError("font unavailable")

    resolver = FontResolver(
        regular_candidates=("missing-regular.ttf",),
        bold_candidates=("missing-bold.ttf",),
        truetype_loader=unavailable,
        default_loader=lambda **kwargs: fallback,
    )

    assert resolver.font(16) is fallback
    assert resolver.font(18, bold=True) is fallback
    assert resolver.resolved_sources == {False: "Pillow default", True: "Pillow default"}

    canvas = PillowCanvas(np.zeros((40, 240, 3), dtype=np.uint8), fonts=resolver)
    canvas.text("Fallback — σ ± …", 2, 2, size=14)
    assert canvas.render().shape == (40, 240, 3)


def test_default_font_resolution_uses_a_scalable_font() -> None:
    resolver = FontResolver()

    assert resolver.font(18).getbbox("Readable presentation text")[2] > 0
    assert resolver.font(18, bold=True).getbbox("Bold status")[2] > 0
    assert resolver.resolved_sources[False]
    assert resolver.resolved_sources[True]


def test_translucent_presentation_fills_are_composited_before_render() -> None:
    canvas = PillowCanvas(np.zeros((20, 20, 3), dtype=np.uint8))
    canvas.rectangle(Rect(0, 0, 20, 20), fill=(100, 160, 220, 64))

    rendered = canvas.render()

    assert tuple(int(value) for value in rendered[10, 10]) == (55, 40, 25)


@pytest.mark.parametrize("width,height", [(1280, 720), (1600, 900), (1920, 1080)])
def test_layout_is_responsive_and_target_cards_never_overlap(width, height) -> None:
    layout = calculate_presentation_layout(width, height, 2)

    assert layout.width / layout.height == pytest.approx(16 / 9)
    assert layout.image.width / (layout.image.width + layout.summary.width) == pytest.approx(
        0.59, abs=0.02
    )
    assert not layout.image.overlaps(layout.summary)
    assert not layout.summary.overlaps(layout.cards[0])
    assert not layout.cards[0].overlaps(layout.cards[1])
    assert all(card.bottom <= layout.footer.y for card in layout.cards)
    assert layout.footer.bottom <= height


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((720, 1280, 3), (1280, 720)),
        ((900, 1600, 3), (1600, 900)),
        ((1080, 1920, 3), (1920, 1080)),
        ((480, 640, 3), (1280, 720)),
    ],
)
def test_presentation_canvas_selects_supported_16_by_9_size(shape, expected) -> None:
    assert presentation_canvas_size(shape) == expected


def test_method_selector_uses_requested_copy_and_complete_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _capture_text(monkeypatch)
    frozen = np.full((720, 1280, 3), 96, dtype=np.uint8)
    original = frozen.copy()

    rendered = render_method_selector(
        frozen, v1_sample_count=10, v2_sample_count=20
    )

    assert rendered.shape == frozen.shape
    assert np.array_equal(frozen, original)
    assert not np.array_equal(rendered, frozen)
    assert "UNCERTAINTY METHOD" in text
    assert "Same frozen frame" in text
    assert "V1 — INPUT ROBUSTNESS" in text
    assert "1 clean + 10 perturbed inputs" in text
    assert "V2 — MC DROPOUT" in text
    assert "20 stochastic model passes" in text
    assert "Neither method is a calibrated probability of correctness." in text
    assert "ESC / SPACE / U — Resume" in text


def test_v1_screen_handles_multiple_targets_long_names_and_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _capture_text(monkeypatch)
    long_name = "tracked_armoured_vehicle_with_an_intentionally_long_presentation_label"

    class StableDetector:
        def detect(self, image):
            return [
                DetectionResult(
                    index,
                    long_name if index == 0 else f"class_{index}",
                    0.86,
                    (20.0 + 90 * index, 30.0, 80.0 + 90 * index, 95.0),
                )
                for index in range(3)
            ]

    inspector = RobustnessInspector(
        StableDetector(), cv2_module=object(), sample_count=2, seed=3
    )
    view = inspector.inspect(np.full((720, 1280, 3), 110, dtype=np.uint8))

    assert view.frame.shape == (720, 1280, 3)
    assert len(view.pages) == 2
    assert any(long_name in line for line in text)
    assert any("width/height σ" in line for line in text)
    assert any("Evidence share (not probability)" in line for line in text)


def test_v1_screen_handles_zero_and_perturbation_only_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoDetections:
        def detect(self, image):
            return []

    zero = RobustnessInspector(
        NoDetections(), cv2_module=object(), sample_count=2
    ).inspect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert zero.frame.shape == (720, 1280, 3)
    assert len(zero.pages) == 1

    text = _capture_text(monkeypatch)

    class VariantOnly:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            if self.calls == 1:
                return []
            return [DetectionResult(1, "variant_vehicle", 0.7, (8, 8, 28, 28))]

    variant = RobustnessInspector(
        VariantOnly(), cv2_module=object(), sample_count=2
    ).inspect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert variant.status == "PERTURBATION-ONLY DETECTIONS"
    assert any(line.startswith("VARIANT-ONLY") for line in text)
    assert any("Appeared only after input perturbation" in line for line in text)


def test_v1_input_sensitive_screen_is_not_presented_as_fully_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _capture_text(monkeypatch)

    class MissingVariant:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            if self.calls == 3:
                return []
            return [DetectionResult(0, "tank", 0.8, (5, 5, 25, 25))]

    view = RobustnessInspector(
        MissingVariant(), cv2_module=object(), sample_count=2
    ).inspect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert view.status == "UNSTABLE / REVIEW"
    assert "REVIEW" in text
    assert any("FRAME REVIEW" in line for line in text)


def test_mcdo_interpretation_names_only_dimensions_that_need_attention() -> None:
    assert _interpret_dimensions(
        "UNSTABLE / REVIEW", "STABLE", "STABLE"
    ) == "Existence unstable; classification and localization remain stable."
    assert _interpret_dimensions("STABLE", "UNCERTAIN", "STABLE") == (
        "Classification uncertain; existence and localization remain stable."
    )
    assert _interpret_dimensions("STABLE", "STABLE", "UNSTABLE / REVIEW") == (
        "Localization unstable; existence and classification remain stable."
    )


def test_v2_frame_summary_explains_the_unstable_dimension() -> None:
    stable = _mcdo_target((0, 1, 2, 3))
    unstable_existence = _mcdo_target((0,))

    stable_count, review_count, overall, reason = mcdo_frame_summary(
        (stable, stable, unstable_existence)
    )

    assert (stable_count, review_count, overall) == (2, 1, "REVIEW")
    assert reason == "FRAME REVIEW — 1 of 3 targets shows unstable existence."
    assert unstable_existence.interpretation == (
        "Existence unstable; classification and localization remain stable."
    )


def test_v2_screen_handles_stable_multiple_targets_and_long_class_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _capture_text(monkeypatch)
    long_name = "military_vehicle_with_a_long_descriptive_class_name_for_projection"
    outputs = [
        [
            Detection(index, long_name if index == 0 else f"target_class_{index}", 0.9, (20 + index * 100, 20, 80 + index * 100, 90))
            for index in range(3)
        ]
        for _ in range(2)
    ]
    inspector = MCDOV2LiveInspector(
        _SequenceRunner(outputs),
        cv2_module=object(),
        config=MCDOV2Config(sample_count=2),
    )

    view = inspector.inspect(np.full((1080, 1920, 3), 80, dtype=np.uint8))

    assert view.frame.shape == (1080, 1920, 3)
    assert len(view.pages) == 2
    assert "V2 — MC DROPOUT UNCERTAINTY" in text
    assert any(long_name in line for line in text)
    assert any("Width/height σ" in line for line in text)
    assert any("Evidence share (not probability)" in line for line in text)
    assert any("FRAME STABLE" in line for line in text)


def test_v2_screen_handles_zero_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _capture_text(monkeypatch)
    inspector = MCDOV2LiveInspector(
        _SequenceRunner([[], []]),
        cv2_module=object(),
        config=MCDOV2Config(sample_count=2),
    )

    view = inspector.inspect(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert view.frame.shape == (720, 1280, 3)
    assert len(view.pages) == 1
    assert "NO STOCHASTIC DETECTIONS" in text
    assert any("FRAME REVIEW" in line for line in text)


def test_runtime_can_page_and_save_the_visible_uncertainty_card(tmp_path: Path) -> None:
    source_frame = np.full((8, 8, 3), 7, dtype=np.uint8)

    class Source:
        metadata = VideoMetadata(tmp_path / "video.mp4", 8, 8, 30.0, 2)

        def __init__(self):
            self.reads = 0

        def __enter__(self):
            return self

        def read(self):
            self.reads += 1
            return source_frame.copy()

        def __exit__(self, *args):
            return None

    class Renderer:
        def draw_hud(self, frame, **kwargs):
            return frame

    class Processor:
        region = CaptureRegion(0, 0, 8, 8)
        renderer = Renderer()
        tracker = PerformanceTracker()
        model_name = "model.pt"
        device = "CPU"
        source_label = "video.mp4"
        last_inspection = None

        def process(self, frame, **kwargs):
            return FrameOutcome(frame.copy(), (), FrameMetrics())

    class View:
        status = "STABLE IN V1 TEST"

        def __init__(self):
            self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
            self.pages = (
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.full((8, 8, 3), 200, dtype=np.uint8),
            )

    class Inspector:
        def render_working(self, frame):
            return frame.copy()

        def inspect(self, frame):
            return View()

    class FakeCV2:
        WINDOW_NORMAL = 0
        WND_PROP_VISIBLE = 1
        keys = iter((ord("u"), -1, ord("d"), ord("s"), ord("p"), ord("q")))
        saved_pixels: list[int] = []

        namedWindow = staticmethod(lambda *args: None)
        resizeWindow = staticmethod(lambda *args: None)
        imshow = staticmethod(lambda *args: None)
        getWindowProperty = staticmethod(lambda *args: 1)
        destroyAllWindows = staticmethod(lambda: None)

        @classmethod
        def waitKey(cls, *args):
            return next(cls.keys)

        @classmethod
        def imwrite(cls, path, frame):
            cls.saved_pixels.append(int(frame[0, 0, 0]))
            return True

    source = Source()
    timestamps = iter(index / 1000 for index in range(100))
    run_video_preview(
        source,
        Processor(),
        ScreenshotStore(tmp_path),
        max_fps=0,
        cv2_module=FakeCV2,
        uncertainty_inspector=Inspector(),
        clock=lambda: next(timestamps),
        sleeper=lambda delay: None,
    )

    assert source.reads == 2
    assert FakeCV2.saved_pixels == [200]
