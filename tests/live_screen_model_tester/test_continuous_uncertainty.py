from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = PROJECT_ROOT / "01_WINDOWS_AI"
for search_path in (PROJECT_ROOT, WINDOWS_AI_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import live_tester as tester  # noqa: E402


def _v1_view() -> SimpleNamespace:
    axis = SimpleNamespace(x=2.0, y=4.0)
    size = SimpleNamespace(x=5.0, y=7.0)
    target = SimpleNamespace(
        detection_persistence=0.91,
        class_agreement=0.88,
        mean_iou_to_reference=0.79,
        bbox_center_std_pixels=axis,
        bbox_size_std_pixels=size,
    )
    analysis = SimpleNamespace(
        baseline_metrics=(target,),
        perturbed_only_metrics=(),
        perturbation_count=10,
    )
    return SimpleNamespace(analysis=analysis, status="INPUT-SENSITIVE")


def _v2_view() -> SimpleNamespace:
    target = SimpleNamespace(
        persistence=0.95,
        winner_class_agreement=0.90,
        mean_reference_iou=0.82,
        winner_confidence_std=0.04,
    )
    analysis = SimpleNamespace(targets=(target,), sample_count=20)
    return SimpleNamespace(analysis=analysis, status="MODEL OUTPUT VARIABLE / REVIEW")


def test_continuous_controller_runs_v1_then_v2_without_queueing_stale_frames() -> None:
    release = threading.Event()
    calls: list[tuple[str, np.ndarray]] = []

    class V1:
        def analyze(self, frame):
            calls.append(("v1", frame.copy()))
            release.wait(timeout=2)
            return _v1_view()

    class V2:
        def analyze(self, frame):
            calls.append(("v2", frame.copy()))
            return _v2_view()

    original = np.full((24, 32, 3), 31, dtype=np.uint8)
    controller = tester.ContinuousUncertaintyController(
        V1(), V2(), interval_seconds=60.0
    )

    assert controller.submit(original, frame_number=1) is True
    original[:] = 99
    assert controller.submit(original, frame_number=2) is False
    assert controller.snapshot().v1.state == "ANALYZING"
    assert controller.snapshot().v2.state == "QUEUED"
    release.set()

    deadline = time.perf_counter() + 2.0
    while controller.snapshot().running and time.perf_counter() < deadline:
        time.sleep(0.005)
    snapshot = controller.snapshot()
    controller.close()

    assert [name for name, _ in calls] == ["v1", "v2"]
    assert all(np.all(frame == 31) for _, frame in calls)
    assert snapshot.v1.state == "READY"
    assert snapshot.v2.state == "READY"
    assert snapshot.v1.frame_number == snapshot.v2.frame_number == 1
    assert "variant-only 0" in snapshot.v1.lines[0]
    assert "20 unchanged-frame passes" in snapshot.v2.lines[0]


def test_continuous_controller_keeps_v2_explicitly_unavailable() -> None:
    controller = tester.ContinuousUncertaintyController(
        object(),
        None,
        interval_seconds=2.0,
        v2_unavailable_reason="checkpoint not configured",
    )

    snapshot = controller.snapshot()
    controller.close()

    assert snapshot.v2.state == "UNAVAILABLE"
    assert snapshot.v2.status == "UNAVAILABLE"
    assert snapshot.v2.lines == ("checkpoint not configured",)


def test_continuous_workspace_renders_live_frame_and_separate_method_cards() -> None:
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    now = time.perf_counter()
    snapshot = tester.ContinuousSnapshot(
        v1=tester.MethodSnapshot(
            "v1",
            "V1 INPUT STABILITY",
            "Input-perturbation robustness",
            "READY",
            "INPUT-SENSITIVE",
            ("Targets 1", "Mean persistence 0.900"),
            frame_number=12,
            duration_ms=140.0,
            updated_at=now,
        ),
        v2=tester.MethodSnapshot(
            "v2",
            "V2 MC DROPOUT",
            "Approximate epistemic signal",
            "UNAVAILABLE",
            "UNAVAILABLE",
            ("Validated checkpoint required",),
        ),
    )

    rendered = tester.ContinuousWorkspaceRenderer().render(
        frame,
        snapshot,
        metrics=tester.FrameMetrics(fps=24.0, inference_ms=18.0),
        model_name="trusted.pt",
        device="CUDA:0",
        source_label="video demo.mp4",
        detection_count=1,
        interval_seconds=2.0,
        paused=False,
        now=now,
        exact_inspection_available=True,
    )

    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8
    assert not np.array_equal(rendered[100:200, 100:200], np.zeros((100, 100, 3), dtype=np.uint8))


def test_continuous_cli_interval_is_validated() -> None:
    parser = tester.build_parser()
    enabled = parser.parse_args(
        ["--continuous-uncertainty", "--continuous-uncertainty-interval", "1.5"]
    )
    tester.validate_numeric_options(enabled)
    assert enabled.continuous_uncertainty is True

    disabled = parser.parse_args(
        ["--continuous-uncertainty", "--disable-uncertainty"]
    )
    with pytest.raises(tester.TesterError, match="cannot be combined"):
        tester.validate_numeric_options(disabled)

    invalid = parser.parse_args(["--continuous-uncertainty-interval", "0.1"])
    with pytest.raises(tester.TesterError, match="between 0.25 and 300"):
        tester.validate_numeric_options(invalid)
