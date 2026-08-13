from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


DASHBOARD_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path[:0] = [str(DASHBOARD_SRC), str(CORE_SRC)]

from uav_uncertainty.detection_types import Detection  # noqa: E402
from uav_uncertainty_dashboard.comparison import (  # noqa: E402
    comparison_rows,
    metric_distributions,
)
from uav_uncertainty_dashboard.configuration import (  # noqa: E402
    DeviceChoice,
    ExperimentSettings,
    VideoSamplingMode,
)
from uav_uncertainty_dashboard.errors import DashboardError, ProcessingCancelled  # noqa: E402
from uav_uncertainty_dashboard.experiment_runner import (  # noqa: E402
    DetectorCache,
    ExperimentRunner,
    ImageExperimentRequest,
    VideoExperimentRequest,
)
from uav_uncertainty_dashboard.output_manager import OutputManager  # noqa: E402
from uav_uncertainty_dashboard.processing_control import ProcessingController  # noqa: E402
from uav_uncertainty_dashboard.result_loader import (  # noqa: E402
    family_rows,
    load_run,
    sample_rows,
    target_rows,
)


class FakeDetector:
    instances: list["FakeDetector"] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls = 0
        type(self).instances.append(self)

    def detect(self, _image: object) -> list[Detection]:
        self.calls += 1
        return [Detection(0, "military_tank", 0.75, 5, 5, 24, 24)]


class FailingDetector(FakeDetector):
    def detect(self, image: object) -> list[Detection]:
        result = super().detect(image)
        if self.calls == 2:
            raise RuntimeError("synthetic inference failure")
        return result


def fixed_clock() -> datetime:
    return datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def write_image(path: Path) -> None:
    if not cv2.imwrite(str(path), np.full((32, 32, 3), 128, dtype=np.uint8)):
        raise RuntimeError("Could not create synthetic image.")


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic video.")
    try:
        for index in range(5):
            writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


class RunnerOutputAndResultTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeDetector.instances.clear()

    def _runner(self, root: Path, factory: object = FakeDetector) -> tuple[ExperimentRunner, OutputManager, ProcessingController]:
        output = OutputManager(root / "outputs", clock=fixed_clock)
        controller = ProcessingController()
        runner = ExperimentRunner(
            DetectorCache(factory=factory),  # type: ignore[arg-type]
            output,
            controller,
            clock=fixed_clock,
        )
        return runner, output, controller

    def _request(self, root: Path, samples: int = 2) -> ImageExperimentRequest:
        image = root / "image.jpg"
        model = root / "model.pt"
        write_image(image)
        model.write_bytes(b"fake model")
        return ImageExperimentRequest(
            image,
            model,
            ExperimentSettings(sample_count=samples, device=DeviceChoice.CPU),
        )

    def test_image_orchestration_publishes_complete_reloadable_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner, output, _ = self._runner(root)
            result = runner.run_image(self._request(root))

            self.assertEqual(FakeDetector.instances[0].calls, 3)
            self.assertTrue(result.published.summary.is_file())
            self.assertTrue(result.published.targets_csv.is_file())
            self.assertTrue(result.published.sample_metadata.is_file())
            self.assertTrue(result.published.baseline_preview.is_file())
            self.assertEqual(list(output.staging_root.iterdir()), [])
            loaded = load_run(result.published.directory)
            self.assertEqual(loaded.summary["sampling"]["total_inference_sample_count"], 3)
            self.assertEqual(len(target_rows(loaded.summary)), 1)
            self.assertEqual(len(sample_rows(loaded.samples)), 3)
            self.assertGreaterEqual(len(family_rows(loaded.samples)), 2)

    def test_batch_reuses_detector_and_comparison_preserves_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner, _, _ = self._runner(root)
            results = runner.run_image_batch(self._request(root, 1), [1, 2])
            self.assertEqual(len(FakeDetector.instances), 1)
            self.assertEqual(FakeDetector.instances[0].calls, 5)
            runs = [result.loaded for result in results]
            rows = comparison_rows(runs)
            distributions = metric_distributions(runs)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row[5] for row in rows], [1, 2])
            self.assertEqual(distributions["detection_persistence"], [[1.0], [1.0]])

    def test_output_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = OutputManager(root / "outputs", clock=fixed_clock)
            source = root / "image.jpg"
            write_image(source)
            first = manager.prepare_run(source, "method", "same-job")
            with self.assertRaises(DashboardError) as raised:
                manager.prepare_run(source, "method", "same-job")
            self.assertEqual(raised.exception.code, "OUTPUT_COLLISION")
            manager.abort(first)

    def test_failure_does_not_publish_or_leave_corrupt_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner, output, _ = self._runner(root, FailingDetector)
            with self.assertRaises(DashboardError):
                runner.run_image(self._request(root))
            self.assertEqual(list(output.staging_root.iterdir()), [])
            self.assertEqual(
                [path for path in output.output_root.iterdir() if path.name != ".staging"],
                [],
            )

    def test_cancel_removes_partial_run_and_allows_next_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner, output, controller = self._runner(root)

            def cancel_on_first_sample(_fraction: float, description: str) -> None:
                if description.startswith("Sample 1/"):
                    controller.request_cancel()

            with self.assertRaises(ProcessingCancelled):
                runner.run_image(self._request(root), progress=cancel_on_first_sample)
            self.assertEqual(list(output.staging_root.iterdir()), [])
            self.assertFalse(controller.is_running)

    def test_video_run_analyzes_only_selected_frames_and_writes_frame_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            model = root / "model.pt"
            write_video(video)
            model.write_bytes(b"fake")
            runner, _, _ = self._runner(root)
            request = VideoExperimentRequest(
                video,
                model,
                ExperimentSettings(sample_count=1, device=DeviceChoice.CPU),
                VideoSamplingMode.INTERVAL,
                interval_seconds=0.2,
                maximum_frames=2,
            )
            estimate = runner.video_estimate(request)
            self.assertEqual(estimate["selected_video_frames"], 2)
            self.assertEqual(estimate["estimated_total_inference_calls"], 4)
            result = runner.run_video(request)
            self.assertEqual(FakeDetector.instances[0].calls, 4)
            self.assertTrue(result.published.video_frames_csv.is_file())
            self.assertEqual(len(result.loaded.video_summary["frames"]), 2)

    def test_loader_keeps_unknown_future_method_and_schema_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir(parents=True)
            metadata = {
                "dashboard_schema_version": "2.0",
                "method_name": "Future Real Method",
                "method_version": "2.0",
                "input_kind": "Image",
                "input_name": "image.jpg",
                "configuration": {},
            }
            (root / "dashboard_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"targets": [], "sampling": {}}), encoding="utf-8")
            (root / "sample_metadata.json").write_text("[]", encoding="utf-8")
            loaded = load_run(root)
            self.assertEqual(loaded.metadata["method_name"], "Future Real Method")
            self.assertEqual(loaded.metadata["dashboard_schema_version"], "2.0")


if __name__ == "__main__":
    unittest.main()
