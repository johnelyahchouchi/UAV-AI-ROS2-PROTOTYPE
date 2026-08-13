from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uav_uncertainty.analysis_engine import AnalysisCancelled, analyze_image  # noqa: E402
from uav_uncertainty.detection_types import Detection  # noqa: E402


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _image: object) -> list[Detection]:
        self.calls += 1
        return [Detection(0, "tank", 0.8, 4, 4, 20, 20)]


class AnalysisEngineTests(unittest.TestCase):
    def test_in_memory_analysis_returns_shared_structured_result(self) -> None:
        detector = FakeDetector()
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        progress: list[tuple[str, int, int]] = []

        result = analyze_image(
            image,
            detector,
            sample_count=3,
            seed=42,
            progress=lambda stage, current, total: progress.append((stage, current, total)),
        )

        self.assertEqual(detector.calls, 4)
        self.assertEqual(len(result.samples), 4)
        self.assertEqual(len(result.detections_by_sample), 4)
        self.assertEqual(result.sample_metadata[0]["family"], "clean_baseline")
        self.assertEqual(result.metrics[0].sample_count, 4)
        self.assertEqual(result.metrics[0].detection_persistence, 1.0)
        self.assertTrue(any(stage.startswith("perturbation:") for stage, _, _ in progress))
        self.assertIn(("matching", 3, 3), progress)
        self.assertIn(("metrics", 3, 3), progress)

    def test_cancellation_is_observed_between_inference_calls(self) -> None:
        detector = FakeDetector()
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        with self.assertRaises(AnalysisCancelled):
            analyze_image(
                image,
                detector,
                sample_count=3,
                cancelled=lambda: detector.calls >= 1,
            )
        self.assertEqual(detector.calls, 1)


if __name__ == "__main__":
    unittest.main()
