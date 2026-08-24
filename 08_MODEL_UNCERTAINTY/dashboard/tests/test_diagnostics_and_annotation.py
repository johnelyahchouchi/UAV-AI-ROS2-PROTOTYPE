from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


DASHBOARD_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path[:0] = [str(DASHBOARD_SRC), str(CORE_SRC)]

from uav_uncertainty.analysis_engine import analyze_image  # noqa: E402
from uav_uncertainty.detection_types import Detection  # noqa: E402
from uav_uncertainty_dashboard.annotation import annotate_sample  # noqa: E402
from uav_uncertainty_dashboard.diagnostics import (  # noqa: E402
    instability_events,
    overlapping_cluster_rows,
)


class FixedDetector:
    def detect(self, _image: object) -> list[Detection]:
        return [Detection(0, "tank", 0.8, 5, 5, 20, 20)]


def target(target_id: str, box: list[float], class_name: str) -> dict[str, object]:
    return {
        "target_id": target_id,
        "dominant_class": class_name,
        "reference_bbox_xyxy": box,
        "detection_persistence": 0.8,
        "confidence_mean": 0.7,
        "sample_count": 5,
        "detected_sample_indices": [0, 1, 2, 3],
        "missing_sample_indices": [4],
        "class_agreement": 1.0,
        "bbox_center_std_pixels": {"x": 1.0, "y": 1.0},
        "bbox_size_std_pixels": {"x": 1.0, "y": 1.0},
    }


class DiagnosticsAndAnnotationTests(unittest.TestCase):
    def test_no_target_clusters_create_explicit_diagnostic_event(self) -> None:
        self.assertEqual(
            instability_events([], []),
            ["frame: no target clusters detected in any inference sample."],
        )

    def test_overlapping_cluster_diagnostic_is_non_merging_and_transparent(self) -> None:
        targets = [
            target("target_1", [0, 0, 20, 20], "military_truck"),
            target("target_2", [1, 1, 21, 21], "military_artillery"),
        ]
        rows = overlapping_cluster_rows(targets, 0.80)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][-1], "Possible overlapping/alternative detection")
        events = instability_events(targets, rows)
        self.assertTrue(any("possible overlapping/alternative" in item for item in events))

    def test_annotation_draws_expected_geometry_without_mutating_source(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        original = image.copy()
        analysis = analyze_image(image, FixedDetector(), sample_count=1)
        annotated = annotate_sample(analysis, 0, selected_target_id="target_1")
        self.assertTrue(np.array_equal(image, original))
        self.assertTrue(np.any(annotated[5, 5] != 0))
        self.assertEqual(annotated.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
