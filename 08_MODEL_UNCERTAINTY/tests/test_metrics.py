from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import pstdev
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uav_uncertainty.detection_matcher import TargetCluster  # noqa: E402
from uav_uncertainty.detection_types import Detection  # noqa: E402
from uav_uncertainty.mc_stability_runner import (  # noqa: E402
    _write_csv,
    _write_json,
    build_summary,
)
from uav_uncertainty.perturbations import PerturbationConfig  # noqa: E402
from uav_uncertainty.uncertainty_metrics import (  # noqa: E402
    calculate_all_metrics,
    calculate_target_metrics,
)


def detection(
    confidence: float,
    class_name: str = "tank",
    box: tuple[float, float, float, float] = (10, 10, 30, 30),
) -> Detection:
    return Detection(0, class_name, confidence, *box)


def cluster(*observations: tuple[int, Detection]) -> TargetCluster:
    result = TargetCluster(cluster_id=1)
    for sample_index, item in observations:
        result.add(sample_index, item)
    return result


class UncertaintyMetricTests(unittest.TestCase):
    def test_persistence_is_detection_count_over_all_samples(self) -> None:
        target = cluster((0, detection(0.8)), (1, detection(0.8)), (3, detection(0.8)))
        metrics = calculate_target_metrics(target, sample_count=4)
        self.assertEqual(metrics.detection_count, 3)
        self.assertEqual(metrics.detection_persistence, 0.75)
        self.assertEqual(metrics.missing_sample_indices, [2])

    def test_confidence_mean_and_population_std_are_correct(self) -> None:
        target = cluster((0, detection(0.7)), (1, detection(0.8)), (2, detection(0.9)))
        metrics = calculate_target_metrics(target, sample_count=3)
        self.assertAlmostEqual(metrics.confidence_mean, 0.8)
        self.assertAlmostEqual(metrics.confidence_std, pstdev([0.7, 0.8, 0.9]))

    def test_dominant_class_agreement_and_distribution_are_correct(self) -> None:
        target = cluster(
            (0, detection(0.8, "tank")),
            (1, detection(0.8, "tank")),
            (2, detection(0.8, "truck")),
        )
        metrics = calculate_target_metrics(target, sample_count=3)
        self.assertEqual(metrics.dominant_class, "tank")
        self.assertEqual(metrics.class_histogram, {"tank": 2, "truck": 1})
        self.assertAlmostEqual(metrics.class_agreement, 2 / 3)
        self.assertAlmostEqual(metrics.class_distribution["truck"], 1 / 3)

    def test_entropy_is_zero_when_classes_agree(self) -> None:
        target = cluster((0, detection(0.8)), (1, detection(0.7)))
        metrics = calculate_target_metrics(target, 2)
        self.assertEqual(metrics.class_entropy_bits, 0.0)
        self.assertEqual(math.copysign(1.0, metrics.class_entropy_bits), 1.0)
        serialized = json.dumps(metrics.to_dict(), sort_keys=True)
        self.assertIn('"class_entropy_bits": 0.0', serialized)
        self.assertNotIn('"class_entropy_bits": -0.0', serialized)

    def test_entropy_is_positive_with_class_disagreement(self) -> None:
        target = cluster((0, detection(0.8, "tank")), (1, detection(0.7, "truck")))
        entropy = calculate_target_metrics(target, 2).class_entropy_bits
        self.assertGreater(entropy, 0.0)
        self.assertEqual(entropy, 1.0)

    def test_one_observation_has_zero_variation_and_full_detected_agreement(self) -> None:
        metrics = calculate_target_metrics(cluster((2, detection(0.75))), sample_count=4)
        self.assertEqual(metrics.detection_persistence, 0.25)
        self.assertEqual(metrics.confidence_std, 0.0)
        self.assertEqual(metrics.bbox_center_std_pixels.x, 0.0)
        self.assertEqual(metrics.bbox_center_std_pixels.y, 0.0)
        self.assertEqual(metrics.bbox_size_std_pixels.x, 0.0)
        self.assertEqual(metrics.bbox_size_std_pixels.y, 0.0)
        self.assertEqual(metrics.class_agreement, 1.0)
        self.assertEqual(metrics.class_entropy_bits, 0.0)
        self.assertEqual(metrics.mean_iou_to_reference, 1.0)
        self.assertEqual(metrics.reference_box_source, "mean_observed_box")

    def test_stable_boxes_have_zero_variation_and_unit_iou(self) -> None:
        target = cluster((0, detection(0.8)), (1, detection(0.7)), (2, detection(0.9)))
        metrics = calculate_target_metrics(target, 3)
        self.assertEqual(metrics.bbox_center_std_pixels.x, 0.0)
        self.assertEqual(metrics.bbox_center_std_pixels.y, 0.0)
        self.assertEqual(metrics.bbox_size_std_pixels.x, 0.0)
        self.assertEqual(metrics.bbox_size_std_pixels.y, 0.0)
        self.assertEqual(metrics.mean_iou_to_reference, 1.0)
        self.assertEqual(metrics.reference_box_source, "clean_baseline")

    def test_moving_boxes_have_more_variation_and_lower_iou(self) -> None:
        stable = cluster((0, detection(0.8)), (1, detection(0.8)))
        moving = cluster(
            (0, detection(0.8, box=(10, 10, 30, 30))),
            (1, detection(0.8, box=(16, 14, 38, 36))),
        )
        stable_metrics = calculate_target_metrics(stable, 2)
        moving_metrics = calculate_target_metrics(moving, 2)
        self.assertGreater(
            moving_metrics.bbox_center_std_pixels.x,
            stable_metrics.bbox_center_std_pixels.x,
        )
        self.assertGreater(
            moving_metrics.bbox_size_std_pixels.x,
            stable_metrics.bbox_size_std_pixels.x,
        )
        self.assertLess(moving_metrics.mean_iou_to_reference, 1.0)

    def test_same_input_produces_identical_metric_output(self) -> None:
        target = cluster(
            (0, detection(0.8, "tank")),
            (1, detection(0.7, "military_vehicle", (11, 10, 31, 30))),
        )
        first = [item.to_dict() for item in calculate_all_metrics([target], 3)]
        second = [item.to_dict() for item in calculate_all_metrics([target], 3)]
        self.assertEqual(first, second)

    def test_same_input_produces_byte_identical_json_and_csv(self) -> None:
        target = cluster((0, detection(0.8)), (1, detection(0.7)))
        metrics = calculate_all_metrics([target], 2)
        arguments = {
            "model_path": Path("model.pt"),
            "image_path": Path("image.jpg"),
            "perturbation_count": 1,
            "seed": 42,
            "image_size": 960,
            "confidence": 0.25,
            "nms_iou": 0.45,
            "match_iou": 0.50,
            "device": "cpu",
            "perturbation_config": PerturbationConfig(),
            "sample_metadata": [
                {"sample_index": 0, "family": "clean_baseline", "parameters": {}},
                {"sample_index": 1, "family": "brightness", "parameters": {"gain": 1.0}},
            ],
            "metrics": metrics,
        }
        first_summary = build_summary(**arguments)
        second_summary = build_summary(**arguments)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_json, second_json = root / "first.json", root / "second.json"
            first_csv, second_csv = root / "first.csv", root / "second.csv"
            _write_json(first_json, first_summary)
            _write_json(second_json, second_summary)
            _write_csv(first_csv, metrics)
            _write_csv(second_csv, metrics)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())


if __name__ == "__main__":
    unittest.main()
