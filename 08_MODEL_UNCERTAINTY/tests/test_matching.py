from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uav_uncertainty.detection_matcher import (  # noqa: E402
    intersection_over_union,
    match_detection_samples,
)
from uav_uncertainty.detection_types import Detection  # noqa: E402
from uav_uncertainty.detector_adapter import detections_from_result  # noqa: E402
from uav_uncertainty.uncertainty_metrics import calculate_target_metrics  # noqa: E402


def detection(
    box: tuple[float, float, float, float],
    class_name: str = "tank",
    confidence: float = 0.8,
    class_id: int = 0,
) -> Detection:
    return Detection(class_id, class_name, confidence, *box)


class DetectionMatchingTests(unittest.TestCase):
    def test_identical_boxes_match(self) -> None:
        box = detection((10, 10, 30, 30))
        clusters = match_detection_samples([[box], [box]])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].observations), {0, 1})
        self.assertEqual(intersection_over_union(box.bbox, box.bbox), 1.0)

    def test_reordered_detections_match_by_location(self) -> None:
        left = detection((0, 0, 20, 20))
        right = detection((50, 0, 70, 20), class_name="truck")
        clusters = match_detection_samples(
            [[left, right], [detection((51, 0, 71, 20)), detection((1, 0, 21, 20))]]
        )
        self.assertEqual(len(clusters), 2)
        self.assertLess(clusters[0].observations[1].x1, 10)
        self.assertGreater(clusters[1].observations[1].x1, 40)

    def test_missing_detection_leaves_cluster_without_observation(self) -> None:
        target = detection((0, 0, 20, 20))
        clusters = match_detection_samples([[target], [], [target]])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].observations), {0, 2})

    def test_extra_detection_creates_new_cluster(self) -> None:
        target = detection((0, 0, 20, 20))
        extra = detection((60, 60, 80, 80), class_name="new")
        clusters = match_detection_samples([[target], [target, extra]])
        self.assertEqual(len(clusters), 2)
        self.assertEqual(set(clusters[1].observations), {1})

    def test_class_disagreement_still_matches_spatially(self) -> None:
        clusters = match_detection_samples(
            [
                [detection((10, 10, 30, 30), "tank")],
                [detection((11, 10, 31, 30), "military_vehicle")],
                [detection((10, 11, 30, 31), "truck")],
            ]
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            {item.class_name for item in clusters[0].observations.values()},
            {"tank", "military_vehicle", "truck"},
        )

    def test_low_iou_boxes_do_not_match(self) -> None:
        clusters = match_detection_samples(
            [[detection((0, 0, 10, 10))], [detection((30, 30, 40, 40))]],
            iou_threshold=0.50,
        )
        self.assertEqual(len(clusters), 2)

    def test_no_detections_returns_no_clusters(self) -> None:
        self.assertEqual(match_detection_samples([[], [], []]), [])

    def test_multiple_nearby_detections_are_one_to_one(self) -> None:
        first = [detection((0, 0, 12, 12)), detection((10, 0, 22, 12))]
        second = [detection((11, 0, 23, 12)), detection((1, 0, 13, 12))]
        clusters = match_detection_samples([first, second], iou_threshold=0.40)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0].observations), 2)
        self.assertEqual(len(clusters[1].observations), 2)
        self.assertLess(clusters[0].observations[1].x1, clusters[1].observations[1].x1)

    def test_perturbed_only_target_keeps_clean_baseline_as_a_miss(self) -> None:
        first = detection((10, 10, 30, 30))
        second = detection((12, 10, 32, 30))
        clusters = match_detection_samples([[], [first], [second]])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].observations), {1, 2})

        metrics = calculate_target_metrics(clusters[0], sample_count=3)
        self.assertEqual(metrics.missing_sample_indices, [0])
        self.assertEqual(metrics.detection_persistence, 2 / 3)
        self.assertEqual(metrics.reference_box_source, "mean_observed_box")
        self.assertEqual(metrics.reference_bbox_xyxy, (11.0, 10.0, 31.0, 30.0))

    def test_ambiguous_greedy_matching_is_deterministic_and_one_to_one(self) -> None:
        baseline_a = detection((0, 0, 20, 20), class_name="physical_a")
        baseline_b = detection((10, 0, 30, 20), class_name="physical_b")
        moved_a = detection((8, 0, 28, 20), class_name="physical_a")
        moved_b = detection((2, 0, 22, 20), class_name="physical_b")
        samples = [[baseline_a, baseline_b], [moved_a, moved_b]]

        first = match_detection_samples(samples, iou_threshold=0.40)
        second = match_detection_samples(samples, iou_threshold=0.40)

        assignments = [
            [cluster.observations[index].class_name for index in (0, 1)]
            for cluster in first
        ]
        self.assertEqual(assignments, [["physical_a", "physical_b"], ["physical_b", "physical_a"]])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            {id(cluster.observations[1]) for cluster in first},
            {id(moved_a), id(moved_b)},
        )


class FakeScalar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def item(self) -> float | int:
        return self.value


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeBox:
    def __init__(
        self,
        coordinates: list[float] | None = None,
        class_id: int = 3,
        confidence: float = 0.72,
    ) -> None:
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]
        self.xyxy = [FakeVector(coordinates or [5.0, 6.0, 25.0, 30.0])]


class DetectorAdapterTests(unittest.TestCase):
    def test_fake_ultralytics_result_converts_without_ultralytics_import(self) -> None:
        result = type("FakeResult", (), {"boxes": [FakeBox()], "names": {3: "tank"}})()
        converted = detections_from_result(result)
        self.assertEqual(
            converted,
            [Detection(3, "tank", 0.72, 5.0, 6.0, 25.0, 30.0)],
        )

    def test_result_without_boxes_converts_to_empty_list(self) -> None:
        result = type("FakeResult", (), {"boxes": None, "names": {}})()
        self.assertEqual(detections_from_result(result), [])

    def test_sequence_class_names_are_supported(self) -> None:
        result = type(
            "FakeResult",
            (),
            {"boxes": [FakeBox(class_id=1)], "names": ["truck", "tank"]},
        )()
        self.assertEqual(detections_from_result(result)[0].class_name, "tank")

    def test_empty_boxes_convert_to_empty_list(self) -> None:
        result = type("FakeResult", (), {"boxes": [], "names": {}})()
        self.assertEqual(detections_from_result(result), [])

    def test_malformed_and_zero_area_boxes_are_skipped(self) -> None:
        result = type(
            "FakeResult",
            (),
            {
                "boxes": [
                    FakeBox([1.0, 2.0, 3.0]),
                    FakeBox([5.0, 6.0, 5.0, 30.0]),
                    FakeBox([5.0, 6.0, 25.0, 6.0]),
                ],
                "names": {3: "tank"},
            },
        )()
        self.assertEqual(detections_from_result(result), [])


class DetectionValidationTests(unittest.TestCase):
    def test_invalid_detection_values_are_rejected(self) -> None:
        cases = {
            "zero width": (0, "tank", 0.5, 10, 10, 10, 20),
            "zero height": (0, "tank", 0.5, 10, 10, 20, 10),
            "reversed width": (0, "tank", 0.5, 20, 10, 10, 20),
            "reversed height": (0, "tank", 0.5, 10, 20, 20, 10),
            "infinite coordinate": (0, "tank", 0.5, 10, 10, float("inf"), 20),
            "nan coordinate": (0, "tank", 0.5, 10, float("nan"), 20, 20),
            "empty class": (0, "   ", 0.5, 10, 10, 20, 20),
            "low confidence": (0, "tank", -0.01, 10, 10, 20, 20),
            "high confidence": (0, "tank", 1.01, 10, 10, 20, 20),
        }
        for name, arguments in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    Detection(*arguments)


if __name__ == "__main__":
    unittest.main()
