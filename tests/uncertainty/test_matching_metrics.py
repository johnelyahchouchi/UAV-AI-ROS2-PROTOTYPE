from __future__ import annotations

from statistics import pstdev

import pytest

from uav_uncertainty.detection import Detection
from uav_uncertainty.matching import TargetCluster, match_detection_samples
from uav_uncertainty.metrics import calculate_target_metrics


def detection(
    confidence: float,
    class_name: str = "tank",
    bbox: tuple[float, float, float, float] = (10, 10, 30, 30),
) -> Detection:
    return Detection(0, class_name, confidence, bbox)


def test_matching_is_class_agnostic_and_one_to_one() -> None:
    clusters = match_detection_samples(
        [
            [detection(0.9, "tank")],
            [detection(0.8, "vehicle", (11, 10, 31, 30))],
        ],
        iou_threshold=0.5,
    )
    assert len(clusters) == 1
    assert [item.class_name for item in clusters[0].observations.values()] == [
        "tank",
        "vehicle",
    ]


def test_metrics_keep_persistence_confidence_class_and_localization_separate() -> None:
    cluster = TargetCluster(1)
    cluster.add(0, detection(0.7, "tank"))
    cluster.add(1, detection(0.8, "tank", (11, 10, 31, 30)))
    cluster.add(3, detection(0.9, "truck", (12, 10, 32, 30)))

    metrics = calculate_target_metrics(cluster, sample_count=4)

    assert metrics.detection_persistence == 0.75
    assert metrics.missing_sample_indices == (2,)
    assert metrics.confidence_mean == pytest.approx(0.8)
    assert metrics.confidence_std == pytest.approx(pstdev([0.7, 0.8, 0.9]))
    assert metrics.class_agreement == pytest.approx(2 / 3)
    assert metrics.class_entropy_bits > 0
    assert metrics.class_evidence_share == {"tank": 2 / 3, "truck": 1 / 3}
    assert metrics.bbox_center_std_pixels.x > 0
    assert metrics.mean_iou_to_reference < 1


def test_detection_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        Detection(0, "tank", 0.5, (10, 10, 10, 20))


def test_matching_multiple_objects_is_stable_when_detection_order_changes() -> None:
    left = detection(0.9, "tank", (5, 5, 25, 25))
    right = detection(0.8, "truck", (50, 5, 70, 25))
    clusters = match_detection_samples(
        [
            [right, left],
            [
                detection(0.7, "vehicle", (51, 5, 71, 25)),
                detection(0.85, "tank", (6, 5, 26, 25)),
            ],
        ],
        iou_threshold=0.5,
    )

    assert len(clusters) == 2
    assert [cluster.observations[0].bbox for cluster in clusters] == [left.bbox, right.bbox]
    assert [cluster.observations[1].class_name for cluster in clusters] == [
        "tank",
        "vehicle",
    ]
