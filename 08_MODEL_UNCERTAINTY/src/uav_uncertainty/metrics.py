"""Transparent per-target metrics for input-perturbation robustness."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from statistics import fmean, pstdev

from .detection import BoundingBox
from .matching import TargetCluster, intersection_over_union


@dataclass(frozen=True)
class AxisVariation:
    """Population standard deviations for two spatial axes."""

    x: float
    y: float


@dataclass(frozen=True)
class TargetMetrics:
    """Raw persistence, classification, confidence, and localization metrics."""

    target_id: str
    sample_count: int
    detection_count: int
    detection_persistence: float
    detected_sample_indices: tuple[int, ...]
    missing_sample_indices: tuple[int, ...]
    confidence_mean: float
    confidence_std: float
    class_histogram: dict[str, int]
    class_evidence_share: dict[str, float]
    dominant_class: str
    class_agreement: float
    class_entropy_bits: float
    bbox_center_std_pixels: AxisVariation
    bbox_size_std_pixels: AxisVariation
    reference_box_source: str
    reference_bbox_xyxy: BoundingBox
    mean_iou_to_reference: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""

        return asdict(self)


def _population_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _mean_box(boxes: list[BoundingBox]) -> BoundingBox:
    return tuple(fmean(box[index] for box in boxes) for index in range(4))  # type: ignore[return-value]


def calculate_target_metrics(
    cluster: TargetCluster, sample_count: int
) -> TargetMetrics:
    """Calculate V1 metrics for one target cluster."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not cluster.observations:
        raise ValueError("Cannot calculate metrics for an empty cluster")
    invalid = [
        index for index in cluster.observations if index < 0 or index >= sample_count
    ]
    if invalid:
        raise ValueError(f"Observation indices outside sample range: {invalid}")

    detected_indices = sorted(cluster.observations)
    detections = [cluster.observations[index] for index in detected_indices]
    detection_count = len(detections)
    confidences = [detection.confidence for detection in detections]
    class_counts = Counter(detection.class_name for detection in detections)
    sorted_counts = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    dominant_class, dominant_count = sorted_counts[0]
    histogram = dict(sorted(class_counts.items()))
    evidence_share = {
        class_name: count / detection_count for class_name, count in histogram.items()
    }
    entropy = sum(
        -share * math.log2(share)
        for share in evidence_share.values()
        if share > 0.0
    )
    entropy = 0.0 if entropy == 0.0 else entropy

    centers = [detection.center for detection in detections]
    sizes = [detection.size for detection in detections]
    boxes = [detection.bbox for detection in detections]
    if 0 in cluster.observations:
        reference_box = cluster.observations[0].bbox
        reference_source = "clean_baseline"
    else:
        reference_box = _mean_box(boxes)
        reference_source = "mean_observed_box"
    mean_iou = fmean(
        intersection_over_union(reference_box, detection.bbox)
        for detection in detections
    )

    return TargetMetrics(
        target_id=f"target_{cluster.cluster_id}",
        sample_count=sample_count,
        detection_count=detection_count,
        detection_persistence=detection_count / sample_count,
        detected_sample_indices=tuple(detected_indices),
        missing_sample_indices=tuple(
            index for index in range(sample_count) if index not in cluster.observations
        ),
        confidence_mean=fmean(confidences),
        confidence_std=_population_std(confidences),
        class_histogram=histogram,
        class_evidence_share=evidence_share,
        dominant_class=dominant_class,
        class_agreement=dominant_count / detection_count,
        class_entropy_bits=entropy,
        bbox_center_std_pixels=AxisVariation(
            _population_std([center[0] for center in centers]),
            _population_std([center[1] for center in centers]),
        ),
        bbox_size_std_pixels=AxisVariation(
            _population_std([size[0] for size in sizes]),
            _population_std([size[1] for size in sizes]),
        ),
        reference_box_source=reference_source,
        reference_bbox_xyxy=reference_box,
        mean_iou_to_reference=mean_iou,
    )


def calculate_all_metrics(
    clusters: list[TargetCluster], sample_count: int
) -> list[TargetMetrics]:
    """Calculate metrics in stable target-ID order."""

    return [
        calculate_target_metrics(cluster, sample_count)
        for cluster in sorted(clusters, key=lambda item: item.cluster_id)
    ]
