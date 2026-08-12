"""Deterministic class-agnostic IoU matching across inference samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Iterable, Sequence

from .detection_types import BoundingBox, Detection


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection over union for two ``xyxy`` boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class TargetCluster:
    """Detections believed to represent one physical target across samples."""

    cluster_id: int
    observations: dict[int, Detection] = field(default_factory=dict)

    def add(self, sample_index: int, detection: Detection) -> None:
        """Add one observation, rejecting duplicate assignments in a sample."""
        if sample_index in self.observations:
            raise ValueError(
                f"Cluster {self.cluster_id} already has an observation for sample "
                f"{sample_index}."
            )
        self.observations[sample_index] = detection

    def reference_box(self) -> BoundingBox:
        """Return the arithmetic mean box of observations collected so far."""
        if not self.observations:
            raise ValueError("Cannot calculate a reference box for an empty cluster.")
        boxes = [detection.bbox for detection in self.observations.values()]
        return tuple(fmean(box[index] for box in boxes) for index in range(4))  # type: ignore[return-value]


def _detection_key(detection: Detection) -> tuple[object, ...]:
    return (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
        detection.class_name,
        detection.class_id,
        -detection.confidence,
    )


def _sorted_detections(detections: Iterable[Detection]) -> list[Detection]:
    return sorted(detections, key=_detection_key)


def match_detection_samples(
    samples: Sequence[Sequence[Detection]],
    iou_threshold: float = 0.50,
) -> list[TargetCluster]:
    """Cluster detections using deterministic one-to-one highest-IoU matching.

    Matching intentionally ignores class identity. Within each sample, all eligible
    cluster/detection pairs are sorted by descending IoU, then stable geometric keys.
    Each cluster and each detection can be selected once. Unmatched detections create
    new clusters in geometric order.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1.")

    clusters: list[TargetCluster] = []
    next_cluster_id = 1

    for sample_index, sample in enumerate(samples):
        detections = _sorted_detections(sample)
        if not clusters:
            for detection in detections:
                cluster = TargetCluster(next_cluster_id)
                cluster.add(sample_index, detection)
                clusters.append(cluster)
                next_cluster_id += 1
            continue

        candidates: list[tuple[float, int, int, tuple[object, ...]]] = []
        for cluster_index, cluster in enumerate(clusters):
            reference = cluster.reference_box()
            for detection_index, detection in enumerate(detections):
                overlap = intersection_over_union(reference, detection.bbox)
                if overlap >= iou_threshold:
                    candidates.append(
                        (
                            overlap,
                            cluster_index,
                            detection_index,
                            tuple(_detection_key(detection)),
                        )
                    )

        candidates.sort(
            key=lambda item: (
                -item[0],
                clusters[item[1]].cluster_id,
                item[3],
            )
        )
        assigned_clusters: set[int] = set()
        assigned_detections: set[int] = set()

        for _, cluster_index, detection_index, _ in candidates:
            if cluster_index in assigned_clusters or detection_index in assigned_detections:
                continue
            clusters[cluster_index].add(sample_index, detections[detection_index])
            assigned_clusters.add(cluster_index)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            cluster = TargetCluster(next_cluster_id)
            cluster.add(sample_index, detection)
            clusters.append(cluster)
            next_cluster_id += 1

    return clusters
