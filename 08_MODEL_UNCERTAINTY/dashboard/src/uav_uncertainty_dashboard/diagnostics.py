"""Transparent dashboard-only diagnostics that do not alter V1 metrics."""

from __future__ import annotations

from itertools import combinations

from uav_uncertainty.detection_matcher import intersection_over_union


OVERLAP_HEADERS = (
    "Target A",
    "Class A",
    "Target B",
    "Class B",
    "Reference IoU",
    "Persistence A",
    "Persistence B",
    "Mean confidence A",
    "Mean confidence B",
    "Diagnostic",
)


def overlapping_cluster_rows(
    targets: list[dict[str, object]],
    threshold: float = 0.80,
) -> list[list[object]]:
    """Report pairs whose reference boxes overlap at or above the threshold."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Overlap threshold must be in (0, 1].")
    rows: list[list[object]] = []
    for first, second in combinations(targets, 2):
        first_box = tuple(float(v) for v in first["reference_bbox_xyxy"])
        second_box = tuple(float(v) for v in second["reference_bbox_xyxy"])
        overlap = intersection_over_union(first_box, second_box)  # type: ignore[arg-type]
        if overlap < threshold:
            continue
        rows.append(
            [
                first["target_id"],
                first["dominant_class"],
                second["target_id"],
                second["dominant_class"],
                overlap,
                first["detection_persistence"],
                second["detection_persistence"],
                first["confidence_mean"],
                second["confidence_mean"],
                "Possible overlapping/alternative detection",
            ]
        )
    return rows


def instability_events(
    targets: list[dict[str, object]],
    overlap_rows: list[list[object]],
    *,
    pixel_variation_threshold: float = 10.0,
) -> list[str]:
    """Derive factual events using explicit, visible rules."""
    if not targets:
        return ["frame: no target clusters detected in any inference sample."]

    events: list[str] = []
    for target in targets:
        target_id = str(target["target_id"])
        detected = [int(value) for value in target["detected_sample_indices"]]
        missing = [int(value) for value in target["missing_sample_indices"]]
        if 0 not in detected:
            events.append(f"{target_id}: appears only under perturbation in this run.")
        if missing:
            events.append(
                f"{target_id}: missing in {len(missing)}/{target['sample_count']} samples."
            )
        if float(target["class_agreement"]) < 1.0:
            events.append(
                f"{target_id}: class disagreement observed; agreement="
                f"{float(target['class_agreement']):.3f}."
            )
        center = target["bbox_center_std_pixels"]
        size = target["bbox_size_std_pixels"]
        values = [float(center["x"]), float(center["y"]), float(size["x"]), float(size["y"])]
        if max(values) > pixel_variation_threshold:
            events.append(
                f"{target_id}: box variation exceeds the dashboard diagnostic rule "
                f"of {pixel_variation_threshold:g} source pixels."
            )
    for row in overlap_rows:
        events.append(
            f"{row[0]} and {row[2]}: possible overlapping/alternative detection "
            f"(reference IoU={float(row[4]):.3f})."
        )
    return events
