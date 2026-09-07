"""Human-readable, non-probabilistic interpretation of robustness metrics."""

from __future__ import annotations

import math

from .metrics import TargetMetrics


def persistence_status(persistence: float) -> str:
    """Classify observed input robustness without implying accuracy."""

    if not math.isfinite(persistence) or not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must be finite and between 0 and 1")
    if persistence >= 0.95:
        return "STABLE"
    if persistence >= 0.75:
        return "INPUT-SENSITIVE"
    return "UNSTABLE / REVIEW"


def interpret_target(target: TargetMetrics) -> str:
    """Describe observed instability without inventing a composite score."""

    observations: list[str] = []
    misses = target.sample_count - target.detection_count
    if misses:
        observations.append(f"{misses} input variant(s) caused a miss")
    if target.class_agreement < 0.80:
        observations.append("class evidence competed across input variants")
    elif target.class_agreement < 1.0:
        observations.append("the predicted class changed at least once")
    if target.confidence_std >= 0.10:
        observations.append("confidence varied substantially")
    if target.mean_iou_to_reference < 0.60:
        observations.append("localization moved substantially")
    elif target.mean_iou_to_reference < 0.80:
        observations.append("localization varied moderately")
    if not observations:
        return "Stable across the tested input perturbations"
    return "; ".join(observations).capitalize() + "."


def overall_status(
    targets: tuple[TargetMetrics, ...], *, perturbed_only_count: int = 0
) -> str:
    """Return a concise HUD label based on transparent threshold checks."""

    if perturbed_only_count < 0:
        raise ValueError("perturbed_only_count cannot be negative")
    if not targets:
        return (
            "PERTURBATION-ONLY DETECTIONS"
            if perturbed_only_count
            else "NO BASELINE DETECTIONS"
        )
    if any(target.detection_persistence < 0.75 for target in targets):
        return "UNSTABLE / REVIEW"
    if any(target.class_agreement < 0.80 for target in targets):
        return "CLASSIFICATION UNSTABLE"
    if any(target.mean_iou_to_reference < 0.60 for target in targets):
        return "LOCALIZATION UNSTABLE"
    if any(target.detection_persistence < 0.95 for target in targets):
        return "INPUT-SENSITIVE"
    if any(target.confidence_std >= 0.10 for target in targets):
        return "CONFIDENCE VARIABLE"
    if any(target.class_agreement < 0.95 for target in targets):
        return "INPUT-SENSITIVE"
    if any(target.mean_iou_to_reference < 0.80 for target in targets):
        return "INPUT-SENSITIVE"
    if perturbed_only_count:
        return "PERTURBATION-ONLY DETECTIONS"
    return "STABLE IN V1 TEST"
