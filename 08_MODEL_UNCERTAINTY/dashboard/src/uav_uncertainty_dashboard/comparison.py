"""Saved-run comparison and sequential batch configuration helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

from .errors import DashboardError
from .result_models import LoadedExperiment


COMPARISON_HEADERS = (
    "Run",
    "Method",
    "Method version",
    "Model",
    "Input",
    "Samples",
    "Seed",
    "Image size",
    "Confidence",
    "Match IoU",
    "Target count",
    "Mean persistence",
    "Mean confidence std",
    "Mean class agreement",
    "Mean entropy",
    "Mean IoU",
)


def parse_sample_counts(value: str) -> list[int]:
    """Parse a sequential batch list while retaining user order."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise DashboardError("BATCH_SAMPLES_INVALID", "Batch sample counts must be comma-separated integers.") from error
    if not values or any(not 1 <= item <= 100 for item in values):
        raise DashboardError("BATCH_SAMPLES_INVALID", "Batch sample counts must each be 1–100.")
    unique: list[int] = []
    for item in values:
        if item not in unique:
            unique.append(item)
    if len(unique) > 4:
        raise DashboardError("BATCH_TOO_LARGE", "Compare at most four sample-count configurations per batch.")
    return unique


def _mean(targets: list[dict[str, object]], key: str) -> float | None:
    return fmean(float(target[key]) for target in targets) if targets else None


def _comparison_targets(run: LoadedExperiment) -> list[dict[str, object]]:
    """Return image targets or all per-frame video targets as one distribution."""
    if not run.video_summary:
        return list(run.summary.get("targets", []))
    targets: list[dict[str, object]] = []
    for frame in run.video_summary.get("frames", []):
        summary_path = run.directory / str(frame["directory"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        targets.extend(summary.get("targets", []))
    return targets


def comparison_rows(runs: list[LoadedExperiment]) -> list[list[object]]:
    """Return configuration plus distribution means for 2–4 runs."""
    if not 2 <= len(runs) <= 4:
        raise DashboardError("COMPARISON_SELECTION", "Select between two and four saved runs.")
    rows: list[list[object]] = []
    for run in runs:
        settings = run.metadata.get("configuration", {})
        targets = _comparison_targets(run)
        rows.append(
            [
                run.directory.name,
                run.metadata.get("method_name"),
                run.metadata.get("method_version"),
                run.metadata.get("model_name"),
                run.metadata.get("input_name"),
                settings.get("sample_count"),
                settings.get("seed"),
                settings.get("image_size"),
                settings.get("confidence"),
                settings.get("match_iou"),
                len(targets),
                _mean(targets, "detection_persistence"),
                _mean(targets, "confidence_std"),
                _mean(targets, "class_agreement"),
                _mean(targets, "class_entropy_bits"),
                _mean(targets, "mean_iou_to_reference"),
            ]
        )
    return rows


def metric_distributions(runs: list[LoadedExperiment]) -> dict[str, list[list[float]]]:
    """Return per-target distributions without reducing them to one score."""
    keys = (
        "detection_persistence",
        "confidence_std",
        "class_agreement",
        "class_entropy_bits",
        "mean_iou_to_reference",
    )
    return {
        key: [
            [float(target[key]) for target in _comparison_targets(run)]
            for run in runs
        ]
        for key in keys
    }


def write_comparison_csv(path: Path, rows: list[list[object]]) -> None:
    """Write a downloadable comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(COMPARISON_HEADERS)
        writer.writerows(rows)
