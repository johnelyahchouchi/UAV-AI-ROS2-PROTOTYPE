"""Simple Matplotlib figures for target, sample, video, and run comparison views."""

from __future__ import annotations

from typing import Iterable

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .comparison import metric_distributions
from .result_models import LoadedExperiment


def _target_values(summary: dict[str, object], key: str) -> tuple[list[str], list[float]]:
    targets = list(summary.get("targets", []))
    return (
        [str(target["target_id"]) for target in targets],
        [float(target[key]) for target in targets],
    )


def target_metrics_figure(summary: dict[str, object]) -> Figure:
    """Plot five independent raw stability indicators by target."""
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    specifications = (
        ("detection_persistence", "Detection persistence", "Proportion"),
        ("confidence_mean", "Mean confidence", "Detector confidence"),
        ("class_agreement", "Class agreement", "Proportion"),
        ("class_entropy_bits", "Class entropy", "Bits"),
        ("mean_iou_to_reference", "Localization consistency", "Mean IoU"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, specifications):
        labels, values = _target_values(summary, key)
        axis.bar(labels, values, color="#3b82f6")
        axis.set_title(title)
        axis.set_xlabel("Target")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
    confidence_axis = axes.flat[1]
    errors = [float(target["confidence_std"]) for target in summary.get("targets", [])]
    if errors:
        labels, values = _target_values(summary, "confidence_mean")
        confidence_axis.errorbar(labels, values, yerr=errors, fmt="none", ecolor="black", capsize=3)
    axes.flat[5].axis("off")
    return figure


def sample_detection_figure(samples: list[dict[str, object]]) -> Figure:
    """Plot observed detection count for each clean/perturbed sample."""
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    indices = [int(item["sample_index"]) for item in samples]
    counts = [int(item["detection_count"]) for item in samples]
    axis.bar(indices, counts, color="#64748b")
    axis.set_title("Detections by inference sample")
    axis.set_xlabel("Sample index (0 = clean baseline)")
    axis.set_ylabel("Detection count")
    return figure


def comparison_figure(runs: list[LoadedExperiment]) -> Figure:
    """Show per-target distributions for each selected saved run."""
    distributions = metric_distributions(runs)
    labels = [run.directory.name[:24] for run in runs]
    specs = (
        ("detection_persistence", "Persistence", "Proportion"),
        ("confidence_std", "Confidence variation", "Population std"),
        ("class_agreement", "Class agreement", "Proportion"),
        ("class_entropy_bits", "Class entropy", "Bits"),
        ("mean_iou_to_reference", "Mean IoU", "IoU"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (key, title, ylabel) in zip(axes.flat, specs):
        values = [items if items else [float("nan")] for items in distributions[key]]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.set_title(title)
        axis.set_xlabel("Saved experiment")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
    axes.flat[5].axis("off")
    return figure


def video_timeline_figure(frame_rows: Iterable[dict[str, object]]) -> Figure:
    """Plot sampled-frame target count and mean persistence over time."""
    rows = list(frame_rows)
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    timestamps = [float(row["timestamp_seconds"]) for row in rows]
    target_counts = [int(row["target_count"]) for row in rows]
    persistence = [float(row["mean_persistence"]) for row in rows]
    axis.plot(timestamps, target_counts, marker="o", label="Target clusters")
    axis.set_title("Video uncertainty analysis across sampled frames")
    axis.set_xlabel("Video timestamp (seconds)")
    axis.set_ylabel("Target cluster count")
    second = axis.twinx()
    second.plot(timestamps, persistence, marker="s", color="#f97316", label="Mean persistence")
    second.set_ylabel("Mean target persistence")
    return figure
