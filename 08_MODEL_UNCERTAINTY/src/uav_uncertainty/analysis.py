"""In-memory orchestration for V1 input-perturbation robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .detection import Detection
from .matching import TargetCluster, match_detection_samples
from .metrics import TargetMetrics, calculate_all_metrics
from .perturbations import Image, PerturbationConfig, generate_perturbations


METHOD_ID = "v1_input_perturbation_robustness"


class Detector(Protocol):
    """Minimal detector boundary required by V1 analysis."""

    def detect(self, image: Image) -> list[Detection]:
        """Return detections for one OpenCV BGR image."""


ProgressCallback = Callable[[str, int, int], None]
CancellationCheck = Callable[[], bool]


class AnalysisCancelled(RuntimeError):
    """Raised between inference calls after cooperative cancellation."""


@dataclass(frozen=True)
class AnalysisSample:
    """One clean or perturbed input and its detector observations."""

    sample_index: int
    family: str
    parameters: dict[str, float | int]
    detections: tuple[Detection, ...]

    def metadata(self) -> dict[str, object]:
        """Return JSON-ready metadata without embedding pixels."""

        return {
            "sample_index": self.sample_index,
            "family": self.family,
            "parameters": dict(self.parameters),
            "detection_count": len(self.detections),
        }


@dataclass(frozen=True)
class ImageAnalysis:
    """Structured V1 result shared by local adapters."""

    method: str
    perturbation_count: int
    seed: int
    match_iou: float
    samples: tuple[AnalysisSample, ...]
    clusters: tuple[TargetCluster, ...]
    metrics: tuple[TargetMetrics, ...]

    @property
    def baseline_metrics(self) -> tuple[TargetMetrics, ...]:
        """Return metrics for objects present in the clean baseline frame."""

        baseline_ids = {
            f"target_{cluster.cluster_id}"
            for cluster in self.clusters
            if 0 in cluster.observations
        }
        return tuple(metric for metric in self.metrics if metric.target_id in baseline_ids)

    @property
    def perturbed_only_metrics(self) -> tuple[TargetMetrics, ...]:
        """Return clusters that appeared only after an input perturbation."""

        baseline_ids = {metric.target_id for metric in self.baseline_metrics}
        return tuple(metric for metric in self.metrics if metric.target_id not in baseline_ids)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready output without pixel data."""

        return {
            "method": self.method,
            "scientific_scope": (
                "input-perturbation robustness; not Bayesian uncertainty or "
                "correctness probability"
            ),
            "perturbation_count": self.perturbation_count,
            "total_inference_sample_count": len(self.samples),
            "seed": self.seed,
            "match_iou": self.match_iou,
            "samples": [sample.metadata() for sample in self.samples],
            "targets": [metric.to_dict() for metric in self.metrics],
        }


def _raise_if_cancelled(cancelled: CancellationCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise AnalysisCancelled("Uncertainty analysis was cancelled")


def analyze_image(
    image: Image,
    detector: Detector,
    *,
    sample_count: int = 10,
    seed: int = 42,
    match_iou: float = 0.50,
    perturbation_config: PerturbationConfig | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCheck | None = None,
) -> ImageAnalysis:
    """Run one clean prediction plus seeded perturbed predictions in memory."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must be greater than 0 and at most 1")
    variants = generate_perturbations(
        image,
        sample_count=sample_count,
        seed=seed,
        config=perturbation_config or PerturbationConfig(),
    )

    def notify(stage: str, current: int) -> None:
        if progress is not None:
            progress(stage, current, sample_count)

    _raise_if_cancelled(cancelled)
    notify("clean_baseline", 0)
    clean = tuple(detector.detect(image.copy()))
    samples = [AnalysisSample(0, "clean_baseline", {}, clean)]
    for variant in variants:
        _raise_if_cancelled(cancelled)
        notify(f"perturbation:{variant.family.value}", variant.sample_index)
        detections = tuple(detector.detect(variant.image))
        samples.append(
            AnalysisSample(
                variant.sample_index,
                variant.family.value,
                dict(variant.parameters),
                detections,
            )
        )

    _raise_if_cancelled(cancelled)
    notify("matching", sample_count)
    detections_by_sample: Sequence[Sequence[Detection]] = [
        sample.detections for sample in samples
    ]
    clusters = match_detection_samples(detections_by_sample, match_iou)
    metrics = calculate_all_metrics(clusters, len(samples))
    notify("metrics", sample_count)
    return ImageAnalysis(
        method=METHOD_ID,
        perturbation_count=sample_count,
        seed=seed,
        match_iou=match_iou,
        samples=tuple(samples),
        clusters=tuple(clusters),
        metrics=tuple(metrics),
    )
