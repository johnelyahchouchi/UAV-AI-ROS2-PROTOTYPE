"""Reusable in-memory orchestration for input-perturbation stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .detection_matcher import TargetCluster, match_detection_samples
from .detection_types import Detection
from .perturbations import Image, PerturbationConfig, generate_perturbations
from .uncertainty_metrics import TargetMetrics, calculate_all_metrics


class Detector(Protocol):
    """Minimal detector boundary required by the V1 analysis engine."""

    def detect(self, image: Image) -> list[Detection]:
        """Return detections for one OpenCV image."""


ProgressCallback = Callable[[str, int, int], None]
CancellationCheck = Callable[[], bool]


class AnalysisCancelled(RuntimeError):
    """Raised between inference calls when cooperative cancellation is requested."""


@dataclass(frozen=True)
class AnalysisSample:
    """One clean or perturbed image and its detector observations."""

    sample_index: int
    family: str
    parameters: dict[str, float | int]
    image: Image
    detections: list[Detection]

    def metadata(self) -> dict[str, object]:
        """Return JSON-ready metadata without embedding image pixels."""
        return {
            "sample_index": self.sample_index,
            "family": self.family,
            "parameters": dict(self.parameters),
            "detection_count": len(self.detections),
        }


@dataclass(frozen=True)
class ImageAnalysis:
    """Structured V1 output shared by the CLI and dashboard."""

    samples: list[AnalysisSample]
    clusters: list[TargetCluster]
    metrics: list[TargetMetrics]

    @property
    def detections_by_sample(self) -> list[list[Detection]]:
        """Return detector observations in sample-index order."""
        return [sample.detections for sample in self.samples]

    @property
    def sample_metadata(self) -> list[dict[str, object]]:
        """Return JSON-ready metadata in sample-index order."""
        return [sample.metadata() for sample in self.samples]


def _raise_if_cancelled(cancelled: CancellationCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise AnalysisCancelled("Uncertainty analysis was cancelled.")


def _report(
    progress: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
) -> None:
    if progress is not None:
        progress(stage, current, total)


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
    """Run one clean prediction plus seeded perturbed predictions in memory.

    This function performs no console output and writes no files. ``sample_count``
    is the number of perturbed samples; the returned analysis therefore always
    contains ``sample_count + 1`` inference samples.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    settings = perturbation_config or PerturbationConfig()
    variants = generate_perturbations(
        image,
        sample_count=sample_count,
        seed=seed,
        config=settings,
    )

    _raise_if_cancelled(cancelled)
    _report(progress, "clean_baseline", 0, sample_count)
    clean_detections = detector.detect(image)
    samples = [
        AnalysisSample(
            sample_index=0,
            family="clean_baseline",
            parameters={},
            image=image,
            detections=clean_detections,
        )
    ]

    for variant in variants:
        _raise_if_cancelled(cancelled)
        _report(
            progress,
            f"perturbation:{variant.family.value}",
            variant.sample_index,
            sample_count,
        )
        detections = detector.detect(variant.image)
        samples.append(
            AnalysisSample(
                sample_index=variant.sample_index,
                family=variant.family.value,
                parameters=dict(variant.parameters),
                image=variant.image,
                detections=detections,
            )
        )

    _raise_if_cancelled(cancelled)
    _report(progress, "matching", sample_count, sample_count)
    detections_by_sample: Sequence[Sequence[Detection]] = [
        sample.detections for sample in samples
    ]
    clusters = match_detection_samples(detections_by_sample, match_iou)
    _raise_if_cancelled(cancelled)
    _report(progress, "metrics", sample_count, sample_count)
    metrics = calculate_all_metrics(clusters, len(samples))
    return ImageAnalysis(samples=samples, clusters=clusters, metrics=metrics)
