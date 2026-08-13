"""Registered uncertainty experiment methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from uav_uncertainty.analysis_engine import Detector, ImageAnalysis, analyze_image
from uav_uncertainty.perturbations import Image, PerturbationConfig

from .configuration import ExperimentSettings, METHOD_INPUT_PERTURBATION_V1


@dataclass(frozen=True)
class MethodIdentity:
    """Stable identity stored with every dashboard run."""

    display_name: str
    identifier: str
    version: str


class ExperimentMethod(ABC):
    """Interface for present and future uncertainty experiment methods."""

    identity: MethodIdentity

    @abstractmethod
    def analyze(
        self,
        image: Image,
        detector: Detector,
        settings: ExperimentSettings,
        *,
        progress: object | None = None,
        cancelled: object | None = None,
    ) -> ImageAnalysis:
        """Analyze one image and return a structured result."""


class InputPerturbationV1(ExperimentMethod):
    """Current seeded input-perturbation method backed by the V1 core."""

    identity = MethodIdentity(
        display_name=METHOD_INPUT_PERTURBATION_V1,
        identifier="monte_carlo_input_perturbation_v1",
        version="1.0",
    )

    def analyze(
        self,
        image: Image,
        detector: Detector,
        settings: ExperimentSettings,
        *,
        progress: object | None = None,
        cancelled: object | None = None,
    ) -> ImageAnalysis:
        return analyze_image(
            image,
            detector,
            sample_count=settings.sample_count,
            seed=settings.seed,
            match_iou=settings.match_iou,
            perturbation_config=PerturbationConfig(),
            progress=progress,  # type: ignore[arg-type]
            cancelled=cancelled,  # type: ignore[arg-type]
        )


METHOD_REGISTRY: dict[str, ExperimentMethod] = {
    METHOD_INPUT_PERTURBATION_V1: InputPerturbationV1(),
}


def method_for(name: str) -> ExperimentMethod:
    """Return an implemented method or fail without advertising future methods."""
    try:
        return METHOD_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unsupported uncertainty method: {name}") from error
