from __future__ import annotations

import json

import numpy as np
import pytest

from uav_uncertainty.analysis import METHOD_ID, analyze_image
from uav_uncertainty.detection import Detection
from uav_uncertainty.methods import mc_dropout_capability
from uav_uncertainty.presentation import (
    interpret_target,
    overall_status,
    persistence_status,
)


class StableDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return [Detection(0, "military_tank", 0.8, (5, 5, 20, 20))]


class LastVariantMissingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.calls == 3:
            return []
        return [Detection(0, "military_tank", 0.8, (5, 5, 20, 20))]


class PerturbedOnlyDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.calls == 1:
            return []
        return [Detection(1, "military_vehicle", 0.6, (6, 6, 21, 21))]


def test_analysis_is_deterministic_and_labels_v1_correctly() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    first = analyze_image(image, StableDetector(), sample_count=5, seed=7)
    second = analyze_image(image, StableDetector(), sample_count=5, seed=7)

    assert first.method == METHOD_ID
    assert len(first.samples) == 6
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    assert "not Bayesian uncertainty" in str(first.to_dict()["scientific_scope"])
    assert overall_status(first.baseline_metrics) == "STABLE IN V1 TEST"


def test_analysis_explains_a_missed_variant() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = analyze_image(image, LastVariantMissingDetector(), sample_count=2)
    target = result.baseline_metrics[0]

    assert target.detection_count == 2
    assert target.sample_count == 3
    assert "caused a miss" in interpret_target(target)
    assert overall_status(result.baseline_metrics) == "UNSTABLE / REVIEW"


def test_v2_runtime_reports_external_checkpoint_requirement() -> None:
    capability = mc_dropout_capability()
    assert capability.runtime_implemented is True
    assert capability.available is False
    assert capability.method == "v2_mc_dropout_model_uncertainty"
    assert any("BatchNorm" in requirement for requirement in capability.requirements)
    assert "checkpoint" in capability.reason

    loaded = mc_dropout_capability(validated_checkpoint_loaded=True)
    assert loaded.runtime_implemented is True
    assert loaded.available is True


def test_perturbed_only_detections_are_retained_and_change_status() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = analyze_image(image, PerturbedOnlyDetector(), sample_count=2)

    assert result.baseline_metrics == ()
    assert len(result.perturbed_only_metrics) == 1
    assert result.perturbed_only_metrics[0].detection_persistence == pytest.approx(2 / 3)
    assert (
        overall_status(
            result.baseline_metrics,
            perturbed_only_count=len(result.perturbed_only_metrics),
        )
        == "PERTURBATION-ONLY DETECTIONS"
    )


@pytest.mark.parametrize(
    ("persistence", "expected"),
    [
        (1.0, "STABLE"),
        (0.95, "STABLE"),
        (0.949, "INPUT-SENSITIVE"),
        (0.75, "INPUT-SENSITIVE"),
        (0.749, "UNSTABLE / REVIEW"),
    ],
)
def test_persistence_status_uses_input_robustness_thresholds(
    persistence: float, expected: str
) -> None:
    assert persistence_status(persistence) == expected
