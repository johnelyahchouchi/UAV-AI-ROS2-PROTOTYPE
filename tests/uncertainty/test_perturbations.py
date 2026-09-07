from __future__ import annotations

import cv2
import numpy as np
import pytest

from uav_uncertainty.perturbations import (
    FAMILY_ORDER,
    PerturbationConfig,
    apply_jpeg_compression,
    generate_perturbations,
)


@pytest.fixture
def image() -> np.ndarray:
    plane = np.tile(np.linspace(0, 255, 48, dtype=np.uint8), (32, 1))
    return np.dstack((plane, np.flipud(plane), np.fliplr(plane)))


def test_seed_reproduces_metadata_and_pixels(image: np.ndarray) -> None:
    first = generate_perturbations(image, sample_count=10, seed=123)
    second = generate_perturbations(image, sample_count=10, seed=123)
    assert [item.metadata() for item in first] == [item.metadata() for item in second]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(first, second))


def test_generation_cycles_all_families(image: np.ndarray) -> None:
    samples = generate_perturbations(image, sample_count=5, seed=42)
    assert [item.family for item in samples] == list(FAMILY_ORDER)
    assert all(item.image.shape == image.shape for item in samples)
    assert all(item.image.dtype == np.uint8 for item in samples)


def test_jpeg_round_trip_remains_decodable(image: np.ndarray) -> None:
    output = apply_jpeg_compression(image, quality=90)
    ok, encoded = cv2.imencode(".jpg", output)
    assert ok
    assert cv2.imdecode(encoded, cv2.IMREAD_COLOR) is not None


def test_generation_does_not_mutate_the_input(image: np.ndarray) -> None:
    original = image.copy()
    generate_perturbations(image, sample_count=10, seed=99)
    assert np.array_equal(image, original)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_configuration_rejects_non_finite_ranges(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PerturbationConfig(noise_sigma=(0.0, value))
