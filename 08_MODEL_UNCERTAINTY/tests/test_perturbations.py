from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uav_uncertainty.perturbations import (  # noqa: E402
    PerturbationFamily,
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    generate_perturbations,
)


class PerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        x = np.linspace(0, 255, 48, dtype=np.uint8)
        plane = np.tile(x, (32, 1))
        self.image = np.dstack((plane, np.flipud(plane), np.fliplr(plane)))

    def test_same_seed_reproduces_metadata_and_pixels(self) -> None:
        first = generate_perturbations(self.image, sample_count=10, seed=123)
        second = generate_perturbations(self.image, sample_count=10, seed=123)

        self.assertEqual(
            [sample.metadata() for sample in first],
            [sample.metadata() for sample in second],
        )
        self.assertTrue(
            all(np.array_equal(left.image, right.image) for left, right in zip(first, second))
        )

    def test_different_fixed_seeds_change_at_least_one_sample(self) -> None:
        first = generate_perturbations(self.image, sample_count=10, seed=123)
        second = generate_perturbations(self.image, sample_count=10, seed=456)
        self.assertNotEqual(first[0].metadata(), second[0].metadata())
        self.assertTrue(
            any(
                not np.array_equal(left.image, right.image)
                for left, right in zip(first, second)
            )
        )

    def test_generation_cycles_through_all_five_families(self) -> None:
        samples = generate_perturbations(self.image, sample_count=5, seed=42)
        self.assertEqual(
            [sample.family for sample in samples],
            [
                PerturbationFamily.BRIGHTNESS,
                PerturbationFamily.CONTRAST,
                PerturbationFamily.GAUSSIAN_BLUR,
                PerturbationFamily.GAUSSIAN_NOISE,
                PerturbationFamily.JPEG_COMPRESSION,
            ],
        )

    def test_every_family_preserves_dimensions_dtype_and_range(self) -> None:
        rng = np.random.default_rng(7)
        outputs = [
            apply_brightness(self.image, gain=1.04, offset=2.0),
            apply_contrast(self.image, gain=0.95),
            apply_gaussian_blur(self.image, kernel_size=3, sigma=0.6),
            apply_gaussian_noise(self.image, sigma=3.0, rng=rng),
            apply_jpeg_compression(self.image, quality=85),
        ]
        for output in outputs:
            with self.subTest():
                self.assertEqual(output.shape, self.image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertGreaterEqual(int(output.min()), 0)
                self.assertLessEqual(int(output.max()), 255)

    def test_jpeg_output_remains_decodable(self) -> None:
        output = apply_jpeg_compression(self.image, quality=90)
        ok, encoded = cv2.imencode(".jpg", output)
        self.assertTrue(ok)
        self.assertIsNotNone(cv2.imdecode(encoded, cv2.IMREAD_COLOR))

    def test_jpeg_encode_failure_raises_clear_error(self) -> None:
        encoded = np.array([], dtype=np.uint8)
        with patch(
            "uav_uncertainty.perturbations.cv2.imencode",
            return_value=(False, encoded),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not encode"):
                apply_jpeg_compression(self.image, quality=90)

    def test_jpeg_decode_failure_raises_clear_error(self) -> None:
        encoded = np.array([1, 2, 3], dtype=np.uint8)
        with patch(
            "uav_uncertainty.perturbations.cv2.imencode",
            return_value=(True, encoded),
        ), patch("uav_uncertainty.perturbations.cv2.imdecode", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "could not decode"):
                apply_jpeg_compression(self.image, quality=90)


if __name__ == "__main__":
    unittest.main()
