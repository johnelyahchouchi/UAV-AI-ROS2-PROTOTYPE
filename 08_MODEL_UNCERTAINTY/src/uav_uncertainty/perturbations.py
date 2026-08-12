"""Seeded, mild image perturbations for detector robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import cv2
import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]


class PerturbationFamily(str, Enum):
    """Supported non-geometric perturbation families."""

    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAUSSIAN_BLUR = "gaussian_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    JPEG_COMPRESSION = "jpeg_compression"


@dataclass(frozen=True)
class PerturbationConfig:
    """Conservative parameter ranges for V1 input perturbations."""

    brightness_gain: tuple[float, float] = (0.94, 1.06)
    brightness_offset: tuple[float, float] = (-4.0, 4.0)
    contrast_gain: tuple[float, float] = (0.92, 1.08)
    blur_kernel_sizes: tuple[int, ...] = (3,)
    blur_sigma: tuple[float, float] = (0.35, 0.90)
    noise_sigma: tuple[float, float] = (1.0, 4.0)
    jpeg_quality: tuple[int, int] = (80, 95)

    def __post_init__(self) -> None:
        for name in (
            "brightness_gain",
            "brightness_offset",
            "contrast_gain",
            "blur_sigma",
            "noise_sigma",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} minimum must not exceed its maximum.")
        if self.brightness_gain[0] <= 0 or self.contrast_gain[0] <= 0:
            raise ValueError("Brightness and contrast gains must be positive.")
        if self.blur_sigma[0] < 0 or self.noise_sigma[0] < 0:
            raise ValueError("Blur and noise sigma values must be non-negative.")
        if not self.blur_kernel_sizes or any(
            size <= 0 or size % 2 == 0 for size in self.blur_kernel_sizes
        ):
            raise ValueError("Blur kernel sizes must be positive odd integers.")
        quality_low, quality_high = self.jpeg_quality
        if not 1 <= quality_low <= quality_high <= 100:
            raise ValueError("JPEG quality range must be within 1 through 100.")

    def to_dict(self) -> dict[str, object]:
        """Return configuration values in a JSON-friendly form."""
        return {
            "brightness_gain": list(self.brightness_gain),
            "brightness_offset": list(self.brightness_offset),
            "contrast_gain": list(self.contrast_gain),
            "blur_kernel_sizes": list(self.blur_kernel_sizes),
            "blur_sigma": list(self.blur_sigma),
            "noise_sigma": list(self.noise_sigma),
            "jpeg_quality": list(self.jpeg_quality),
        }


@dataclass(frozen=True)
class PerturbedImage:
    """One generated sample and the parameters that produced it."""

    sample_index: int
    family: PerturbationFamily
    parameters: Mapping[str, float | int]
    image: Image

    def metadata(self) -> dict[str, object]:
        """Return sample metadata without embedding image pixels."""
        return {
            "sample_index": self.sample_index,
            "family": self.family.value,
            "parameters": dict(self.parameters),
        }


FAMILY_ORDER = (
    PerturbationFamily.BRIGHTNESS,
    PerturbationFamily.CONTRAST,
    PerturbationFamily.GAUSSIAN_BLUR,
    PerturbationFamily.GAUSSIAN_NOISE,
    PerturbationFamily.JPEG_COMPRESSION,
)


def _validate_image(image: Image) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Image must be a non-empty 2D or 3D uint8 array.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("A 3D image must be an OpenCV BGR image with 3 channels.")


def _uint8(values: NDArray[np.floating]) -> Image:
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def apply_brightness(image: Image, gain: float, offset: float) -> Image:
    """Apply a mild affine intensity change without changing geometry."""
    _validate_image(image)
    if gain <= 0:
        raise ValueError("Brightness gain must be positive.")
    return _uint8(image.astype(np.float32) * gain + offset)


def apply_contrast(image: Image, gain: float) -> Image:
    """Scale contrast around the per-channel image mean."""
    _validate_image(image)
    if gain <= 0:
        raise ValueError("Contrast gain must be positive.")
    pixels = image.astype(np.float32)
    axes = (0, 1) if image.ndim == 3 else None
    mean = pixels.mean(axis=axes, keepdims=True)
    return _uint8((pixels - mean) * gain + mean)


def apply_gaussian_blur(image: Image, kernel_size: int, sigma: float) -> Image:
    """Apply a small Gaussian blur while preserving image dimensions."""
    _validate_image(image)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Gaussian blur kernel size must be a positive odd integer.")
    if sigma < 0:
        raise ValueError("Gaussian blur sigma must be non-negative.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def apply_gaussian_noise(
    image: Image,
    sigma: float,
    rng: np.random.Generator,
) -> Image:
    """Add zero-mean Gaussian sensor noise from the supplied RNG."""
    _validate_image(image)
    if sigma < 0:
        raise ValueError("Gaussian noise sigma must be non-negative.")
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return _uint8(image.astype(np.float32) + noise)


def apply_jpeg_compression(image: Image, quality: int) -> Image:
    """Round-trip an image through JPEG at the requested quality."""
    _validate_image(image)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be within 1 through 100.")
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode the JPEG perturbation.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("OpenCV could not decode the JPEG perturbation.")
    if decoded.shape != image.shape:
        raise RuntimeError("JPEG perturbation did not preserve image dimensions.")
    return decoded.astype(np.uint8, copy=False)


def generate_perturbations(
    image: Image,
    sample_count: int = 10,
    seed: int = 42,
    config: PerturbationConfig | None = None,
) -> list[PerturbedImage]:
    """Generate seeded samples, cycling through all five families in fixed order."""
    _validate_image(image)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    settings = config or PerturbationConfig()
    rng = np.random.default_rng(seed)
    samples: list[PerturbedImage] = []

    for zero_based_index in range(sample_count):
        family = FAMILY_ORDER[zero_based_index % len(FAMILY_ORDER)]
        parameters: dict[str, float | int]

        if family is PerturbationFamily.BRIGHTNESS:
            gain = float(rng.uniform(*settings.brightness_gain))
            offset = float(rng.uniform(*settings.brightness_offset))
            parameters = {"gain": gain, "offset": offset}
            output = apply_brightness(image, gain, offset)
        elif family is PerturbationFamily.CONTRAST:
            gain = float(rng.uniform(*settings.contrast_gain))
            parameters = {"gain": gain}
            output = apply_contrast(image, gain)
        elif family is PerturbationFamily.GAUSSIAN_BLUR:
            kernel_size = int(rng.choice(settings.blur_kernel_sizes))
            sigma = float(rng.uniform(*settings.blur_sigma))
            parameters = {"kernel_size": kernel_size, "sigma": sigma}
            output = apply_gaussian_blur(image, kernel_size, sigma)
        elif family is PerturbationFamily.GAUSSIAN_NOISE:
            sigma = float(rng.uniform(*settings.noise_sigma))
            parameters = {"sigma": sigma}
            output = apply_gaussian_noise(image, sigma, rng)
        else:
            low, high = settings.jpeg_quality
            quality = int(rng.integers(low, high + 1))
            parameters = {"quality": quality}
            output = apply_jpeg_compression(image, quality)

        samples.append(
            PerturbedImage(
                sample_index=zero_based_index + 1,
                family=family,
                parameters=parameters,
                image=output,
            )
        )

    return samples
