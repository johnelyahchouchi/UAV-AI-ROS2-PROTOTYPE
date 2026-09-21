"""Optional presentation observations emitted only after completed inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .detection import Detection


@dataclass(frozen=True)
class SampleObservation:
    """One actual input and its output; sample indices are zero based."""

    image: np.ndarray
    sample_index: int
    total: int
    family: str
    parameters: tuple[tuple[str, float | int], ...]
    detections: tuple[Detection, ...]


SampleObserver = Callable[[SampleObservation], None]


def observe_sample(
    observer: SampleObserver | None,
    image: np.ndarray,
    *,
    sample_index: int,
    total: int,
    family: str,
    parameters: tuple[tuple[str, float | int], ...] = (),
    detections: tuple[Detection, ...],
) -> None:
    """Give the observer an owned, read-only image, separate from inference input."""

    if observer is None:
        return
    pixels = image.copy()
    pixels.setflags(write=False)
    observer(SampleObservation(pixels, sample_index, total, family, parameters, detections))
