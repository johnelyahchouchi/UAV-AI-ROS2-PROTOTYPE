"""Detector-independent detection values used by uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One object detection in source-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        values = (self.confidence, *self.bbox)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detection confidence and coordinates must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detection bounding box must have positive dimensions")
        if not self.class_name.strip():
            raise ValueError("Detection class_name must not be empty")

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in pixels."""

        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        """Return bounding-box width and height in pixels."""

        x1, y1, x2, y2 = self.bbox
        return x2 - x1, y2 - y1
