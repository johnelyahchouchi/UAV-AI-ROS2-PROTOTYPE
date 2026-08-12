"""Detector-independent domain types used by the uncertainty core.Instead of letting all the other files depend directly on Ultralytics objects, I created my own simple Detection object."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One object detection expressed in source-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.confidence, self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detection confidence and coordinates must be finite.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Detection bounding box must have positive width and height.")
        if not self.class_name.strip():
            raise ValueError("Detection class_name must not be empty.")

    @property
    def bbox(self) -> BoundingBox:
        """Return the bounding box as ``(x1, y1, x2, y2)``."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in pixels."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        """Return bounding-box width and height in pixels."""
        return self.x2 - self.x1, self.y2 - self.y1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }
