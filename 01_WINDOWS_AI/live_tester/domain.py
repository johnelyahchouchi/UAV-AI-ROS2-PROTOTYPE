"""Shared value objects and small pure helpers for the live model tester."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_NAME = "UAV Live Screen Model Tester"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "08_OUTPUTS" / "live_screen_tester"
TANK_CLASS_NAME = "military_tank"


class TesterError(RuntimeError):
    """Raised for user-correctable configuration or runtime failures."""


@dataclass(frozen=True)
class CaptureRegion:
    """An absolute desktop rectangle in pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DetectionResult:
    """One clipped detection ready for filtering and display."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameMetrics:
    """Rolling timing statistics shown in the preview HUD."""

    fps: float = 0.0
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class FrameOutcome:
    """Rendered result of processing one captured frame."""

    annotated_frame: Any
    detections: tuple[DetectionResult, ...]
    metrics: FrameMetrics


@dataclass(frozen=True)
class VideoMetadata:
    """Validated metadata for a local MP4 source."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


class VideoSourceEnded(Exception):
    """Signals normal end-of-file for a non-looping video source."""


class DetectorProtocol(Protocol):
    """Inference boundary used by the frame processor and tests."""

    def detect(self, frame: Any) -> list[DetectionResult]:
        """Return detections for one BGR frame."""


class RendererProtocol(Protocol):
    """Rendering boundary used by the frame processor and tests."""

    def draw_detections(
        self, frame: Any, detections: Sequence[DetectionResult]
    ) -> Any:
        """Return a frame containing detection annotations."""

    def draw_hud(
        self,
        frame: Any,
        *,
        metrics: FrameMetrics,
        model_name: str,
        region: CaptureRegion,
        device: str,
        detection_count: int,
        source_label: str | None = None,
        paused: bool = False,
        last_inspection: str | None = None,
        uncertainty_available: bool = True,
        mcdo_v2_available: bool = False,
        continuous_uncertainty: bool = False,
    ) -> Any:
        """Draw operational status information onto a frame."""


def normalize_class_name(value: str) -> str:
    """Normalize a class label for case-insensitive filtering."""

    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def parse_class_filter(
    value: str, *, tank_only: bool = False
) -> frozenset[str] | None:
    """Parse a comma-separated class filter; ``None`` means all classes."""

    if tank_only:
        return frozenset({TANK_CLASS_NAME})
    raw = str(value).strip()
    if not raw or raw.lower() == "all":
        return None
    names = frozenset(
        normalize_class_name(item) for item in raw.split(",") if item.strip()
    )
    if not names:
        raise TesterError("--classes must be 'all' or a comma-separated class list")
    return names


def filter_detections(
    detections: Iterable[DetectionResult], allowed_classes: frozenset[str] | None
) -> tuple[DetectionResult, ...]:
    """Return detections matching the normalized display filter."""

    if allowed_classes is None:
        return tuple(detections)
    return tuple(
        detection
        for detection in detections
        if normalize_class_name(detection.class_name) in allowed_classes
    )


def contains_rectangle(outer: CaptureRegion, inner: CaptureRegion) -> bool:
    """Return whether one rectangle fully contains another."""

    return (
        inner.left >= outer.left
        and inner.top >= outer.top
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def rectangles_intersect(first: CaptureRegion, second: CaptureRegion) -> bool:
    """Return whether two desktop rectangles overlap."""

    return not (
        first.right <= second.left
        or second.right <= first.left
        or first.bottom <= second.top
        or second.bottom <= first.top
    )


def validate_manual_region(
    region: CaptureRegion, monitors: Sequence[CaptureRegion] = ()
) -> CaptureRegion:
    """Validate an explicit user-entered region and optional monitor bounds."""

    if region.left < 0 or region.top < 0:
        raise TesterError("Manual region left/top coordinates must be non-negative")
    if region.width <= 0 or region.height <= 0:
        raise TesterError("Capture region width and height must be positive")
    if monitors and not any(contains_rectangle(monitor, region) for monitor in monitors):
        raise TesterError("Manual capture region must fit within one active monitor")
    return region


def monitor_by_number(
    monitors: Sequence[CaptureRegion], monitor_number: int
) -> CaptureRegion:
    """Resolve a one-based physical monitor number."""

    if not monitors:
        raise TesterError("No active monitors were reported by the capture backend")
    if monitor_number < 1 or monitor_number > len(monitors):
        raise TesterError(
            f"Monitor {monitor_number} is unavailable; choose 1 through {len(monitors)}"
        )
    return monitors[monitor_number - 1]


def clip_bbox(
    coordinates: Sequence[float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int] | None:
    """Clip an XYXY box to a frame, rejecting malformed or empty boxes."""

    if frame_width <= 0 or frame_height <= 0 or len(coordinates) != 4:
        return None
    try:
        values = tuple(float(value) for value in coordinates)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    maximum_x = frame_width - 1
    maximum_y = frame_height - 1
    clipped = (
        max(0, min(maximum_x, int(round(x1)))),
        max(0, min(maximum_y, int(round(y1)))),
        max(0, min(maximum_x, int(round(x2)))),
        max(0, min(maximum_y, int(round(y2)))),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def safe_overlay_text(value: object, maximum_length: int = 52) -> str:
    """Make external labels safe and bounded for one-line overlays."""

    text = " ".join(str(value).replace("\x00", "").split())
    if not text:
        return "unknown"
    return text if len(text) <= maximum_length else text[: maximum_length - 1] + "…"


def scalar(value: Any) -> float:
    """Convert tensor-like scalar output to float without retaining device state."""

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "item"):
        return float(current.item())
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        if not current:
            raise ValueError("empty scalar sequence")
        return scalar(current[0])
    return float(current)


def coordinates(value: Any) -> Sequence[float]:
    """Convert tensor-like XYXY output to a flat sequence."""

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        current = current.tolist()
    if isinstance(current, Sequence) and current and isinstance(current[0], Sequence):
        current = current[0]
    return current


def class_name_for(names: Any, class_id: int) -> str:
    """Read a class label from list- or mapping-style metadata."""

    try:
        if isinstance(names, Mapping):
            value = names.get(class_id, names.get(str(class_id), class_id))
        else:
            value = names[class_id]
    except (IndexError, KeyError, TypeError):
        value = class_id
    return safe_overlay_text(value)
