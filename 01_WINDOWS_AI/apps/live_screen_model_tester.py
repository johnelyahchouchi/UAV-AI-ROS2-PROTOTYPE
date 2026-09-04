#!/usr/bin/env python3
"""Local Windows screen-capture tester for trusted Ultralytics YOLO models."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.model_integrity import (  # noqa: E402
    ModelIntegrityError,
    load_trusted_yolo,
    verify_trusted_model,
)


WINDOW_NAME = "UAV Live Screen Model Tester"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "08_OUTPUTS" / "live_screen_tester"
TANK_CLASS_NAME = "military_tank"


class TesterError(RuntimeError):
    """Raised for user-correctable tester configuration or runtime failures."""


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
    uncertainty: Mapping[str, float] | None = None


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
    ) -> Any:
        """Draw operational status information onto a frame."""


def normalize_class_name(value: str) -> str:
    """Normalize a class label for case-insensitive CLI filtering."""

    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def parse_class_filter(value: str, *, tank_only: bool = False) -> frozenset[str] | None:
    """Parse a comma-separated class filter; ``None`` means all classes."""

    if tank_only:
        return frozenset({TANK_CLASS_NAME})

    raw = str(value).strip()
    if not raw or raw.lower() == "all":
        return None

    names = frozenset(normalize_class_name(item) for item in raw.split(",") if item.strip())
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


def discover_monitors(mss_factory: Callable[[], Any] | None = None) -> tuple[CaptureRegion, ...]:
    """Return physical monitors reported by MSS, excluding its virtual aggregate."""

    if mss_factory is None:
        try:
            import mss
        except ImportError as error:
            raise TesterError(
                "Screen capture requires mss. Install requirements-windows.txt in "
                "the controlled UAV YOLO environment."
            ) from error
        mss_factory = mss.MSS

    try:
        with mss_factory() as grabber:
            raw_monitors = list(grabber.monitors)[1:]
    except Exception as error:
        raise TesterError(f"Could not enumerate monitors: {error}") from error

    monitors = tuple(
        CaptureRegion(
            left=int(item["left"]),
            top=int(item["top"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in raw_monitors
    )
    if not monitors:
        raise TesterError("No active monitors were reported by MSS")
    return monitors


def choose_preview_position(
    capture_region: CaptureRegion,
    monitors: Sequence[CaptureRegion],
    preview_width: int,
    preview_height: int,
    margin: int = 20,
) -> tuple[int, int, bool]:
    """Choose a preview origin, preferring a monitor outside the capture area."""

    if preview_width <= 0 or preview_height <= 0:
        raise TesterError("Preview dimensions must be positive")

    for monitor in monitors:
        if not rectangles_intersect(capture_region, monitor):
            return monitor.left + margin, monitor.top + margin, False

    containing = next(
        (monitor for monitor in monitors if contains_rectangle(monitor, capture_region)),
        monitors[0] if monitors else capture_region,
    )
    candidates = (
        (capture_region.right + margin, capture_region.top),
        (capture_region.left - preview_width - margin, capture_region.top),
        (capture_region.left, capture_region.bottom + margin),
        (capture_region.left, capture_region.top - preview_height - margin),
    )
    for left, top in candidates:
        preview = CaptureRegion(left, top, preview_width, preview_height)
        if contains_rectangle(containing, preview) and not rectangles_intersect(
            capture_region, preview
        ):
            return left, top, False

    return containing.left + margin, containing.top + margin, True


def resolve_model_path(
    cli_value: str | None, environment: Mapping[str, str] | None = None
) -> Path:
    """Resolve an explicit model path or the ``UAV_MODEL_PATH`` fallback."""

    source = os.environ if environment is None else environment
    raw_value = str(cli_value or source.get("UAV_MODEL_PATH", "")).strip()
    if not raw_value:
        raise TesterError("Provide --model or set UAV_MODEL_PATH to a trusted .pt file")

    candidate = Path(raw_value).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TesterError(f"Model checkpoint does not exist: {candidate}") from error
    if not resolved.is_file():
        raise TesterError(f"Model checkpoint is not a file: {resolved}")
    if resolved.suffix.lower() != ".pt":
        raise TesterError("Model checkpoint must be a trusted .pt file")
    return resolved


def windows_desktop_path() -> Path:
    """Resolve the current user's Windows Desktop, including OneDrive redirection."""

    if os.name == "nt":
        try:
            import winreg

            key_path = (
                r"Software\Microsoft\Windows\CurrentVersion\Explorer"
                r"\User Shell Folders"
            )
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(value))).expanduser()
        except (OSError, ImportError):
            pass
    return Path.home() / "Desktop"


def discover_first_mp4(
    search_directories: Sequence[Path] | None = None,
) -> Path:
    """Find the newest MP4 in the first search directory containing one."""

    directories = search_directories or (
        PROJECT_ROOT,
        windows_desktop_path(),
        Path.home() / "Videos",
        Path.cwd(),
    )
    visited: set[Path] = set()
    for raw_directory in directories:
        directory = Path(raw_directory).expanduser()
        try:
            resolved_directory = directory.resolve(strict=True)
        except OSError:
            continue
        if resolved_directory in visited or not resolved_directory.is_dir():
            continue
        visited.add(resolved_directory)
        try:
            candidates = [
                path
                for path in resolved_directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            ]
        except OSError:
            continue
        if candidates:
            ranked: list[tuple[int, str, Path]] = []
            for path in candidates:
                try:
                    ranked.append(
                        (-path.stat().st_mtime_ns, path.name.casefold(), path.resolve())
                    )
                except OSError:
                    continue
            if ranked:
                ranked.sort()
                return ranked[0][2]
    raise TesterError(
        "No MP4 video was found in the repository root, Windows Desktop, "
        "user Videos directory, or current directory. Provide --video instead."
    )


def resolve_video_path(
    cli_value: Path | None,
    *,
    auto_detect: bool,
    environment: Mapping[str, str] | None = None,
    search_directories: Sequence[Path] | None = None,
) -> Path:
    """Resolve an explicit/configured MP4 or automatically discover one."""

    source = os.environ if environment is None else environment
    raw_value = str(cli_value or source.get("UAV_VIDEO_PATH", "")).strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise TesterError(f"Video file does not exist: {candidate}") from error
        if not resolved.is_file() or resolved.suffix.lower() != ".mp4":
            raise TesterError(f"Video source must be an existing MP4 file: {resolved}")
        return resolved
    if auto_detect:
        return discover_first_mp4(search_directories)
    raise TesterError("Provide --video, use --auto-video, or set UAV_VIDEO_PATH")


def select_video_with_dialog(
    *,
    tk_factory: Callable[[], Any] | None = None,
    askopenfilename: Callable[..., str] | None = None,
) -> Path:
    """Open a native file picker and return the user-selected local MP4."""

    if tk_factory is None or askopenfilename is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise TesterError(
                "The video picker requires Tkinter in the selected Python environment"
            ) from error
        tk_factory = tk.Tk
        askopenfilename = filedialog.askopenfilename

    try:
        root = tk_factory()
    except Exception as error:
        raise TesterError(f"Could not create the video selection dialog: {error}") from error

    try:
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = askopenfilename(
            parent=root,
            title="Select an MP4 video for UAV model testing",
            filetypes=(("MP4 video", "*.mp4"),),
        )
    except Exception as error:
        raise TesterError(f"Video selection dialog failed: {error}") from error
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if not selected:
        raise TesterError("Video selection was cancelled")
    return resolve_video_path(
        Path(selected), auto_detect=False, environment={}
    )


def resolve_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
    cuda_device_count: int | None = None,
) -> str:
    """Resolve automatic, CPU, or explicit CUDA execution."""

    normalized = str(requested).strip().lower()
    if not normalized:
        normalized = "auto"

    if cuda_available is None or cuda_device_count is None:
        try:
            import torch
        except ImportError as error:
            raise TesterError(
                "PyTorch is unavailable in the selected Python environment"
            ) from error
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())

    if normalized == "auto":
        return "0" if cuda_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"gpu", "cuda"}:
        normalized = "0"
    elif normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]

    if normalized.isdigit():
        index = int(normalized)
        if not cuda_available:
            raise TesterError(
                "CUDA was requested but is unavailable; use --device auto or --device cpu"
            )
        if index >= int(cuda_device_count):
            raise TesterError(
                f"CUDA device {index} is unavailable; detected {cuda_device_count} device(s)"
            )
        return str(index)

    raise TesterError("--device must be auto, cpu, gpu, cuda, cuda:N, or a GPU index")


def device_display_name(device: str) -> str:
    """Return a concise human-facing label for a resolved inference device."""

    return "CPU" if device == "cpu" else f"CUDA:{device}"


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
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def safe_overlay_text(value: object, maximum_length: int = 52) -> str:
    """Make external model labels safe and bounded for a single-line overlay."""

    text = " ".join(str(value).replace("\x00", "").split())
    if not text:
        return "unknown"
    if len(text) <= maximum_length:
        return text
    return text[: max(1, maximum_length - 1)] + "…"


def _scalar(value: Any) -> float:
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
        return _scalar(current[0])
    return float(current)


def _coordinates(value: Any) -> Sequence[float]:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        current = current.tolist()
    if (
        isinstance(current, Sequence)
        and current
        and isinstance(current[0], Sequence)
    ):
        current = current[0]
    return current


def class_name_for(names: Any, class_id: int) -> str:
    """Read a class label from Ultralytics list- or mapping-style names."""

    try:
        if isinstance(names, Mapping):
            value = names.get(class_id, names.get(str(class_id), class_id))
        else:
            value = names[class_id]
    except (IndexError, KeyError, TypeError):
        value = class_id
    return safe_overlay_text(value)


class PerformanceTracker:
    """Track rolling capture, inference, total time, and displayed FPS."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise TesterError("Metrics window size must be positive")
        self._capture_ms: deque[float] = deque(maxlen=window_size)
        self._inference_ms: deque[float] = deque(maxlen=window_size)
        self._total_ms: deque[float] = deque(maxlen=window_size)
        self._frame_intervals: deque[float] = deque(maxlen=window_size)
        self._last_frame_timestamp: float | None = None

    def record(
        self,
        *,
        capture_ms: float,
        inference_ms: float,
        total_ms: float,
        frame_timestamp: float,
    ) -> FrameMetrics:
        values = (capture_ms, inference_ms, total_ms, frame_timestamp)
        if not all(math.isfinite(float(value)) for value in values):
            raise TesterError("Frame timings must be finite")
        if capture_ms < 0 or inference_ms < 0 or total_ms < 0:
            raise TesterError("Frame timings cannot be negative")

        if self._last_frame_timestamp is not None:
            interval = frame_timestamp - self._last_frame_timestamp
            if interval > 0:
                self._frame_intervals.append(interval)
        self._last_frame_timestamp = frame_timestamp
        self._capture_ms.append(float(capture_ms))
        self._inference_ms.append(float(inference_ms))
        self._total_ms.append(float(total_ms))
        return self.snapshot()

    def snapshot(self) -> FrameMetrics:
        def average(values: deque[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        average_interval = average(self._frame_intervals)
        return FrameMetrics(
            fps=(1.0 / average_interval) if average_interval > 0 else 0.0,
            capture_ms=average(self._capture_ms),
            inference_ms=average(self._inference_ms),
            total_ms=average(self._total_ms),
        )


class MSSScreenSource:
    """Capture one fixed desktop region and convert MSS BGRA frames to BGR."""

    def __init__(
        self,
        region: CaptureRegion,
        *,
        mss_factory: Callable[[], Any] | None = None,
        cv2_module: Any | None = None,
        numpy_module: Any | None = None,
    ) -> None:
        self.region = region
        self._mss_factory = mss_factory
        self._cv2 = cv2_module
        self._numpy = numpy_module
        self._grabber: Any | None = None

    def __enter__(self) -> "MSSScreenSource":
        if self._mss_factory is None:
            try:
                import mss
            except ImportError as error:
                raise TesterError(
                    "Screen capture requires mss. Install requirements-windows.txt."
                ) from error
            self._mss_factory = mss.MSS
        try:
            self._grabber = self._mss_factory()
        except Exception as error:
            raise TesterError(f"Could not initialize screen capture: {error}") from error
        return self

    def read(self) -> Any:
        if self._grabber is None:
            raise TesterError("Screen source is not open")
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        if self._numpy is None:
            import numpy as np

            self._numpy = np
        try:
            bgra = self._numpy.asarray(self._grabber.grab(self.region.as_mss_dict()))
            if bgra.ndim != 3 or bgra.shape[2] != 4:
                raise TesterError(f"Unexpected MSS frame shape: {getattr(bgra, 'shape', None)}")
            return self._cv2.cvtColor(bgra, self._cv2.COLOR_BGRA2BGR)
        except TesterError:
            raise
        except Exception as error:
            raise TesterError(f"Screen capture failed: {error}") from error

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._grabber is not None:
            try:
                self._grabber.close()
            except Exception:
                pass
        self._grabber = None


def inspect_video(video_path: Path, cv2_module: Any) -> VideoMetadata:
    """Open a local MP4 long enough to validate and read its metadata."""

    capture = cv2_module.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise TesterError(f"OpenCV could not open video: {video_path}")
        width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise TesterError(f"Video reports invalid dimensions: {video_path}")
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        return VideoMetadata(video_path, width, height, fps, max(0, frame_count))
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"Could not inspect video {video_path}: {error}") from error
    finally:
        capture.release()


class VideoFileSource:
    """Sequential local MP4 frame source with optional end-of-file looping."""

    def __init__(
        self,
        metadata: VideoMetadata,
        *,
        loop: bool = False,
        cv2_module: Any | None = None,
    ) -> None:
        self.metadata = metadata
        self.loop = loop
        self._cv2 = cv2_module
        self._capture: Any | None = None

    def __enter__(self) -> "VideoFileSource":
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        self._capture = self._cv2.VideoCapture(str(self.metadata.path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise TesterError(f"OpenCV could not open video: {self.metadata.path}")
        return self

    def read(self) -> Any:
        if self._capture is None:
            raise TesterError("Video source is not open")
        try:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                return frame
            if self.loop:
                self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    return frame
        except Exception as error:
            raise TesterError(f"Video decoding failed: {error}") from error
        raise VideoSourceEnded

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None


class YoloDetector:
    """Trusted Ultralytics loader and inference adapter."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        registry_path: Path | None = None,
        verifier: Callable[..., str] = verify_trusted_model,
        model_loader: Callable[..., Any] = load_trusted_yolo,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

        # Verification deliberately precedes any Ultralytics/PyTorch deserialization.
        self.model_sha256 = verifier(model_path, registry_path)
        try:
            self.model = model_loader(model_path, registry_path=registry_path)
        except ModelIntegrityError:
            raise
        except Exception as error:
            raise TesterError(f"Trusted YOLO model could not be loaded: {error}") from error

    def detect(self, frame: Any) -> list[DetectionResult]:
        try:
            raw_results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as error:
            raise TesterError(f"YOLO inference failed: {error}") from error

        if not raw_results:
            return []
        result = raw_results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            raise TesterError("Detector received a frame without valid dimensions")
        frame_height, frame_width = int(shape[0]), int(shape[1])
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        detections: list[DetectionResult] = []
        for box in boxes:
            try:
                bbox = clip_bbox(_coordinates(box.xyxy), frame_width, frame_height)
                confidence = _scalar(box.conf)
                class_id = int(_scalar(box.cls))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if bbox is None or not math.isfinite(confidence):
                continue
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=class_name_for(names, class_id),
                    confidence=max(0.0, min(1.0, confidence)),
                    bbox=bbox,
                )
            )
        return detections


class OverlayRenderer:
    """Draw clean detection boxes, labels, and a compact performance HUD."""

    _PALETTE = (
        (0, 220, 255),
        (255, 165, 0),
        (80, 220, 100),
        (255, 120, 210),
        (100, 190, 255),
        (220, 180, 70),
    )

    def __init__(self, cv2_module: Any | None = None) -> None:
        if cv2_module is None:
            import cv2

            cv2_module = cv2
        self.cv2 = cv2_module

    def draw_detections(
        self, frame: Any, detections: Sequence[DetectionResult]
    ) -> Any:
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        for detection in detections:
            clipped = clip_bbox(detection.bbox, frame_width, frame_height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            color = self._PALETTE[detection.class_id % len(self._PALETTE)]
            self.cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = safe_overlay_text(
                f"{detection.class_name} {detection.confidence:.2f}"
            )
            (text_width, text_height), baseline = self.cv2.getTextSize(
                label, self.cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            label_x = min(max(0, x1), max(0, frame_width - text_width - 8))
            label_bottom = y1 - 5 if y1 >= text_height + baseline + 8 else min(
                frame_height - baseline - 1, y1 + text_height + baseline + 8
            )
            label_top = max(0, label_bottom - text_height - baseline - 5)
            self.cv2.rectangle(
                annotated,
                (label_x, label_top),
                (min(frame_width - 1, label_x + text_width + 7), label_bottom + 1),
                color,
                -1,
            )
            self.cv2.putText(
                annotated,
                label,
                (label_x + 3, max(text_height, label_bottom - baseline - 2)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (15, 15, 15),
                1,
                self.cv2.LINE_AA,
            )
        return annotated

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
    ) -> Any:
        source_text = source_label or (
            f"region {region.left},{region.top} {region.width}x{region.height}"
        )
        lines = (
            "LIVE UAV AI MODEL TESTER",
            f"FPS {metrics.fps:5.1f} | capture {metrics.capture_ms:5.1f} ms | "
            f"infer {metrics.inference_ms:5.1f} ms | frame {metrics.total_ms:5.1f} ms",
            f"model {safe_overlay_text(model_name, 36)} | device {device} | "
            f"detections {detection_count}",
            safe_overlay_text(source_text, 88),
            "Q/ESC quit | P pause | S screenshot | H HUD",
        )
        height = 24 + len(lines) * 24
        width = min(frame.shape[1] - 1, 790)
        if width > 0:
            overlay = frame.copy()
            self.cv2.rectangle(overlay, (0, 0), (width, height), (12, 12, 12), -1)
            self.cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        for index, line in enumerate(lines):
            self.cv2.putText(
                frame,
                line,
                (12, 24 + index * 24),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (235, 235, 235),
                1,
                self.cv2.LINE_AA,
            )
        if paused:
            self.cv2.putText(
                frame,
                "PAUSED",
                (max(10, frame.shape[1] // 2 - 85), max(45, frame.shape[0] // 2)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 220, 255),
                3,
                self.cv2.LINE_AA,
            )
        return frame


class FrameProcessor:
    """Coordinate inference, display filtering, metrics, and rendering."""

    def __init__(
        self,
        detector: DetectorProtocol,
        renderer: RendererProtocol,
        *,
        allowed_classes: frozenset[str] | None,
        model_name: str,
        region: CaptureRegion,
        device: str,
        source_label: str | None = None,
        tracker: PerformanceTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.detector = detector
        self.renderer = renderer
        self.allowed_classes = allowed_classes
        self.model_name = model_name
        self.region = region
        self.device = device
        self.source_label = source_label
        self.tracker = tracker or PerformanceTracker()
        self.clock = clock

    def process(
        self,
        frame: Any,
        *,
        capture_ms: float,
        frame_started_at: float,
        frame_timestamp: float,
        hud_visible: bool,
    ) -> FrameOutcome:
        inference_started = self.clock()
        detections = self.detector.detect(frame)
        inference_ms = (self.clock() - inference_started) * 1000.0
        displayed = filter_detections(detections, self.allowed_classes)
        annotated = self.renderer.draw_detections(frame, displayed)
        total_ms = (self.clock() - frame_started_at) * 1000.0
        metrics = self.tracker.record(
            capture_ms=capture_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
            frame_timestamp=frame_timestamp,
        )
        if hud_visible:
            self.renderer.draw_hud(
                annotated,
                metrics=metrics,
                model_name=self.model_name,
                region=self.region,
                device=self.device,
                detection_count=len(displayed),
                source_label=self.source_label,
            )
        return FrameOutcome(annotated, displayed, metrics)


class ScreenshotStore:
    """Create unique timestamped screenshots without overwriting prior output."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = Path(output_directory).expanduser()

    def next_path(self, when: datetime | None = None) -> Path:
        timestamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
        base = self.output_directory / f"live_screen_{timestamp}.png"
        candidate = base
        counter = 1
        while candidate.exists():
            candidate = base.with_name(f"{base.stem}_{counter}{base.suffix}")
            counter += 1
        return candidate

    def save(self, frame: Any, cv2_module: Any) -> Path:
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TesterError(
                f"Could not create screenshot directory: {self.output_directory}"
            ) from error
        path = self.next_path()
        try:
            saved = bool(cv2_module.imwrite(str(path), frame))
        except Exception as error:
            raise TesterError(f"Could not save screenshot: {error}") from error
        if not saved:
            raise TesterError(f"Could not save screenshot: {path}")
        return path


def scaled_preview_size(
    region: CaptureRegion,
    maximum_width: int = 1280,
    maximum_height: int = 720,
) -> tuple[int, int]:
    """Return a preview size that preserves aspect ratio and bounds window size."""

    scale = min(1.0, maximum_width / region.width, maximum_height / region.height)
    return max(1, int(region.width * scale)), max(1, int(region.height * scale))


def select_region_interactively(
    monitor_region: CaptureRegion,
    *,
    cv2_module: Any,
    source_factory: Callable[[CaptureRegion], Any] = MSSScreenSource,
) -> CaptureRegion:
    """Capture a monitor once and let the user select a region with OpenCV."""

    selection_window = f"{WINDOW_NAME} - Select Region"
    try:
        with source_factory(monitor_region) as source:
            frame = source.read()
        x, y, width, height = cv2_module.selectROI(
            selection_window, frame, showCrosshair=True, fromCenter=False
        )
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"Interactive region selection failed: {error}") from error
    finally:
        try:
            cv2_module.destroyWindow(selection_window)
        except Exception:
            pass

    selected = CaptureRegion(
        monitor_region.left + int(x),
        monitor_region.top + int(y),
        int(width),
        int(height),
    )
    if selected.width <= 0 or selected.height <= 0:
        raise TesterError("Region selection was cancelled or empty")
    if not contains_rectangle(monitor_region, selected):
        raise TesterError("Selected region falls outside the chosen monitor")
    return selected


def validate_numeric_options(args: argparse.Namespace) -> None:
    """Validate numeric inference and pacing arguments."""

    if not 0.0 <= args.conf <= 1.0:
        raise TesterError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise TesterError("--iou must be between 0 and 1")
    if args.imgsz <= 0 or args.imgsz > 8192:
        raise TesterError("--imgsz must be between 1 and 8192")
    if not math.isfinite(args.max_fps) or args.max_fps < 0 or args.max_fps > 240:
        raise TesterError("--max-fps must be between 0 and 240 (0 disables throttling)")

    video_mode = args.video is not None or args.auto_video or args.select_video
    manual_region = any(
        value is not None for value in (args.left, args.top, args.width, args.height)
    )
    if args.test_frame and video_mode:
        raise TesterError("--test-frame cannot be combined with --video or --auto-video")
    if video_mode and (manual_region or args.select_region):
        raise TesterError("Video mode cannot be combined with screen-region options")
    if args.loop_video and not video_mode:
        raise TesterError("--loop-video requires --video or --auto-video")


def capture_region_from_args(
    args: argparse.Namespace,
    monitors: Sequence[CaptureRegion],
    *,
    cv2_module: Any,
) -> CaptureRegion:
    """Resolve manual, interactive, or full-monitor capture configuration."""

    manual_values = (args.left, args.top, args.width, args.height)
    manual_supplied = [value is not None for value in manual_values]
    if any(manual_supplied) and not all(manual_supplied):
        raise TesterError("Specify --left, --top, --width, and --height together")
    if all(manual_supplied):
        if args.select_region:
            raise TesterError("Manual coordinates cannot be combined with --select-region")
        return validate_manual_region(
            CaptureRegion(*(int(value) for value in manual_values)), monitors
        )

    monitor = monitor_by_number(monitors, args.monitor)
    if args.select_region:
        return select_region_interactively(monitor, cv2_module=cv2_module)
    return monitor


def run_test_frame(
    frame_path: Path,
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    cv2_module: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Run a single local image through loading, inference, rendering, and save."""

    try:
        resolved = frame_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise TesterError(f"Test frame does not exist: {frame_path}") from error
    frame = cv2_module.imread(str(resolved))
    if frame is None:
        raise TesterError(f"OpenCV could not decode test frame: {resolved}")
    started = clock()
    outcome = processor.process(
        frame,
        capture_ms=0.0,
        frame_started_at=started,
        frame_timestamp=started,
        hud_visible=True,
    )
    return screenshot_store.save(outcome.annotated_frame, cv2_module)


def run_live_preview(
    region: CaptureRegion,
    monitors: Sequence[CaptureRegion],
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    max_fps: float,
    cv2_module: Any,
    source_factory: Callable[[CaptureRegion], Any] = MSSScreenSource,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Run the keyboard-controlled capture, inference, and preview loop."""

    try:
        preview_width, preview_height = scaled_preview_size(region)
        window_left, window_top, overlap_expected = choose_preview_position(
            region, monitors, preview_width, preview_height
        )
        cv2_module.namedWindow(WINDOW_NAME, cv2_module.WINDOW_NORMAL)
        cv2_module.resizeWindow(WINDOW_NAME, preview_width, preview_height)
        cv2_module.moveWindow(WINDOW_NAME, window_left, window_top)
        if overlap_expected:
            print(
                "Warning: no non-overlapping preview position is available. "
                "The preview may appear in the captured image; move it to another "
                "monitor if possible."
            )

        paused = False
        hud_visible = True
        last_frame: Any | None = None
        last_detection_count = 0
        target_period = (1.0 / max_fps) if max_fps > 0 else 0.0

        with source_factory(region) as source:
            while True:
                loop_started = clock()
                if not paused:
                    capture_started = clock()
                    frame = source.read()
                    capture_ms = (clock() - capture_started) * 1000.0
                    outcome = processor.process(
                        frame,
                        capture_ms=capture_ms,
                        frame_started_at=loop_started,
                        frame_timestamp=loop_started,
                        hud_visible=hud_visible,
                    )
                    last_frame = outcome.annotated_frame
                    last_detection_count = len(outcome.detections)

                if last_frame is not None:
                    display_frame = last_frame.copy()
                    if paused and hud_visible:
                        processor.renderer.draw_hud(
                            display_frame,
                            metrics=processor.tracker.snapshot(),
                            model_name=processor.model_name,
                            region=region,
                            device=processor.device,
                            detection_count=last_detection_count,
                            source_label=processor.source_label,
                            paused=True,
                        )
                    cv2_module.imshow(WINDOW_NAME, display_frame)

                key = cv2_module.waitKey(30 if paused else 1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key in (ord("h"), ord("H")):
                    hud_visible = not hud_visible
                    print("HUD on" if hud_visible else "HUD off")
                elif key in (ord("s"), ord("S")):
                    if last_frame is None:
                        print("No frame is available to save yet.")
                    else:
                        saved_path = screenshot_store.save(last_frame, cv2_module)
                        print(f"Screenshot saved: {saved_path}")

                try:
                    if cv2_module.getWindowProperty(
                        WINDOW_NAME, cv2_module.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except Exception:
                    pass

                if target_period > 0 and not paused:
                    delay = target_period - (clock() - loop_started)
                    if delay > 0:
                        sleeper(delay)
    except KeyboardInterrupt:
        print("Interrupted; closing preview.")
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"OpenCV preview failed: {error}") from error
    finally:
        try:
            cv2_module.destroyAllWindows()
        except Exception:
            pass


def run_video_preview(
    source: VideoFileSource,
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    max_fps: float,
    cv2_module: Any,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Play a local video through the detector until EOF or user exit."""

    region = processor.region
    preview_width, preview_height = scaled_preview_size(region)
    try:
        cv2_module.namedWindow(WINDOW_NAME, cv2_module.WINDOW_NORMAL)
        cv2_module.resizeWindow(WINDOW_NAME, preview_width, preview_height)
        paused = False
        hud_visible = True
        last_frame: Any | None = None
        last_detection_count = 0
        playback_fps = source.metadata.fps
        if max_fps > 0:
            playback_fps = min(playback_fps, max_fps)
        target_period = 1.0 / playback_fps

        with source:
            while True:
                loop_started = clock()
                if not paused:
                    decode_started = clock()
                    try:
                        frame = source.read()
                    except VideoSourceEnded:
                        print("Video reached the end; closing preview.")
                        break
                    decode_ms = (clock() - decode_started) * 1000.0
                    outcome = processor.process(
                        frame,
                        capture_ms=decode_ms,
                        frame_started_at=loop_started,
                        frame_timestamp=loop_started,
                        hud_visible=hud_visible,
                    )
                    last_frame = outcome.annotated_frame
                    last_detection_count = len(outcome.detections)

                if last_frame is not None:
                    display_frame = last_frame.copy()
                    if paused and hud_visible:
                        processor.renderer.draw_hud(
                            display_frame,
                            metrics=processor.tracker.snapshot(),
                            model_name=processor.model_name,
                            region=region,
                            device=processor.device,
                            detection_count=last_detection_count,
                            source_label=processor.source_label,
                            paused=True,
                        )
                    cv2_module.imshow(WINDOW_NAME, display_frame)

                key = cv2_module.waitKey(30 if paused else 1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key in (ord("h"), ord("H")):
                    hud_visible = not hud_visible
                    print("HUD on" if hud_visible else "HUD off")
                elif key in (ord("s"), ord("S")):
                    if last_frame is None:
                        print("No frame is available to save yet.")
                    else:
                        saved_path = screenshot_store.save(last_frame, cv2_module)
                        print(f"Screenshot saved: {saved_path}")

                try:
                    if cv2_module.getWindowProperty(
                        WINDOW_NAME, cv2_module.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except Exception:
                    pass

                if not paused:
                    delay = target_period - (clock() - loop_started)
                    if delay > 0:
                        sleeper(delay)
    except KeyboardInterrupt:
        print("Interrupted; closing preview.")
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"OpenCV video preview failed: {error}") from error
    finally:
        try:
            cv2_module.destroyAllWindows()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Read a local MP4 or capture a Windows desktop region, run a trusted "
            "local YOLO model, and show a local OpenCV preview."
        )
    )
    parser.add_argument("--model", help="Trusted local .pt model (or set UAV_MODEL_PATH)")
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--video", type=Path, help="Read frames from one local MP4")
    video_group.add_argument(
        "--auto-video",
        action="store_true",
        help="Use the newest MP4 from the repository, Desktop, Videos, or current directory",
    )
    video_group.add_argument(
        "--select-video",
        action="store_true",
        help="Open a file picker and require an MP4 selection",
    )
    parser.add_argument(
        "--loop-video", action="store_true", help="Restart the selected video at EOF"
    )
    parser.add_argument("--monitor", type=int, default=1, help="One-based monitor number")
    parser.add_argument(
        "--list-monitors", action="store_true", help="List active monitors and exit"
    )
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Interactively select a region on --monitor before inference",
    )
    parser.add_argument("--left", type=int, help="Manual capture region left coordinate")
    parser.add_argument("--top", type=int, help="Manual capture region top coordinate")
    parser.add_argument("--width", type=int, help="Manual capture region width")
    parser.add_argument("--height", type=int, help="Manual capture region height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:N, or a GPU index"
    )
    parser.add_argument(
        "--max-fps", type=float, default=0.0, help="Optional FPS cap; 0 means uncapped"
    )
    parser.add_argument(
        "--classes",
        default="all",
        help="Display all or a comma-separated class list",
    )
    parser.add_argument(
        "--tank-only",
        action="store_true",
        help=f"Display only the primary class '{TANK_CLASS_NAME}'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Screenshot output directory",
    )
    parser.add_argument(
        "--test-frame",
        type=Path,
        help="Process one local image, save the rendered result, and exit",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional trusted model SHA-256 registry override",
    )
    return parser


def print_monitors(monitors: Sequence[CaptureRegion]) -> None:
    """Print physical monitor indices and bounds."""

    for index, monitor in enumerate(monitors, start=1):
        print(
            f"Monitor {index}: left={monitor.left} top={monitor.top} "
            f"width={monitor.width} height={monitor.height}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_numeric_options(args)
        if args.list_monitors:
            print_monitors(discover_monitors())
            return 0

        model_path = resolve_model_path(args.model)
        device = resolve_device(args.device)
        allowed_classes = parse_class_filter(args.classes, tank_only=args.tank_only)

        try:
            import cv2
        except ImportError as error:
            raise TesterError(
                "OpenCV is unavailable. Install requirements-windows.txt in the "
                "controlled UAV YOLO environment."
            ) from error

        video_path: Path | None = None
        video_metadata: VideoMetadata | None = None
        if args.test_frame:
            frame = cv2.imread(str(args.test_frame.expanduser()))
            if frame is None:
                raise TesterError(f"OpenCV could not decode test frame: {args.test_frame}")
            frame_height, frame_width = frame.shape[:2]
            region = CaptureRegion(0, 0, frame_width, frame_height)
        elif args.video is not None or args.auto_video or args.select_video:
            if args.select_video:
                video_path = select_video_with_dialog()
            else:
                video_path = resolve_video_path(
                    args.video, auto_detect=args.auto_video
                )
            video_metadata = inspect_video(video_path, cv2)
            region = CaptureRegion(
                0, 0, video_metadata.width, video_metadata.height
            )
        else:
            monitors = discover_monitors()
            print_monitors(monitors)
            region = capture_region_from_args(args, monitors, cv2_module=cv2)

        detector = YoloDetector(
            model_path,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=device,
            registry_path=args.registry,
        )
        renderer = OverlayRenderer(cv2)
        display_device = device_display_name(device)
        processor = FrameProcessor(
            detector,
            renderer,
            allowed_classes=allowed_classes,
            model_name=model_path.name,
            region=region,
            device=display_device,
            source_label=(f"video {video_path.name}" if video_path else None),
        )
        screenshot_store = ScreenshotStore(args.output_dir)

        filter_summary = "all" if allowed_classes is None else ", ".join(sorted(allowed_classes))
        print(f"Model: {model_path}")
        print(f"Verified SHA-256: {detector.model_sha256}")
        if args.test_frame:
            capture_mode = "test frame"
        elif video_path is not None:
            if args.select_video:
                capture_mode = "file-picker video"
            elif args.auto_video:
                capture_mode = "automatically selected video"
            else:
                capture_mode = "video file"
        elif args.select_region:
            capture_mode = f"interactive selection on monitor {args.monitor}"
        elif args.left is not None:
            capture_mode = "manual region"
        else:
            capture_mode = f"full monitor {args.monitor}"
        print(f"Device: {display_device}")
        print(f"Capture mode: {capture_mode}")
        if video_metadata is not None:
            print(f"Video: {video_metadata.path}")
            print(
                f"Video metadata: {video_metadata.width}x{video_metadata.height} "
                f"at {video_metadata.fps:.3f} FPS, {video_metadata.frame_count} frames"
            )
        print(
            f"Capture region: left={region.left} top={region.top} "
            f"width={region.width} height={region.height}"
        )
        print(
            f"Inference settings: conf={args.conf:.3f} iou={args.iou:.3f} "
            f"imgsz={args.imgsz} max_fps={args.max_fps:g}"
        )
        print(f"Display classes: {filter_summary}")

        if args.test_frame:
            saved = run_test_frame(
                args.test_frame, processor, screenshot_store, cv2_module=cv2
            )
            print(f"Test-frame result saved: {saved}")
            return 0

        if video_metadata is not None:
            run_video_preview(
                VideoFileSource(
                    video_metadata, loop=args.loop_video, cv2_module=cv2
                ),
                processor,
                screenshot_store,
                max_fps=args.max_fps,
                cv2_module=cv2,
            )
            return 0

        run_live_preview(
            region,
            monitors,
            processor,
            screenshot_store,
            max_fps=args.max_fps,
            cv2_module=cv2,
        )
        return 0
    except (TesterError, ModelIntegrityError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
