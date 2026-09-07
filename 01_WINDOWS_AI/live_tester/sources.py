"""Screen and local-video frame sources for the live tester."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from .domain import (
    CaptureRegion,
    TesterError,
    VideoMetadata,
    VideoSourceEnded,
    WINDOW_NAME,
    contains_rectangle,
)


class MSSScreenSource:
    """Capture one fixed desktop region and convert MSS BGRA to BGR."""

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
                raise TesterError("Screen capture requires mss") from error
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
                raise TesterError(
                    f"Unexpected MSS frame shape: {getattr(bgra, 'shape', None)}"
                )
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
    """Validate a local MP4 and read its dimensions and timing metadata."""

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
    """Sequential local MP4 frame source with optional looping."""

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


def scaled_preview_size(
    region: CaptureRegion, maximum_width: int = 1280, maximum_height: int = 720
) -> tuple[int, int]:
    """Return an aspect-preserving bounded preview size."""

    scale = min(1.0, maximum_width / region.width, maximum_height / region.height)
    return max(1, int(region.width * scale)), max(1, int(region.height * scale))


def select_region_interactively(
    monitor_region: CaptureRegion,
    *,
    cv2_module: Any,
    source_factory: Callable[[CaptureRegion], Any] = MSSScreenSource,
) -> CaptureRegion:
    """Capture a monitor once and let the operator choose an ROI."""

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
