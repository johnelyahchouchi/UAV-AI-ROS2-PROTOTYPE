"""Uploaded-video source adapter with an interface for future source types."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .errors import DashboardError


@dataclass(frozen=True)
class VideoMetadata:
    """Validated metadata required for progress, timestamps, and output."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        """Return duration when frame count is known."""
        if self.frame_count <= 0:
            return None
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoFrame:
    """One decoded video frame and its source-media timing."""

    number: int
    timestamp_seconds: float
    image: np.ndarray


class FrameSource(Protocol):
    """Interface implemented by uploaded videos and future camera/RTSP adapters."""

    @property
    def metadata(self) -> VideoMetadata:
        """Return validated source metadata."""

    def frames(self) -> Iterator[VideoFrame]:
        """Yield decoded frames in source order."""

    def close(self) -> None:
        """Release source resources."""


class UploadedVideoSource:
    """OpenCV-backed source for one uploaded local video file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise DashboardError(
                "VIDEO_OPEN_FAILED",
                f"OpenCV could not open the selected video: {self.path.name}",
                recovery="Choose a valid video with a codec supported by this OpenCV build.",
            )

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            self.close()
            raise DashboardError(
                "VIDEO_METADATA_INVALID",
                "The video reports invalid frame dimensions.",
            )
        if not math.isfinite(fps) or fps <= 0:
            self.close()
            raise DashboardError(
                "VIDEO_METADATA_INVALID",
                "The video reports an invalid or missing frame rate.",
                recovery="Re-encode the video with a valid constant frame rate.",
            )

        success, first_frame = self._capture.read()
        if not success or first_frame is None or first_frame.size == 0:
            self.close()
            raise DashboardError(
                "VIDEO_NO_READABLE_FRAMES",
                "The video opened but no frame could be decoded.",
                recovery="Check that the file is not corrupt and uses a supported codec.",
            )
        if first_frame.shape[1] != width or first_frame.shape[0] != height:
            width = int(first_frame.shape[1])
            height = int(first_frame.shape[0])

        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=max(0, frame_count),
        )
        self._first_frame = first_frame
        self._closed = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[VideoFrame]:
        """Yield every frame exactly once with a stable timestamp."""
        if self._closed:
            raise DashboardError(
                "VIDEO_SOURCE_CLOSED",
                "The uploaded video source is already closed.",
            )

        frame_number = 1
        yield VideoFrame(
            number=frame_number,
            timestamp_seconds=0.0,
            image=self._first_frame,
        )

        while True:
            success, frame = self._capture.read()
            if not success:
                if (
                    self._metadata.frame_count > 0
                    and frame_number < self._metadata.frame_count
                ):
                    raise DashboardError(
                        "VIDEO_DECODE_FAILED",
                        (
                            f"Video decoding stopped at frame {frame_number} "
                            f"before the reported {self._metadata.frame_count} frames."
                        ),
                        recovery="Check whether the video is truncated or uses an unsupported codec.",
                    )
                break
            if frame is None or frame.size == 0:
                raise DashboardError(
                    "VIDEO_DECODE_FAILED",
                    f"OpenCV returned an invalid frame at frame {frame_number + 1}.",
                )
            frame_number += 1
            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000.0
            if not math.isfinite(timestamp) or timestamp < 0:
                timestamp = (frame_number - 1) / self._metadata.fps
            yield VideoFrame(
                number=frame_number,
                timestamp_seconds=timestamp,
                image=frame,
            )

    def close(self) -> None:
        """Release the OpenCV capture idempotently."""
        if getattr(self, "_closed", False):
            return
        self._capture.release()
        self._closed = True

    def __enter__(self) -> "UploadedVideoSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
