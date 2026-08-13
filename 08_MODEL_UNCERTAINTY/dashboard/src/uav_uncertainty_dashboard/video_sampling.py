"""Bounded OpenCV video-frame selection for independent V1 analyses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .errors import DashboardError


@dataclass(frozen=True)
class VideoInfo:
    """Video metadata required for safe sampling estimates."""

    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    """One decoded video frame selected for an independent V1 run."""

    selection_index: int
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


def probe_video(path: Path) -> VideoInfo:
    """Open a video and return validated metadata."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DashboardError("VIDEO_OPEN_FAILED", f"OpenCV could not open: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise DashboardError("VIDEO_METADATA_INVALID", "Video metadata is incomplete or invalid.")
        return VideoInfo(fps, frame_count, frame_count / fps, width, height)
    finally:
        capture.release()


def interval_timestamps(
    duration_seconds: float,
    interval_seconds: float,
    maximum_frames: int,
) -> list[float]:
    """Return timestamps starting at zero without exceeding the configured bound."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Video duration must be positive and finite.")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Sampling interval must be positive and finite.")
    if maximum_frames < 1:
        raise ValueError("Maximum sampled frames must be at least 1.")
    count = min(maximum_frames, max(1, int(math.ceil(duration_seconds / interval_seconds))))
    return [index * interval_seconds for index in range(count) if index * interval_seconds < duration_seconds]


def manual_timestamps(value: str, duration_seconds: float, maximum_frames: int) -> list[float]:
    """Parse a comma-separated, unique, bounded timestamp list."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise DashboardError("TIMESTAMPS_INVALID", "Manual timestamps must be comma-separated numbers.") from error
    if not values:
        raise DashboardError("TIMESTAMPS_REQUIRED", "Enter at least one video timestamp.")
    if any(not math.isfinite(item) or item < 0 or item >= duration_seconds for item in values):
        raise DashboardError(
            "TIMESTAMP_OUT_OF_RANGE",
            f"Timestamps must be at least 0 and below {duration_seconds:.3f} seconds.",
        )
    unique = sorted(set(values))
    if len(unique) > maximum_frames:
        raise DashboardError(
            "TOO_MANY_VIDEO_FRAMES",
            f"Manual selection exceeds the maximum of {maximum_frames} frames.",
        )
    return unique


def total_inference_calls(selected_frames: int, perturbation_count: int) -> int:
    """Return selected frames × (one clean + N perturbed predictions)."""
    if selected_frames < 0 or perturbation_count < 1:
        raise ValueError("Frame count must be non-negative and perturbation count positive.")
    return selected_frames * (perturbation_count + 1)


def sample_video_frames(path: Path, timestamps: list[float]) -> list[SampledFrame]:
    """Decode only the requested timestamps; never iterate through every frame."""
    info = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    frames: list[SampledFrame] = []
    try:
        for selection_index, timestamp in enumerate(timestamps, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            success, image = capture.read()
            if not success or image is None:
                raise DashboardError(
                    "VIDEO_FRAME_DECODE_FAILED",
                    f"Could not decode the frame near {timestamp:.3f} seconds.",
                )
            frames.append(
                SampledFrame(
                    selection_index=selection_index,
                    frame_index=int(round(timestamp * info.fps)),
                    timestamp_seconds=timestamp,
                    image=image,
                )
            )
    finally:
        capture.release()
    return frames
