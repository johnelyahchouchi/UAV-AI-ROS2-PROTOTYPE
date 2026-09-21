"""Bounded visual records of actual analysis inputs and completed model passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .presentation_ui import PillowCanvas, Rect, THEME


MAX_VISUAL_SAMPLES = 32


@dataclass(frozen=True)
class FramePreview:
    """Read-only overview and center crop; neither is used as model input."""

    overview: np.ndarray
    zoom: np.ndarray
    crop: tuple[float, float, float, float]


@dataclass(frozen=True)
class SamplePreview:
    """Small visual record of one completed inference, with normalized boxes."""

    pixels: FramePreview
    sample_index: int
    total: int
    family: str
    parameters: str
    boxes: tuple[tuple[float, float, float, float], ...]
    detection_count: int


def frame_preview(frame: np.ndarray) -> FramePreview:
    """Bound preview memory and retain a true 2x center crop before downsampling."""

    height, width = frame.shape[:2]

    def thumbnail(pixels: np.ndarray) -> np.ndarray:
        h, w = pixels.shape[:2]
        factor = min(1.0, 480 / w, 270 / h)
        image = cv2.resize(pixels, (max(1, round(w * factor)), max(1, round(h * factor))))
        image.setflags(write=False)
        return image

    crop_width, crop_height = max(1, width // 2), max(1, height // 2)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return FramePreview(
        thumbnail(frame), thumbnail(frame[top:top + crop_height, left:left + crop_width]),
        (left / width, top / height, crop_width / width, crop_height / height),
    )


def sample_preview(observation: Any) -> SamplePreview:
    """Reduce a real completed observation to display-only pixels and metadata."""

    height, width = observation.image.shape[:2]
    parameters = " | ".join(f"{key}={value:.3g}" for key, value in observation.parameters)
    boxes = tuple(
        (item.bbox[0] / width, item.bbox[1] / height,
         item.bbox[2] / width, item.bbox[3] / height)
        for item in observation.detections
    )
    return SamplePreview(
        frame_preview(observation.image), observation.sample_index, observation.total,
        observation.family.replace("_", " "), parameters or "Unchanged input pixels",
        boxes, len(observation.detections),
    )


def draw_preview(
    canvas: PillowCanvas,
    pixels: FramePreview,
    rect: Rect,
    *,
    zoomed: bool,
    boxes: tuple[tuple[float, float, float, float], ...] = (),
    history: tuple[SamplePreview, ...] = (),
) -> None:
    """Letterbox a preview with actual boxes; avoid decorative blur of input pixels."""

    frame = pixels.zoom if zoomed else pixels.overview
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
    image.thumbnail((rect.width, rect.height))
    left = rect.x + (rect.width - image.width) // 2
    top = rect.y + (rect.height - image.height) // 2
    canvas.rectangle(rect, fill=(4, 9, 17, 255))
    canvas.image.paste(image, (left, top))

    def outline(box: tuple[float, float, float, float], color: tuple[int, ...]) -> None:
        x1, y1, x2, y2 = box
        if zoomed:
            crop_x, crop_y, crop_w, crop_h = pixels.crop
            x1, x2 = (x1 - crop_x) / crop_w, (x2 - crop_x) / crop_w
            y1, y2 = (y1 - crop_y) / crop_h, (y2 - crop_y) / crop_h
        if x2 <= 0 or y2 <= 0 or x1 >= 1 or y1 >= 1:
            return
        x1, y1, x2, y2 = (max(0.0, min(1.0, value)) for value in (x1, y1, x2, y2))
        if x2 <= x1 or y2 <= y1:
            return
        canvas.draw.rectangle(
            (left + x1 * (image.width - 1), top + y1 * (image.height - 1),
             left + x2 * (image.width - 1), top + y2 * (image.height - 1)),
            outline=color, width=1,
        )

    for sample in history:
        for box in sample.boxes:
            outline(box, (*THEME.review, 75))
    for box in boxes:
        outline(box, (*THEME.accent, 255))
