"""Pre-decode and post-decode validation for untrusted JPEG payloads."""

from __future__ import annotations

from typing import Any

from .config import SecurityLimits


SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
STANDALONE_MARKERS = frozenset({0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)})


class ImageValidationError(ValueError):
    """Raised when encoded or decoded image content violates a safety limit."""


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without invoking a native image decoder."""

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ImageValidationError("Payload is not a JPEG image")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in STANDALONE_MARKERS:
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            raise ImageValidationError("JPEG contains a malformed marker segment")
        if marker in SOF_MARKERS:
            if segment_length < 7:
                raise ImageValidationError("JPEG SOF segment is too short")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise ImageValidationError("JPEG declares impossible dimensions")
            return width, height
        if marker == 0xDA:
            break
        offset += segment_length
    raise ImageValidationError("JPEG has no supported start-of-frame marker")


def decode_and_validate_jpeg(
    data: bytes,
    *,
    limits: SecurityLimits | None = None,
    cv2_module: Any = None,
    numpy_module: Any = None,
):
    """Reject oversized dimensions before native decode, then validate output shape."""

    budget = limits or SecurityLimits.from_environment()
    if not isinstance(data, bytes) or not 0 < len(data) <= budget.max_jpeg_size:
        raise ImageValidationError("JPEG encoded size is outside the configured limit")
    width, height = jpeg_dimensions(data)
    if (
        width > budget.max_image_width
        or height > budget.max_image_height
        or width * height > budget.max_image_pixels
    ):
        raise ImageValidationError("JPEG dimensions exceed the configured image limits")
    if cv2_module is None:
        import cv2 as cv2_module
    if numpy_module is None:
        import numpy as numpy_module
    encoded = numpy_module.frombuffer(data, dtype=numpy_module.uint8)
    frame = cv2_module.imdecode(encoded, cv2_module.IMREAD_COLOR)
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise ImageValidationError("OpenCV could not decode a three-channel JPEG")
    decoded_height, decoded_width = frame.shape[:2]
    if (decoded_width, decoded_height) != (width, height):
        raise ImageValidationError("Decoded JPEG dimensions do not match its header")
    return frame
