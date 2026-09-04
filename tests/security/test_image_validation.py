from __future__ import annotations

import cv2
import numpy as np
import pytest

from uav_security.config import SecurityLimits
from uav_security.image_validation import ImageValidationError, decode_and_validate_jpeg


def encoded_jpeg(width=32, height=24) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_valid_jpeg_decodes_with_expected_shape():
    frame = decode_and_validate_jpeg(encoded_jpeg())
    assert frame.shape == (24, 32, 3)


def test_non_jpeg_is_rejected_before_decode():
    with pytest.raises(ImageValidationError, match="not a JPEG"):
        decode_and_validate_jpeg(b"not-a-jpeg")


def test_oversized_dimensions_are_rejected_before_decode():
    limits = SecurityLimits(max_image_width=16, max_image_height=64, max_image_pixels=4096)
    with pytest.raises(ImageValidationError, match="dimensions"):
        decode_and_validate_jpeg(encoded_jpeg(width=32), limits=limits)
