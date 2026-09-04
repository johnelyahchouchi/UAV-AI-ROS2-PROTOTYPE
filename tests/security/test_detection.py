from __future__ import annotations

import json
import math

import pytest

from uav_security.config import SecurityLimits
from uav_security.detection import (
    DetectionValidationError,
    parse_header_json,
    sanitize_frame_header,
    validate_header_envelope,
)


def test_valid_detection_is_sanitized_and_unknown_fields_are_dropped(valid_header):
    result = sanitize_frame_header(valid_header, 640, 480)
    detection = result["detections"][0]
    assert detection["class_name"] == "BTR"
    assert detection["bbox"] == [10, 20, 100, 120]
    assert "unknown_network_field" not in detection


def test_excessive_detection_count_is_rejected(valid_header, valid_detection):
    valid_header["detections"] = [valid_detection, valid_detection]
    with pytest.raises(DetectionValidationError, match="per-frame"):
        validate_header_envelope(valid_header, limits=SecurityLimits(max_detections=1))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_confidence_is_rejected(valid_header, value):
    valid_header["detections"][0]["confidence"] = value
    with pytest.raises(DetectionValidationError, match="finite"):
        sanitize_frame_header(valid_header, 640, 480)


def test_nonstandard_json_infinity_is_rejected(valid_header):
    encoded = json.dumps(valid_header).replace("0.91", "Infinity").encode()
    with pytest.raises(DetectionValidationError, match="Non-finite"):
        parse_header_json(encoded)


def test_oversized_class_string_is_rejected(valid_header):
    valid_header["detections"][0]["class_name"] = "x" * 257
    with pytest.raises(DetectionValidationError, match="oversized"):
        sanitize_frame_header(valid_header, 640, 480)


def test_malformed_bbox_is_rejected(valid_header):
    valid_header["detections"][0]["bbox"] = [100, 20, 10, 120]
    with pytest.raises(DetectionValidationError, match="reversed"):
        sanitize_frame_header(valid_header, 640, 480)


def test_bbox_is_clamped_to_actual_decoded_image(valid_header):
    valid_header["detections"][0]["bbox"] = [-20, -10, 900, 800]
    result = sanitize_frame_header(valid_header, 640, 480)
    assert result["detections"][0]["bbox"] == [0, 0, 639, 479]


def test_declared_dimensions_must_match_decoded_image(valid_header):
    with pytest.raises(DetectionValidationError, match="do not match"):
        sanitize_frame_header(valid_header, 320, 240)


def test_wrong_protocol_version_is_clear(valid_header):
    valid_header["protocol_version"] = 1
    with pytest.raises(DetectionValidationError, match="expected 2"):
        validate_header_envelope(valid_header)
