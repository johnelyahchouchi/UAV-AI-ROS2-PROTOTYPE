"""Strict validation and sanitization for network-supplied detection JSON."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .config import PROTOCOL_VERSION, SecurityLimits


SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_SEQUENCE = (1 << 63) - 1
MAX_TIMESTAMP = 253_402_300_799.0
MAX_JSON_DEPTH = 8

STRING_FIELDS = frozenset(
    {
        "uav_id",
        "source",
        "model",
        "class",
        "class_name",
        "final_class",
        "platform_type",
        "platform_category",
        "label",
        "name",
        "target_id",
        "source_id",
        "target_status",
        "status",
        "threat_level",
        "priority",
    }
)
INTEGER_FIELDS = frozenset(
    {
        "class_id",
        "cls",
        "track_id",
        "raw_id",
        "raw_track_id",
        "clean_track_id",
        "alert_priority",
    }
)
FLOAT_RANGES = {
    "confidence": (0.0, 1.0),
    "conf": (0.0, 1.0),
    "threat_score": (0.0, 100.0),
    "base_threat": (0.0, 100.0),
    "image_area_ratio": (0.0, 1.0),
    "norm_center_x": (0.0, 1.0),
    "norm_center_y": (0.0, 1.0),
    "timestamp": (0.0, MAX_TIMESTAMP),
}
BOOLEAN_FIELDS = frozenset({"is_target"})
GEOMETRY_FIELDS = frozenset(
    {
        "bbox",
        "bbox_xyxy",
        "bbox_xywh",
        "x1",
        "y1",
        "x2",
        "y2",
        "x",
        "y",
        "w",
        "h",
        "center_x",
        "center_y",
        "cx",
        "cy",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "source_width",
        "source_height",
    }
)
ALLOWED_DETECTION_KEYS = STRING_FIELDS | INTEGER_FIELDS | frozenset(FLOAT_RANGES) | BOOLEAN_FIELDS | GEOMETRY_FIELDS


class DetectionValidationError(ValueError):
    """Raised when a frame header or detection violates the protocol schema."""


def _reject_json_constant(value: str) -> None:
    raise DetectionValidationError(f"Non-finite JSON number is forbidden: {value}")


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise DetectionValidationError("JSON structure is nested too deeply")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DetectionValidationError("JSON object keys must be strings")
            _check_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1)


def parse_header_json(header_bytes: bytes) -> dict[str, Any]:
    """Decode one bounded UTF-8 JSON object and reject non-standard numbers."""

    try:
        decoded = header_bytes.decode("utf-8", errors="strict")
        value = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DetectionValidationError("Frame header is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise DetectionValidationError("Frame header must be a JSON object")
    _check_depth(value)
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectionValidationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise DetectionValidationError(f"{name} is outside the allowed range")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise DetectionValidationError(f"{name} is outside the allowed finite range")
    return result


def _string(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise DetectionValidationError(f"{name} must be a string")
    if not value or len(value) > limit or "\x00" in value:
        raise DetectionValidationError(f"{name} is empty, oversized, or contains NUL")
    return value


def validate_header_envelope(
    header: dict[str, Any],
    *,
    limits: SecurityLimits | None = None,
) -> dict[str, Any]:
    """Validate fields needed before reading the JPEG payload."""

    budget = limits or SecurityLimits.from_environment()
    required = {
        "protocol_version",
        "session_id",
        "seq",
        "timestamp",
        "source_width",
        "source_height",
        "jpeg_size",
        "detections",
    }
    missing = required.difference(header)
    if missing:
        raise DetectionValidationError(f"Frame header is missing fields: {sorted(missing)}")
    version = _integer(header["protocol_version"], "protocol_version", 0, 2**31 - 1)
    if version != PROTOCOL_VERSION:
        raise DetectionValidationError(
            f"Unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
        )
    session_id = _string(header["session_id"], "session_id", 64).lower()
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise DetectionValidationError("session_id must be 32 random bytes encoded as hex")
    seq = _integer(header["seq"], "seq", 0, MAX_SEQUENCE)
    timestamp = _number(header["timestamp"], "timestamp", 0.0, MAX_TIMESTAMP)
    width = _integer(header["source_width"], "source_width", 1, budget.max_image_width)
    height = _integer(header["source_height"], "source_height", 1, budget.max_image_height)
    if width * height > budget.max_image_pixels:
        raise DetectionValidationError("Declared image pixel count exceeds the configured limit")
    jpeg_size = _integer(header["jpeg_size"], "jpeg_size", 1, budget.max_jpeg_size)
    detections = header["detections"]
    if not isinstance(detections, list):
        raise DetectionValidationError("detections must be a list")
    if len(detections) > budget.max_detections:
        raise DetectionValidationError("detections exceeds the per-frame limit")
    return {
        "protocol_version": version,
        "session_id": session_id,
        "seq": seq,
        "timestamp": timestamp,
        "source_width": width,
        "source_height": height,
        "jpeg_size": jpeg_size,
        "detections": detections,
    }


def _bbox_for(detection: dict[str, Any]) -> list[Any]:
    if "bbox" in detection:
        bbox = detection["bbox"]
    elif "bbox_xyxy" in detection:
        bbox = detection["bbox_xyxy"]
    elif all(key in detection for key in ("x1", "y1", "x2", "y2")):
        bbox = [detection[key] for key in ("x1", "y1", "x2", "y2")]
    else:
        raise DetectionValidationError("detection must contain an xyxy bounding box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise DetectionValidationError("bbox must be a four-element list")
    return bbox


def sanitize_detection(
    detection: Any,
    image_width: int,
    image_height: int,
    *,
    limits: SecurityLimits | None = None,
) -> dict[str, Any]:
    """Discard unknown keys, validate types, and canonicalize geometry."""

    budget = limits or SecurityLimits.from_environment()
    if not isinstance(detection, dict):
        raise DetectionValidationError("each detection must be an object")
    if not any(name in detection for name in ("class", "class_name", "final_class")):
        raise DetectionValidationError("detection has no class name")
    if not any(name in detection for name in ("class_id", "cls")):
        raise DetectionValidationError("detection has no class ID")
    if "confidence" not in detection and "conf" not in detection:
        raise DetectionValidationError("detection has no confidence")

    output: dict[str, Any] = {}
    for key in ALLOWED_DETECTION_KEYS.intersection(detection):
        value = detection[key]
        if key in STRING_FIELDS:
            output[key] = _string(value, key, budget.max_string_length)
        elif key in INTEGER_FIELDS:
            if value is None and key in {"raw_id", "raw_track_id"}:
                output[key] = None
            else:
                minimum = 0 if key in {"class_id", "cls", "alert_priority"} else -1
                output[key] = _integer(value, key, minimum, MAX_SEQUENCE)
        elif key in FLOAT_RANGES:
            output[key] = _number(value, key, *FLOAT_RANGES[key])
        elif key in BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise DetectionValidationError(f"{key} must be a boolean")
            output[key] = value

    raw_bbox = _bbox_for(detection)
    coordinates = [
        _number(value, f"bbox[{index}]", -1_000_000_000.0, 1_000_000_000.0)
        for index, value in enumerate(raw_bbox)
    ]
    x1_raw, y1_raw, x2_raw, y2_raw = coordinates
    if x1_raw > x2_raw or y1_raw > y2_raw:
        raise DetectionValidationError("bbox coordinates are reversed")
    x1 = int(max(0, min(image_width - 1, x1_raw)))
    y1 = int(max(0, min(image_height - 1, y1_raw)))
    x2 = int(max(0, min(image_width - 1, x2_raw)))
    y2 = int(max(0, min(image_height - 1, y2_raw)))
    if x2 <= x1 or y2 <= y1:
        raise DetectionValidationError("bbox is empty after clamping to the decoded image")
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    output.update(
        {
            "bbox": [x1, y1, x2, y2],
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, box_width, box_height],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x": x1,
            "y": y1,
            "w": box_width,
            "h": box_height,
            "center_x": center_x,
            "center_y": center_y,
            "cx": center_x,
            "cy": center_y,
            "norm_center_x": round(center_x / image_width, 6),
            "norm_center_y": round(center_y / image_height, 6),
            "bbox_width": box_width,
            "bbox_height": box_height,
            "bbox_area": box_width * box_height,
            "source_width": image_width,
            "source_height": image_height,
        }
    )
    return output


def sanitize_frame_header(
    header: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    limits: SecurityLimits | None = None,
) -> dict[str, Any]:
    """Validate decoded dimensions and return a publishable allowlisted header."""

    budget = limits or SecurityLimits.from_environment()
    envelope = validate_header_envelope(header, limits=budget)
    if envelope["source_width"] != image_width or envelope["source_height"] != image_height:
        raise DetectionValidationError("Declared dimensions do not match the decoded JPEG")
    envelope["detections"] = [
        sanitize_detection(item, image_width, image_height, limits=budget)
        for item in envelope["detections"]
    ]
    return envelope
