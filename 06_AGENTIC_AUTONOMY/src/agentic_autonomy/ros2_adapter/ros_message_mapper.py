from __future__ import annotations

import json
import math
from typing import Callable

from agentic_autonomy.scenario_loader import _reject_constant, _reject_duplicate_keys

from .errors import AdapterEventError
from .event_domain import EventTimestamp, NormalizedEvent
from .normalized_events import parse_event


def map_canonical_json_message(message, ingestion_sequence: int) -> NormalizedEvent:
    """Map a std_msgs/String-like object while preserving upstream order separately."""
    try:
        raw = json.loads(
            message.data,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Exception as exc:
        raise AdapterEventError(f"canonical ROS JSON message is malformed: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdapterEventError("canonical ROS JSON message must contain one event object")
    raw = dict(raw)
    source = dict(raw.get("source", {}))
    upstream_sequence = raw.get("sequence")
    if upstream_sequence is not None:
        if source.get("upstream_sequence") not in {None, upstream_sequence}:
            raise AdapterEventError(
                "canonical ROS message has conflicting upstream sequence values"
            )
        source["upstream_sequence"] = upstream_sequence
    raw["source"] = source
    raw["sequence"] = ingestion_sequence
    return parse_event(raw, "canonical_ros_event")


def map_battery_state(
    message,
    *,
    mission_id: str,
    ingestion_sequence: int,
    uav_id: str,
    source_id: str,
    receipt_time: EventTimestamp,
    topic: str | None = None,
) -> NormalizedEvent:
    """Map sensor_msgs/BatteryState.percentage from its defined 0.0-1.0 semantics."""
    percentage = getattr(message, "percentage", None)
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, (int, float))
        or not math.isfinite(float(percentage))
        or not 0 <= float(percentage) <= 1
    ):
        raise AdapterEventError(
            "sensor_msgs/BatteryState.percentage must be finite and in [0.0, 1.0]"
        )
    observed_at = _header_timestamp(message, receipt_time)
    return parse_event(
        _event_dict(
            mission_id=mission_id,
            event_id=f"{source_id}-battery-{ingestion_sequence:09d}",
            sequence=ingestion_sequence,
            event_type="UAV_STATE_UPDATED",
            observed_at=observed_at,
            source_id=source_id,
            topic=topic,
            message_type="sensor_msgs/msg/BatteryState",
            source_uav_id=uav_id,
            source_timestamp=_source_stamp_label(message),
            payload={"uav_id": uav_id, "battery_percent": float(percentage) * 100.0},
        )
    )


def map_pose_stamped(
    message,
    *,
    mission_id: str,
    ingestion_sequence: int,
    uav_id: str,
    source_id: str,
    receipt_time: EventTimestamp,
    topic: str | None = None,
) -> NormalizedEvent:
    """Map a geometry_msgs/PoseStamped-like object without performing TF conversion."""
    try:
        position = message.pose.position
        x, y, z = float(position.x), float(position.y), float(position.z)
        frame_id = str(message.header.frame_id).strip()
    except Exception as exc:
        raise AdapterEventError(f"PoseStamped message is incomplete: {exc}") from exc
    if not frame_id or not all(math.isfinite(item) for item in (x, y, z)):
        raise AdapterEventError(
            "PoseStamped requires a non-empty frame_id and finite x/y/z"
        )
    observed_at = _header_timestamp(message, receipt_time)
    return parse_event(
        _event_dict(
            mission_id=mission_id,
            event_id=f"{source_id}-pose-{ingestion_sequence:09d}",
            sequence=ingestion_sequence,
            event_type="UAV_STATE_UPDATED",
            observed_at=observed_at,
            source_id=source_id,
            topic=topic,
            message_type="geometry_msgs/msg/PoseStamped",
            source_uav_id=uav_id,
            source_timestamp=_source_stamp_label(message),
            payload={
                "uav_id": uav_id,
                "position": {"x": x, "y": y, "z": z, "frame_id": frame_id},
            },
        )
    )


def map_legacy_detection_message(
    message,
    *,
    mission_id: str,
    next_sequence: Callable[[], int],
    source_uav_id: str,
    source_session_id: str | None,
    receipt_time: EventTimestamp,
    topic: str | None = None,
) -> tuple[NormalizedEvent, ...]:
    """Map legacy detection JSON without generating tasks or world coordinates."""
    try:
        raw = json.loads(
            message.data,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Exception as exc:
        raise AdapterEventError(f"legacy detection JSON is malformed: {exc}") from exc
    if isinstance(raw, dict):
        detections = raw.get("detections", [raw])
        upstream_sequence = raw.get("seq")
    elif isinstance(raw, list):
        detections = raw
        upstream_sequence = None
    else:
        raise AdapterEventError("legacy detection JSON must be an object or array")
    if not isinstance(detections, list):
        raise AdapterEventError("legacy detections property must be an array")
    events = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise AdapterEventError(f"legacy detection [{index}] must be an object")
        sequence = next_sequence()
        confidence = detection.get("confidence", detection.get("conf"))
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise AdapterEventError(
                f"legacy detection [{index}] confidence must be finite and in [0, 1]"
            )
        class_name = _first_nonempty(
            detection, ("class_name", "class", "final_class", "label")
        )
        if class_name is None:
            raise AdapterEventError(f"legacy detection [{index}] has no class name")
        global_target_id = _optional_string(detection.get("global_target_id"))
        local_track_id = _optional_track_id(
            detection.get(
                "clean_track_id",
                detection.get("track_id", detection.get("raw_track_id")),
            )
        )
        status = "TRACKED" if local_track_id is not None else "DETECTED"
        raw_priority = detection.get("priority")
        priority = None
        if isinstance(raw_priority, str) and raw_priority.upper() in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }:
            priority = raw_priority.upper()
        raw_timestamp = detection.get("timestamp")
        source_timestamp = (
            str(raw_timestamp)
            if isinstance(raw_timestamp, (int, float))
            and not isinstance(raw_timestamp, bool)
            and math.isfinite(float(raw_timestamp))
            else None
        )
        event_raw = _event_dict(
            mission_id=mission_id,
            event_id=f"legacy-{source_uav_id}-{sequence:09d}",
            sequence=sequence,
            event_type="TARGET_OBSERVED",
            observed_at=receipt_time,
            source_id="legacy_yolo_detection",
            topic=topic,
            message_type="std_msgs/msg/String",
            source_uav_id=source_uav_id,
            source_session_id=source_session_id,
            source_timestamp=source_timestamp,
            upstream_sequence=upstream_sequence,
            payload={
                "global_target_id": global_target_id,
                "source_uav_id": source_uav_id,
                "local_track_id": local_track_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "status": status,
                "position": None,
                "priority": priority,
                "required_capabilities": [],
                "continuity_uav_id": None,
            },
        )
        events.append(parse_event(event_raw))
    return tuple(events)


def _event_dict(
    *,
    mission_id,
    event_id,
    sequence,
    event_type,
    observed_at,
    source_id,
    payload,
    topic=None,
    message_type=None,
    source_uav_id=None,
    source_session_id=None,
    source_timestamp=None,
    upstream_sequence=None,
):
    return {
        "schema_version": "1.0",
        "mission_id": mission_id,
        "event_id": event_id,
        "sequence": sequence,
        "event_type": event_type,
        "observed_at": {
            "clock_id": observed_at.clock_id,
            "sec": observed_at.sec,
            "nanosec": observed_at.nanosec,
        },
        "source": {
            "source_id": source_id,
            "source_node": None,
            "topic": topic,
            "message_type": message_type,
            "source_uav_id": source_uav_id,
            "source_session_id": source_session_id,
            "source_timestamp": source_timestamp,
            "upstream_sequence": upstream_sequence,
        },
        "payload": payload,
    }


def _header_timestamp(message, fallback: EventTimestamp) -> EventTimestamp:
    try:
        stamp = message.header.stamp
        sec, nanosec = int(stamp.sec), int(stamp.nanosec)
        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            raise ValueError("invalid stamp")
        return EventTimestamp(fallback.clock_id, sec, nanosec)
    except Exception:
        return fallback


def _source_stamp_label(message) -> str | None:
    try:
        stamp = message.header.stamp
        return f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"
    except Exception:
        return None


def _first_nonempty(data: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _optional_string(value) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value)


def _optional_track_id(value) -> str | None:
    if value is None or str(value) in {"", "None", "unknown", "-1"}:
        return None
    return str(value)
