from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_autonomy.domain import Capability, Priority, RegionType, TargetStatus, TaskType
from agentic_autonomy.replanning_domain import TaskLifecycleState
from agentic_autonomy.scenario_loader import (
    _array,
    _enum,
    _enum_list,
    _integer,
    _keys,
    _number,
    _object,
    _read_json,
    _string,
)

from .errors import AdapterEventError, AdapterLimitError
from .event_domain import (
    AdapterAvailability,
    AdapterEventType,
    EventTimestamp,
    LinkState,
    MissionStatus,
    NormalizedEvent,
    Position3D,
    SourceMetadata,
)

STREAM_KEYS = {"schema_version", "mission_id", "events"}
EVENT_KEYS = {
    "schema_version",
    "mission_id",
    "event_id",
    "sequence",
    "event_type",
    "observed_at",
    "source",
    "payload",
}
TIMESTAMP_KEYS = {"clock_id", "sec", "nanosec"}
SOURCE_KEYS = {
    "source_id",
    "source_node",
    "topic",
    "message_type",
    "source_uav_id",
    "source_session_id",
    "source_timestamp",
    "upstream_sequence",
}
POSITION_KEYS = {"x", "y", "z", "frame_id"}
REGION_KEYS = {"id", "region_type", "vertices", "priority", "required_capabilities"}
POINT_KEYS = {"x", "y"}

PAYLOAD_KEYS = {
    AdapterEventType.MISSION_CONFIGURED: {
        "scenario_id",
        "regions",
        "operating_region_id",
        "exclusion_region_ids",
    },
    AdapterEventType.REGION_UPDATED: {"region"},
    AdapterEventType.UAV_CONFIGURED: {
        "uav_id",
        "capabilities",
        "max_workload",
        "max_task_distance",
    },
    AdapterEventType.UAV_STATE_UPDATED: {
        "uav_id",
        "position",
        "availability",
        "battery_percent",
        "link_state",
        "link_quality",
        "external_workload",
        "mission_status",
        "current_target_id",
    },
    AdapterEventType.TARGET_OBSERVED: {
        "global_target_id",
        "source_uav_id",
        "local_track_id",
        "class_name",
        "confidence",
        "status",
        "position",
        "priority",
        "required_capabilities",
        "continuity_uav_id",
    },
    AdapterEventType.TARGET_STATE_UPDATED: {
        "target_id",
        "status",
        "position",
        "priority",
        "required_capabilities",
        "continuity_uav_id",
    },
    AdapterEventType.TASK_CREATED: {
        "request_id",
        "task_type",
        "priority",
        "required_capabilities",
        "target_id",
        "region_id",
        "lifecycle_state",
        "reason",
    },
    AdapterEventType.TASK_UPDATED: {"request_id", "priority", "required_capabilities"},
    AdapterEventType.TASK_LIFECYCLE_CHANGED: {"request_id", "state", "reason"},
    AdapterEventType.SNAPSHOT_TICK: {"reason"},
    AdapterEventType.SNAPSHOT_REQUESTED: {"reason"},
}


def load_event_stream(path: str | Path, policy: dict) -> tuple[NormalizedEvent, ...]:
    """Load a bounded canonical event stream from JSON."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise AdapterEventError(f"could not inspect adapter event stream: {exc}") from exc
    maximum = policy["limits"]["maximum_message_bytes"]
    if size > maximum:
        raise AdapterLimitError(
            f"adapter event stream is {size} bytes; maximum_message_bytes is {maximum}"
        )
    try:
        return parse_event_stream(_read_json(source, "adapter event stream"))
    except (AdapterEventError, AdapterLimitError):
        raise
    except Exception as exc:
        raise AdapterEventError(str(exc)) from exc


def parse_event_stream(value: object) -> tuple[NormalizedEvent, ...]:
    """Validate a complete ordered adapter event stream."""
    try:
        raw = _object(value, "adapter_event_stream")
        _keys(raw, STREAM_KEYS, STREAM_KEYS, "adapter_event_stream")
        version = _string(raw["schema_version"], "adapter_event_stream.schema_version")
        if version != "1.0":
            raise AdapterEventError(
                f"unsupported adapter_event_stream.schema_version {version!r}; expected '1.0'"
            )
        mission_id = _string(raw["mission_id"], "adapter_event_stream.mission_id")
        event_items = _array(raw["events"], "adapter_event_stream.events")
        if not event_items:
            raise AdapterEventError("adapter_event_stream.events must contain at least one event")
        events = []
        previous_sequence = None
        for index, item in enumerate(event_items):
            event = parse_event(item, f"adapter_event_stream.events[{index}]")
            if event.mission_id != mission_id:
                raise AdapterEventError(
                    f"event {event.event_id} mission_id {event.mission_id!r} does not match stream mission_id {mission_id!r}"
                )
            if previous_sequence is not None and event.sequence <= previous_sequence:
                raise AdapterEventError(
                    "adapter event sequence values must be strictly increasing in input order"
                )
            previous_sequence = event.sequence
            events.append(event)
        return tuple(events)
    except AdapterEventError:
        raise
    except Exception as exc:
        raise AdapterEventError(str(exc)) from exc


def parse_event(value: object, path: str = "adapter_event") -> NormalizedEvent:
    """Validate one canonical normalized event."""
    try:
        raw = _object(value, path)
        _keys(raw, EVENT_KEYS, EVENT_KEYS, path)
        version = _string(raw["schema_version"], f"{path}.schema_version")
        if version != "1.0":
            raise AdapterEventError(
                f"unsupported {path}.schema_version {version!r}; expected '1.0'"
            )
        mission_id = _string(raw["mission_id"], f"{path}.mission_id")
        event_id = _string(raw["event_id"], f"{path}.event_id")
        sequence = _integer(raw["sequence"], f"{path}.sequence")
        if sequence < 0:
            raise AdapterEventError(f"{path}.sequence must be nonnegative")
        event_type = _enum(AdapterEventType, raw["event_type"], f"{path}.event_type")
        observed_at = _parse_timestamp(raw["observed_at"], f"{path}.observed_at")
        source = _parse_source(raw["source"], f"{path}.source")
        payload = _parse_payload(event_type, raw["payload"], f"{path}.payload")
        return NormalizedEvent(
            version,
            mission_id,
            event_id,
            sequence,
            event_type,
            observed_at,
            source,
            payload,
        )
    except AdapterEventError:
        raise
    except Exception as exc:
        raise AdapterEventError(str(exc)) from exc


def _parse_timestamp(value: object, path: str) -> EventTimestamp:
    data = _object(value, path)
    _keys(data, TIMESTAMP_KEYS, TIMESTAMP_KEYS, path)
    clock_id = _string(data["clock_id"], f"{path}.clock_id")
    sec = _integer(data["sec"], f"{path}.sec")
    nanosec = _integer(data["nanosec"], f"{path}.nanosec")
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise AdapterEventError(
            f"{path} requires nonnegative sec and nanosec in [0, 1000000000)"
        )
    return EventTimestamp(clock_id, sec, nanosec)


def _parse_source(value: object, path: str) -> SourceMetadata:
    data = _object(value, path)
    _keys(data, SOURCE_KEYS, {"source_id"}, path)
    upstream = data.get("upstream_sequence")
    if upstream is not None:
        upstream = _integer(upstream, f"{path}.upstream_sequence")
        if upstream < 0:
            raise AdapterEventError(f"{path}.upstream_sequence must be nonnegative")
    return SourceMetadata(
        source_id=_string(data["source_id"], f"{path}.source_id"),
        source_node=_string(data.get("source_node"), f"{path}.source_node", nullable=True),
        topic=_string(data.get("topic"), f"{path}.topic", nullable=True),
        message_type=_string(data.get("message_type"), f"{path}.message_type", nullable=True),
        source_uav_id=_string(
            data.get("source_uav_id"), f"{path}.source_uav_id", nullable=True
        ),
        source_session_id=_string(
            data.get("source_session_id"), f"{path}.source_session_id", nullable=True
        ),
        source_timestamp=_string(
            data.get("source_timestamp"), f"{path}.source_timestamp", nullable=True
        ),
        upstream_sequence=upstream,
    )


def _parse_position(value: object, path: str) -> dict:
    data = _object(value, path)
    _keys(data, POSITION_KEYS, POSITION_KEYS - {"z"}, path)
    position = Position3D(
        x=_number(data["x"], f"{path}.x"),
        y=_number(data["y"], f"{path}.y"),
        z=_number(data["z"], f"{path}.z") if data.get("z") is not None else None,
        frame_id=_string(data["frame_id"], f"{path}.frame_id"),
    )
    return {
        "x": position.x,
        "y": position.y,
        "z": position.z,
        "frame_id": position.frame_id,
    }


def _parse_region(value: object, path: str) -> dict:
    data = _object(value, path)
    _keys(data, REGION_KEYS, REGION_KEYS - {"required_capabilities"}, path)
    vertices = []
    for index, item in enumerate(_array(data["vertices"], f"{path}.vertices")):
        point_path = f"{path}.vertices[{index}]"
        point = _object(item, point_path)
        _keys(point, POINT_KEYS, POINT_KEYS, point_path)
        vertices.append(
            {
                "x": _number(point["x"], f"{point_path}.x"),
                "y": _number(point["y"], f"{point_path}.y"),
            }
        )
    if len(vertices) < 3:
        raise AdapterEventError(f"{path}.vertices must contain at least three points")
    return {
        "id": _string(data["id"], f"{path}.id"),
        "region_type": _enum(RegionType, data["region_type"], f"{path}.region_type").value,
        "vertices": vertices,
        "priority": _enum(Priority, data["priority"], f"{path}.priority").value,
        "required_capabilities": sorted(
            item.value
            for item in _enum_list(
                Capability,
                data.get("required_capabilities", []),
                f"{path}.required_capabilities",
            )
        ),
    }


def _parse_payload(event_type: AdapterEventType, value: object, path: str) -> dict[str, Any]:
    data = _object(value, path)
    allowed = PAYLOAD_KEYS[event_type]
    required = _required_payload_keys(event_type)
    _keys(data, allowed, required, path)

    if event_type == AdapterEventType.MISSION_CONFIGURED:
        regions = [
            _parse_region(item, f"{path}.regions[{index}]")
            for index, item in enumerate(_array(data["regions"], f"{path}.regions"))
        ]
        ids = [item["id"] for item in regions]
        if len(ids) != len(set(ids)):
            raise AdapterEventError(f"{path}.regions contains duplicate identifiers")
        return {
            "scenario_id": _string(data["scenario_id"], f"{path}.scenario_id"),
            "regions": regions,
            "operating_region_id": _string(
                data.get("operating_region_id"),
                f"{path}.operating_region_id",
                nullable=True,
            ),
            "exclusion_region_ids": [
                _string(item, f"{path}.exclusion_region_ids[{index}]")
                for index, item in enumerate(
                    _array(data["exclusion_region_ids"], f"{path}.exclusion_region_ids")
                )
            ],
        }
    if event_type == AdapterEventType.REGION_UPDATED:
        return {"region": _parse_region(data["region"], f"{path}.region")}
    if event_type == AdapterEventType.UAV_CONFIGURED:
        maximum_distance = data.get("max_task_distance")
        if maximum_distance is not None:
            maximum_distance = _number(maximum_distance, f"{path}.max_task_distance")
            if maximum_distance <= 0:
                raise AdapterEventError(f"{path}.max_task_distance must be greater than zero")
        maximum_workload = _integer(data["max_workload"], f"{path}.max_workload")
        if maximum_workload < 1:
            raise AdapterEventError(f"{path}.max_workload must be positive")
        return {
            "uav_id": _string(data["uav_id"], f"{path}.uav_id"),
            "capabilities": sorted(
                item.value
                for item in _enum_list(
                    Capability, data["capabilities"], f"{path}.capabilities"
                )
            ),
            "max_workload": maximum_workload,
            "max_task_distance": maximum_distance,
        }
    if event_type == AdapterEventType.UAV_STATE_UPDATED:
        return _parse_uav_update(data, path)
    if event_type == AdapterEventType.TARGET_OBSERVED:
        return _parse_target_observation(data, path)
    if event_type == AdapterEventType.TARGET_STATE_UPDATED:
        return _parse_target_update(data, path)
    if event_type == AdapterEventType.TASK_CREATED:
        return _parse_task_created(data, path)
    if event_type == AdapterEventType.TASK_UPDATED:
        result: dict[str, Any] = {
            "request_id": _string(data["request_id"], f"{path}.request_id")
        }
        if "priority" in data:
            result["priority"] = _enum(Priority, data["priority"], f"{path}.priority").value
        if "required_capabilities" in data:
            result["required_capabilities"] = sorted(
                item.value
                for item in _enum_list(
                    Capability,
                    data["required_capabilities"],
                    f"{path}.required_capabilities",
                )
            )
        if len(result) == 1:
            raise AdapterEventError(f"{path} must update priority or required_capabilities")
        return result
    if event_type == AdapterEventType.TASK_LIFECYCLE_CHANGED:
        return {
            "request_id": _string(data["request_id"], f"{path}.request_id"),
            "state": _enum(TaskLifecycleState, data["state"], f"{path}.state").value,
            "reason": _string(data.get("reason"), f"{path}.reason", nullable=True),
        }
    return {"reason": _string(data.get("reason"), f"{path}.reason", nullable=True)}


def _required_payload_keys(event_type: AdapterEventType) -> set[str]:
    return {
        AdapterEventType.MISSION_CONFIGURED: {
            "scenario_id",
            "regions",
            "operating_region_id",
            "exclusion_region_ids",
        },
        AdapterEventType.REGION_UPDATED: {"region"},
        AdapterEventType.UAV_CONFIGURED: {
            "uav_id",
            "capabilities",
            "max_workload",
            "max_task_distance",
        },
        AdapterEventType.UAV_STATE_UPDATED: {"uav_id"},
        AdapterEventType.TARGET_OBSERVED: {
            "global_target_id",
            "source_uav_id",
            "local_track_id",
            "class_name",
            "confidence",
            "status",
            "position",
            "priority",
            "required_capabilities",
            "continuity_uav_id",
        },
        AdapterEventType.TARGET_STATE_UPDATED: {"target_id", "status"},
        AdapterEventType.TASK_CREATED: {
            "request_id",
            "task_type",
            "priority",
            "required_capabilities",
            "target_id",
            "region_id",
            "lifecycle_state",
            "reason",
        },
        AdapterEventType.TASK_UPDATED: {"request_id"},
        AdapterEventType.TASK_LIFECYCLE_CHANGED: {"request_id", "state"},
        AdapterEventType.SNAPSHOT_TICK: set(),
        AdapterEventType.SNAPSHOT_REQUESTED: set(),
    }[event_type]


def _parse_uav_update(data: dict, path: str) -> dict:
    result: dict[str, Any] = {"uav_id": _string(data["uav_id"], f"{path}.uav_id")}
    if len(data) == 1:
        raise AdapterEventError(f"{path} must contain at least one dynamic UAV field")
    if "position" in data:
        result["position"] = _parse_position(data["position"], f"{path}.position")
    if "availability" in data:
        result["availability"] = _enum(
            AdapterAvailability, data["availability"], f"{path}.availability"
        ).value
    if "battery_percent" in data:
        result["battery_percent"] = _number(data["battery_percent"], f"{path}.battery_percent")
    if "link_state" in data:
        result["link_state"] = _enum(LinkState, data["link_state"], f"{path}.link_state").value
    if "link_quality" in data:
        quality = _number(data["link_quality"], f"{path}.link_quality")
        if not 0 <= quality <= 1:
            raise AdapterEventError(f"{path}.link_quality must be in [0, 1]")
        result["link_quality"] = quality
    if "external_workload" in data:
        workload = _integer(data["external_workload"], f"{path}.external_workload")
        if workload < 0:
            raise AdapterEventError(f"{path}.external_workload must be nonnegative")
        result["external_workload"] = workload
    if "mission_status" in data:
        result["mission_status"] = _enum(
            MissionStatus, data["mission_status"], f"{path}.mission_status"
        ).value
    if "current_target_id" in data:
        result["current_target_id"] = _string(
            data["current_target_id"], f"{path}.current_target_id", nullable=True
        )
    if result.get("link_state") == LinkState.LOST.value and result.get(
        "link_quality"
    ) != 0:
        raise AdapterEventError(
            f"{path} LOST link_state requires link_quality 0.0 in the same event"
        )
    return result


def _parse_target_observation(data: dict, path: str) -> dict:
    confidence = _number(data["confidence"], f"{path}.confidence")
    if not 0 <= confidence <= 1:
        raise AdapterEventError(f"{path}.confidence must be in [0, 1]")
    return {
        "global_target_id": _string(
            data["global_target_id"], f"{path}.global_target_id", nullable=True
        ),
        "source_uav_id": _string(data["source_uav_id"], f"{path}.source_uav_id"),
        "local_track_id": _string(
            data["local_track_id"], f"{path}.local_track_id", nullable=True
        ),
        "class_name": _string(data["class_name"], f"{path}.class_name"),
        "confidence": confidence,
        "status": _enum(TargetStatus, data["status"], f"{path}.status").value,
        "position": (
            _parse_position(data["position"], f"{path}.position")
            if data["position"] is not None
            else None
        ),
        "priority": (
            _enum(Priority, data["priority"], f"{path}.priority").value
            if data["priority"] is not None
            else None
        ),
        "required_capabilities": sorted(
            item.value
            for item in _enum_list(
                Capability,
                data["required_capabilities"],
                f"{path}.required_capabilities",
            )
        ),
        "continuity_uav_id": _string(
            data["continuity_uav_id"], f"{path}.continuity_uav_id", nullable=True
        ),
    }


def _parse_target_update(data: dict, path: str) -> dict:
    result: dict[str, Any] = {
        "target_id": _string(data["target_id"], f"{path}.target_id"),
        "status": _enum(TargetStatus, data["status"], f"{path}.status").value,
    }
    if "position" in data:
        result["position"] = _parse_position(data["position"], f"{path}.position")
    if "priority" in data:
        result["priority"] = _enum(Priority, data["priority"], f"{path}.priority").value
    if "required_capabilities" in data:
        result["required_capabilities"] = sorted(
            item.value
            for item in _enum_list(
                Capability,
                data["required_capabilities"],
                f"{path}.required_capabilities",
            )
        )
    if "continuity_uav_id" in data:
        result["continuity_uav_id"] = _string(
            data["continuity_uav_id"], f"{path}.continuity_uav_id", nullable=True
        )
    return result


def _parse_task_created(data: dict, path: str) -> dict:
    task_type = _enum(TaskType, data["task_type"], f"{path}.task_type")
    target_id = _string(data["target_id"], f"{path}.target_id", nullable=True)
    region_id = _string(data["region_id"], f"{path}.region_id", nullable=True)
    target_tasks = {TaskType.INVESTIGATE_TARGET, TaskType.TRACK_TARGET}
    region_tasks = {
        TaskType.SEARCH_REGION,
        TaskType.OBSERVE_REGION,
        TaskType.RELAY_COMMUNICATIONS,
    }
    if task_type in target_tasks and (target_id is None or region_id is not None):
        raise AdapterEventError(
            f"{path} target task requires target_id and forbids region_id"
        )
    if task_type in region_tasks and (region_id is None or target_id is not None):
        raise AdapterEventError(
            f"{path} region task requires region_id and forbids target_id"
        )
    lifecycle = _enum(
        TaskLifecycleState, data["lifecycle_state"], f"{path}.lifecycle_state"
    )
    if lifecycle not in {TaskLifecycleState.PENDING, TaskLifecycleState.ACTIVE}:
        raise AdapterEventError(
            f"{path}.lifecycle_state must begin as PENDING or ACTIVE"
        )
    return {
        "request_id": _string(data["request_id"], f"{path}.request_id"),
        "task_type": task_type.value,
        "priority": _enum(Priority, data["priority"], f"{path}.priority").value,
        "required_capabilities": sorted(
            item.value
            for item in _enum_list(
                Capability,
                data["required_capabilities"],
                f"{path}.required_capabilities",
            )
        ),
        "target_id": target_id,
        "region_id": region_id,
        "lifecycle_state": lifecycle.value,
        "reason": _string(data["reason"], f"{path}.reason", nullable=True),
    }
