from __future__ import annotations

from pathlib import Path

from agentic_autonomy.scenario_loader import (
    _array,
    _boolean,
    _integer,
    _keys,
    _number,
    _object,
    _read_json,
    _string,
)

from .errors import AdapterConfigurationError

POLICY_KEYS = {
    "adapter_policy_version",
    "planning_frame",
    "freshness_thresholds_seconds",
    "battery",
    "snapshot_trigger",
    "limits",
    "ros",
}
FRESHNESS_KEYS = {"position", "availability", "battery", "link", "workload", "target"}
BATTERY_KEYS = {"clamp_tolerance_percent"}
SNAPSHOT_KEYS = {"periodic_interval_seconds", "unchanged_heartbeat_seconds", "emit_on_safety_change"}
LIMIT_KEYS = {
    "maximum_message_bytes",
    "maximum_snapshots",
    "maximum_uavs",
    "maximum_targets",
    "maximum_tasks",
    "maximum_diagnostics",
}
ROS_KEYS = {
    "event_topic",
    "legacy_detection_topics",
    "battery_topics",
    "pose_topics",
    "snapshot_topic",
    "diagnostics_topic",
    "advisory_topic",
    "snapshot_service",
    "invoke_replanner",
    "output_directory",
}
LEGACY_TOPIC_KEYS = {"topic", "uav_id", "source_session_id"}
STATE_TOPIC_KEYS = {"topic", "uav_id", "source_id"}


def _raise_configuration(exc: Exception) -> None:
    raise AdapterConfigurationError(str(exc)) from exc


def load_adapter_policy(path: str | Path) -> dict:
    """Load and strictly validate the adapter policy."""
    try:
        return parse_adapter_policy(_read_json(path, "ROS 2 adapter policy"))
    except AdapterConfigurationError:
        raise
    except Exception as exc:
        _raise_configuration(exc)


def parse_adapter_policy(value: object) -> dict:
    """Validate an in-memory adapter policy without ROS dependencies."""
    try:
        raw = _object(value, "adapter_policy")
        _keys(raw, POLICY_KEYS, POLICY_KEYS, "adapter_policy")
        version = _string(raw["adapter_policy_version"], "adapter_policy.adapter_policy_version")
        if version != "1.0":
            raise AdapterConfigurationError(
                f"unsupported adapter_policy.adapter_policy_version {version!r}; expected '1.0'"
            )
        frame = _string(raw["planning_frame"], "adapter_policy.planning_frame")

        freshness_raw = _object(raw["freshness_thresholds_seconds"], "adapter_policy.freshness_thresholds_seconds")
        _keys(freshness_raw, FRESHNESS_KEYS, FRESHNESS_KEYS, "adapter_policy.freshness_thresholds_seconds")
        freshness = {
            key: _number(freshness_raw[key], f"adapter_policy.freshness_thresholds_seconds.{key}")
            for key in sorted(FRESHNESS_KEYS)
        }
        if any(item <= 0 for item in freshness.values()):
            raise AdapterConfigurationError("all freshness thresholds must be greater than zero")

        battery_raw = _object(raw["battery"], "adapter_policy.battery")
        _keys(battery_raw, BATTERY_KEYS, BATTERY_KEYS, "adapter_policy.battery")
        tolerance = _number(
            battery_raw["clamp_tolerance_percent"],
            "adapter_policy.battery.clamp_tolerance_percent",
        )
        if tolerance < 0 or tolerance > 1:
            raise AdapterConfigurationError("battery clamp tolerance must be in [0, 1]")

        trigger_raw = _object(raw["snapshot_trigger"], "adapter_policy.snapshot_trigger")
        _keys(trigger_raw, SNAPSHOT_KEYS, SNAPSHOT_KEYS, "adapter_policy.snapshot_trigger")
        periodic = _number(
            trigger_raw["periodic_interval_seconds"],
            "adapter_policy.snapshot_trigger.periodic_interval_seconds",
        )
        heartbeat = _number(
            trigger_raw["unchanged_heartbeat_seconds"],
            "adapter_policy.snapshot_trigger.unchanged_heartbeat_seconds",
        )
        immediate = _boolean(
            trigger_raw["emit_on_safety_change"],
            "adapter_policy.snapshot_trigger.emit_on_safety_change",
        )
        if periodic <= 0 or heartbeat < periodic:
            raise AdapterConfigurationError(
                "snapshot periodic interval must be positive and heartbeat must be at least the periodic interval"
            )

        limits_raw = _object(raw["limits"], "adapter_policy.limits")
        _keys(limits_raw, LIMIT_KEYS, LIMIT_KEYS, "adapter_policy.limits")
        limits = {
            key: _integer(limits_raw[key], f"adapter_policy.limits.{key}")
            for key in sorted(LIMIT_KEYS)
        }
        if any(item < 1 for item in limits.values()):
            raise AdapterConfigurationError("all adapter limits must be positive integers")

        ros_raw = _object(raw["ros"], "adapter_policy.ros")
        _keys(ros_raw, ROS_KEYS, ROS_KEYS, "adapter_policy.ros")
        legacy_topics = _topic_list(
            ros_raw["legacy_detection_topics"],
            "adapter_policy.ros.legacy_detection_topics",
            legacy=True,
        )
        battery_topics = _topic_list(
            ros_raw["battery_topics"],
            "adapter_policy.ros.battery_topics",
            legacy=False,
        )
        pose_topics = _topic_list(
            ros_raw["pose_topics"],
            "adapter_policy.ros.pose_topics",
            legacy=False,
        )
        ros = {
            "event_topic": _string(ros_raw["event_topic"], "adapter_policy.ros.event_topic"),
            "legacy_detection_topics": legacy_topics,
            "battery_topics": battery_topics,
            "pose_topics": pose_topics,
            "snapshot_topic": _string(ros_raw["snapshot_topic"], "adapter_policy.ros.snapshot_topic"),
            "diagnostics_topic": _string(
                ros_raw["diagnostics_topic"], "adapter_policy.ros.diagnostics_topic"
            ),
            "advisory_topic": _string(ros_raw["advisory_topic"], "adapter_policy.ros.advisory_topic"),
            "snapshot_service": _string(
                ros_raw["snapshot_service"], "adapter_policy.ros.snapshot_service"
            ),
            "invoke_replanner": _boolean(
                ros_raw["invoke_replanner"], "adapter_policy.ros.invoke_replanner"
            ),
            "output_directory": _string(
                ros_raw["output_directory"], "adapter_policy.ros.output_directory"
            ),
        }
        return {
            "adapter_policy_version": version,
            "planning_frame": frame,
            "freshness_thresholds_seconds": freshness,
            "battery": {"clamp_tolerance_percent": tolerance},
            "snapshot_trigger": {
                "periodic_interval_seconds": periodic,
                "unchanged_heartbeat_seconds": heartbeat,
                "emit_on_safety_change": immediate,
            },
            "limits": limits,
            "ros": ros,
        }
    except AdapterConfigurationError:
        raise
    except Exception as exc:
        _raise_configuration(exc)


def _topic_list(value: object, path: str, *, legacy: bool) -> list[dict]:
    result = []
    expected = LEGACY_TOPIC_KEYS if legacy else STATE_TOPIC_KEYS
    for index, item in enumerate(_array(value, path)):
        item_path = f"{path}[{index}]"
        data = _object(item, item_path)
        _keys(data, expected, expected, item_path)
        parsed = {
            "topic": _string(data["topic"], f"{item_path}.topic"),
            "uav_id": _string(data["uav_id"], f"{item_path}.uav_id"),
        }
        if legacy:
            parsed["source_session_id"] = _string(
                data.get("source_session_id"),
                f"{item_path}.source_session_id",
                nullable=True,
            )
        else:
            parsed["source_id"] = _string(data["source_id"], f"{item_path}.source_id")
        result.append(parsed)
    topics = [item["topic"] for item in result]
    if len(topics) != len(set(topics)):
        raise AdapterConfigurationError(f"{path} must not contain duplicate topics")
    return result
