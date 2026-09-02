import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_autonomy.ros2_adapter.errors import AdapterEventError
from agentic_autonomy.ros2_adapter.event_domain import EventTimestamp
from agentic_autonomy.ros2_adapter.ros_message_mapper import (
    map_battery_state,
    map_canonical_json_message,
    map_legacy_detection_message,
    map_pose_stamped,
)

from adapter_test_helpers import full_uav_state


def _header(sec=5, nanosec=10, frame_id="mission_local"):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id
    )


def test_battery_state_mapper_converts_defined_fraction_to_percent():
    message = SimpleNamespace(percentage=0.625, header=_header())
    mapped = map_battery_state(
        message,
        mission_id="test-mission",
        ingestion_sequence=10,
        uav_id="uav-1",
        source_id="battery-source",
        receipt_time=EventTimestamp("mission", 6, 0),
    )
    assert mapped.payload["battery_percent"] == 62.5
    assert mapped.source.source_timestamp == "5.000000010"


@pytest.mark.parametrize("value", [None, True, -0.01, 1.01, float("nan")])
def test_battery_state_mapper_rejects_missing_or_invalid_percentage(value):
    message = SimpleNamespace(percentage=value, header=_header())
    with pytest.raises(AdapterEventError, match=r"\[0.0, 1.0\]"):
        map_battery_state(
            message,
            mission_id="test-mission",
            ingestion_sequence=10,
            uav_id="uav-1",
            source_id="battery-source",
            receipt_time=EventTimestamp("mission", 6, 0),
        )


def test_pose_mapper_preserves_exact_frame_without_conversion():
    message = SimpleNamespace(
        header=_header(frame_id="mission_local"),
        pose=SimpleNamespace(position=SimpleNamespace(x=1, y=2, z=3)),
    )
    mapped = map_pose_stamped(
        message,
        mission_id="test-mission",
        ingestion_sequence=11,
        uav_id="uav-1",
        source_id="pose-source",
        receipt_time=EventTimestamp("mission", 6, 0),
    )
    assert mapped.payload["position"] == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "frame_id": "mission_local",
    }


def test_canonical_mapper_preserves_upstream_sequence_separately():
    raw = full_uav_state(sequence=42)
    raw["source"]["source_timestamp"] = "producer:5.0"
    message = SimpleNamespace(data=json.dumps(raw))
    mapped = map_canonical_json_message(message, ingestion_sequence=7)
    assert mapped.sequence == 7
    assert mapped.source.upstream_sequence == 42
    assert mapped.source.source_timestamp == "producer:5.0"


def test_legacy_detection_mapper_emits_target_only_and_no_world_position():
    message = SimpleNamespace(
        data=json.dumps(
            {
                "detections": [
                    {"track_id": 7, "class_name": "vehicle", "confidence": 0.8}
                ]
            }
        )
    )
    next_value = iter([20]).__next__
    events = map_legacy_detection_message(
        message,
        mission_id="test-mission",
        next_sequence=next_value,
        source_uav_id="uav-1",
        source_session_id="session-a",
        receipt_time=EventTimestamp("mission", 2, 0),
        topic="/uav_1/coco_detections",
    )
    assert len(events) == 1
    assert events[0].event_type.value == "TARGET_OBSERVED"
    assert events[0].payload["position"] is None


def test_ros_modules_are_lazy_and_no_resources_exist_at_import():
    module = importlib.import_module("agentic_autonomy.ros2_adapter.ros_node")
    assert callable(module.main)
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_names = {
        alias.name.split(".")[0]
        for node in top_level_imports
        for alias in node.names
    }
    assert "rclpy" not in imported_names
    assert "geometry_msgs" not in imported_names
    assert "sensor_msgs" not in imported_names
    assert "std_msgs" not in imported_names


def test_ros_shell_has_no_flight_command_surface():
    module = importlib.import_module("agentic_autonomy.ros2_adapter.ros_node")
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "cmd_vel",
        "trajectory_msgs",
        "mavros_msgs",
        "px4_msgs",
        "actuatorcontrol",
    ):
        assert forbidden not in source
