from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from agentic_autonomy.ros2_adapter.adapter_configuration import load_adapter_policy
from agentic_autonomy.scenario_loader import load_policy

ROOT = Path(__file__).parents[1]


def adapter_policy() -> dict:
    return load_adapter_policy(ROOT / "config" / "ros2_adapter_policy.json")


def planner_policy() -> dict:
    return load_policy(ROOT / "config" / "default_policy.json")


def event(
    sequence: int,
    event_type: str,
    payload: dict,
    *,
    sec: int = 0,
    mission_id: str = "test-mission",
    event_id: str | None = None,
    source: dict | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "mission_id": mission_id,
        "event_id": event_id or f"event-{sequence:03d}",
        "sequence": sequence,
        "event_type": event_type,
        "observed_at": {"clock_id": "mission", "sec": sec, "nanosec": 0},
        "source": deepcopy(source or {"source_id": "pytest"}),
        "payload": deepcopy(payload),
    }


def mission_event(sequence: int = 1) -> dict:
    return event(
        sequence,
        "MISSION_CONFIGURED",
        {
            "scenario_id": "test-scenario",
            "regions": [],
            "operating_region_id": None,
            "exclusion_region_ids": [],
        },
    )


def uav_config_event(sequence: int = 2, uav_id: str = "uav-1") -> dict:
    return event(
        sequence,
        "UAV_CONFIGURED",
        {
            "uav_id": uav_id,
            "capabilities": ["TARGET_TRACKING"],
            "max_workload": 2,
            "max_task_distance": 1500.0,
        },
    )


def full_uav_state(
    sequence: int = 3,
    *,
    sec: int = 0,
    uav_id: str = "uav-1",
) -> dict:
    return event(
        sequence,
        "UAV_STATE_UPDATED",
        {
            "uav_id": uav_id,
            "position": {
                "x": 0.0,
                "y": 0.0,
                "z": 20.0,
                "frame_id": "mission_local",
            },
            "availability": "AVAILABLE",
            "battery_percent": 80.0,
            "link_state": "GOOD",
            "link_quality": 0.9,
            "external_workload": 0,
            "mission_status": "IDLE",
            "current_target_id": None,
        },
        sec=sec,
        source={"source_id": "uav-state", "source_uav_id": uav_id},
    )


def target_event(
    sequence: int = 4,
    *,
    sec: int = 0,
    global_target_id: str | None = "target-1",
    local_track_id: str | None = "track-1",
    source_session_id: str | None = "session-1",
    source_uav_id: str = "uav-1",
) -> dict:
    return event(
        sequence,
        "TARGET_OBSERVED",
        {
            "global_target_id": global_target_id,
            "source_uav_id": source_uav_id,
            "local_track_id": local_track_id,
            "class_name": "vehicle",
            "confidence": 0.9,
            "status": "TRACKED",
            "position": {
                "x": 100.0,
                "y": 0.0,
                "z": 0.0,
                "frame_id": "mission_local",
            },
            "priority": "HIGH",
            "required_capabilities": ["TARGET_TRACKING"],
            "continuity_uav_id": None,
        },
        sec=sec,
        source={
            "source_id": "detector",
            "source_uav_id": source_uav_id,
            "source_session_id": source_session_id,
        },
    )


def task_event(sequence: int = 5, *, sec: int = 0) -> dict:
    return event(
        sequence,
        "TASK_CREATED",
        {
            "request_id": "request-1",
            "task_type": "TRACK_TARGET",
            "priority": "HIGH",
            "required_capabilities": [],
            "target_id": "target-1",
            "region_id": None,
            "lifecycle_state": "ACTIVE",
            "reason": "Explicit test request.",
        },
        sec=sec,
    )
