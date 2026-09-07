from copy import deepcopy

import pytest

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.ros2_adapter.adapter import MissionStateAdapter
from agentic_autonomy.ros2_adapter.errors import AdapterLimitError, AdapterSnapshotError
from agentic_autonomy.ros2_adapter.normalized_events import parse_event
from agentic_autonomy.ros2_adapter.validation import validate_phase2_history

from adapter_test_helpers import (
    adapter_policy,
    event,
    full_uav_state,
    mission_event,
    planner_policy,
    target_event,
    task_event,
    uav_config_event,
)


@pytest.mark.parametrize(
    "stale_fields",
    [
        ("position",),
        ("availability",),
        ("battery_percent",),
        ("link_state", "link_quality"),
    ],
)
def test_stale_required_uav_field_cannot_pass_planner_eligibility(stale_fields):
    old = full_uav_state(sequence=3, sec=0)
    old["payload"] = {
        "uav_id": "uav-1",
        **{name: old["payload"][name] for name in stale_fields},
    }
    fresh = full_uav_state(sequence=4, sec=10)
    for name in stale_fields:
        fresh["payload"].pop(name)
    raw_events = [
        mission_event(),
        uav_config_event(),
        old,
        fresh,
        target_event(sequence=5, sec=10),
        task_event(sequence=6, sec=10),
        event(7, "SNAPSHOT_TICK", {"reason": "freshness test"}, sec=10),
    ]
    adapter = MissionStateAdapter(adapter_policy(), planner_policy())
    history_raw = adapter.process_events(parse_event(item) for item in raw_events)
    projected = history_raw["snapshots"][0]["scenario"]["uavs"][0]
    assert projected["status"] == "UNAVAILABLE"
    result = replan_history(validate_phase2_history(history_raw), planner_policy())
    candidate = result["snapshots"][0]["plan"]["candidate_evaluations"][
        "task-001-request-1"
    ][0]
    assert candidate["decision"] == "REJECTED"
    assert any("UAV_AVAILABLE failed" in reason for reason in candidate["reasons"])


def test_missing_required_dynamic_state_is_never_fabricated():
    raw_events = [
        mission_event(),
        uav_config_event(),
        event(
            3,
            "UAV_STATE_UPDATED",
            {"uav_id": "uav-1", "battery_percent": 80.0},
        ),
        event(4, "SNAPSHOT_TICK", {"reason": "incomplete state"}, sec=1),
    ]
    adapter = MissionStateAdapter(adapter_policy(), planner_policy())
    with pytest.raises(AdapterSnapshotError, match="without producing"):
        adapter.process_events(parse_event(item) for item in raw_events)
    assert any(
        item.code == "UAV_OMITTED_MISSING_STATE"
        for item in adapter.store.diagnostics
    )


def test_specifically_missing_battery_prevents_snapshot_eligibility():
    incomplete = full_uav_state()
    incomplete["payload"].pop("battery_percent")
    adapter = MissionStateAdapter(adapter_policy(), planner_policy())
    raw_events = [
        mission_event(),
        uav_config_event(),
        incomplete,
        event(4, "SNAPSHOT_TICK", {"reason": "missing battery"}, sec=1),
    ]
    with pytest.raises(AdapterSnapshotError, match="without producing"):
        adapter.process_events(parse_event(item) for item in raw_events)
    diagnostic = next(
        item
        for item in adapter.store.diagnostics
        if item.code == "UAV_OMITTED_MISSING_STATE"
    )
    assert "battery_percent" in diagnostic.message


def test_explicit_unavailable_projects_uav_unavailable():
    adapter = MissionStateAdapter(adapter_policy(), planner_policy())
    initial = [mission_event(), uav_config_event(), full_uav_state()]
    for raw in initial:
        assert adapter.process_event(parse_event(raw)) is None
    snapshot = adapter.process_event(
        parse_event(
            event(
                4,
                "UAV_STATE_UPDATED",
                {"uav_id": "uav-1", "availability": "UNAVAILABLE"},
                sec=1,
            )
        )
    )
    assert snapshot["scenario"]["uavs"][0]["status"] == "UNAVAILABLE"


def test_snapshot_limit_fails_without_truncating_existing_history():
    policy = adapter_policy()
    policy["limits"]["maximum_snapshots"] = 1
    adapter = MissionStateAdapter(policy, planner_policy())
    raw_events = [
        mission_event(),
        uav_config_event(),
        full_uav_state(),
        target_event(),
        task_event(),
        event(6, "SNAPSHOT_TICK", {"reason": "first"}, sec=1),
        event(7, "SNAPSHOT_REQUESTED", {"reason": "second"}, sec=1),
    ]
    parsed = [parse_event(item) for item in raw_events]
    for item in parsed[:6]:
        adapter.process_event(item)
    with pytest.raises(AdapterLimitError, match="maximum_snapshots"):
        adapter.process_event(parsed[6])
    assert len(adapter.snapshots) == 1
