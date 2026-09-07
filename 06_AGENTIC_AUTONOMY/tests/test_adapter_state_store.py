from copy import deepcopy

import pytest

from agentic_autonomy.ros2_adapter.errors import AdapterEventError, AdapterLimitError
from agentic_autonomy.ros2_adapter.normalized_events import parse_event
from agentic_autonomy.ros2_adapter.state_store import MissionStateStore

from adapter_test_helpers import (
    adapter_policy,
    event,
    full_uav_state,
    mission_event,
    target_event,
    task_event,
    uav_config_event,
)


def _store_with_uav():
    store = MissionStateStore(adapter_policy())
    store.apply(parse_event(mission_event()))
    store.apply(parse_event(uav_config_event()))
    return store


def test_detection_never_creates_a_task():
    store = _store_with_uav()
    store.apply(parse_event(target_event(sequence=3)))
    assert set(store.state.targets) == {"target-1"}
    assert store.state.tasks == {}


def test_explicit_task_event_is_the_only_task_creation_path():
    store = _store_with_uav()
    store.apply(parse_event(target_event(sequence=3)))
    store.apply(parse_event(task_event(sequence=4)))
    assert set(store.state.tasks) == {"request-1"}


def test_fallback_target_identity_is_source_session_scoped():
    store = _store_with_uav()
    first = target_event(
        sequence=3,
        global_target_id=None,
        local_track_id="7",
        source_session_id="session-a",
    )
    second = target_event(
        sequence=4,
        global_target_id=None,
        local_track_id="7",
        source_session_id="session-b",
    )
    store.apply(parse_event(first))
    store.apply(parse_event(second))
    assert set(store.state.targets) == {
        "target:uav-1:session-a:7",
        "target:uav-1:session-b:7",
    }


def test_legacy_identity_without_session_is_diagnostic_only():
    store = _store_with_uav()
    raw = target_event(
        sequence=3,
        global_target_id=None,
        local_track_id="7",
        source_session_id=None,
    )
    result = store.apply(parse_event(raw))
    assert not result.state_changed
    assert store.state.targets == {}
    assert store.diagnostics[-1].code == "TARGET_IDENTITY_INSUFFICIENT"


def test_duplicate_event_is_idempotent_but_conflict_is_rejected():
    store = _store_with_uav()
    original = full_uav_state(sequence=3)
    store.apply(parse_event(original))
    duplicate = deepcopy(original)
    duplicate["sequence"] = 4
    result = store.apply(parse_event(duplicate))
    assert result.duplicate
    conflicting = deepcopy(original)
    conflicting["sequence"] = 5
    conflicting["payload"]["battery_percent"] = 79.0
    with pytest.raises(AdapterEventError, match="conflicting content"):
        store.apply(parse_event(conflicting))


def test_battery_tolerance_clamps_and_emits_diagnostic():
    store = _store_with_uav()
    raw = full_uav_state(sequence=3)
    raw["payload"]["battery_percent"] = 100.0005
    store.apply(parse_event(raw))
    assert store.state.uavs["uav-1"].battery_percent.value == 100.0
    assert store.diagnostics[-1].code == "BATTERY_CLAMPED_WITHIN_TOLERANCE"


def test_invalid_battery_cannot_partially_mutate_uav_state():
    store = _store_with_uav()
    raw = full_uav_state(sequence=3)
    raw["payload"]["battery_percent"] = 100.1
    with pytest.raises(AdapterEventError, match=r"\[0, 100\]"):
        store.apply(parse_event(raw))
    uav = store.state.uavs["uav-1"]
    assert uav.position is None
    assert uav.battery_percent is None


@pytest.mark.parametrize("terminal_state", ["COMPLETED", "CANCELLED"])
def test_terminal_task_cannot_be_reactivated(terminal_state):
    store = _store_with_uav()
    store.apply(parse_event(target_event(sequence=3)))
    store.apply(parse_event(task_event(sequence=4)))
    store.apply(
        parse_event(
            event(
                5,
                "TASK_LIFECYCLE_CHANGED",
                {
                    "request_id": "request-1",
                    "state": terminal_state,
                    "reason": "Terminal test state.",
                },
            )
        )
    )
    with pytest.raises(AdapterEventError, match="invalid lifecycle transition"):
        store.apply(
            parse_event(
                event(
                    6,
                    "TASK_LIFECYCLE_CHANGED",
                    {"request_id": "request-1", "state": "ACTIVE", "reason": None},
                )
            )
        )


def test_out_of_order_field_update_is_rejected():
    store = _store_with_uav()
    store.apply(parse_event(full_uav_state(sequence=3, sec=5)))
    with pytest.raises(AdapterEventError, match="out-of-order timestamp"):
        store.apply(
            parse_event(
                event(
                    4,
                    "UAV_STATE_UPDATED",
                    {"uav_id": "uav-1", "battery_percent": 79.0},
                    sec=4,
                )
            )
        )


def test_uav_update_rejects_conflicting_source_identity():
    store = _store_with_uav()
    raw = full_uav_state(sequence=3)
    raw["source"]["source_uav_id"] = "another-uav"
    with pytest.raises(AdapterEventError, match="conflicts with event source"):
        store.apply(parse_event(raw))


def test_configured_entity_limits_fail_before_exceeding_limit():
    policy = adapter_policy()
    policy["limits"]["maximum_uavs"] = 1
    store = MissionStateStore(policy)
    store.apply(parse_event(mission_event()))
    store.apply(parse_event(uav_config_event(sequence=2, uav_id="uav-1")))
    with pytest.raises(AdapterLimitError, match="maximum_uavs"):
        store.apply(parse_event(uav_config_event(sequence=3, uav_id="uav-2")))
