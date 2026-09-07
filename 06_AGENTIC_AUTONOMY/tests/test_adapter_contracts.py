import json
from copy import deepcopy

import pytest

from agentic_autonomy.ros2_adapter.adapter_configuration import (
    parse_adapter_policy,
)
from agentic_autonomy.ros2_adapter.errors import (
    AdapterConfigurationError,
    AdapterEventError,
)
from agentic_autonomy.ros2_adapter.normalized_events import (
    parse_event,
    parse_event_stream,
)
from test_serialization import _validate_schema

from adapter_test_helpers import ROOT, adapter_policy, full_uav_state, mission_event


def test_default_adapter_policy_is_strict_and_valid():
    policy = adapter_policy()
    assert policy["adapter_policy_version"] == "1.0"
    assert policy["planning_frame"] == "mission_local"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("limits", "maximum_uavs"), True),
        (("freshness_thresholds_seconds", "position"), float("nan")),
        (("snapshot_trigger", "periodic_interval_seconds"), 0),
    ],
)
def test_adapter_policy_rejects_invalid_numeric_values(path, value):
    policy = json.loads(
        (ROOT / "config" / "ros2_adapter_policy.json").read_text(encoding="utf-8")
    )
    policy[path[0]][path[1]] = value
    with pytest.raises(AdapterConfigurationError):
        parse_adapter_policy(policy)


def test_adapter_policy_rejects_unknown_property():
    policy = json.loads(
        (ROOT / "config" / "ros2_adapter_policy.json").read_text(encoding="utf-8")
    )
    policy["hidden_default"] = 1
    with pytest.raises(AdapterConfigurationError, match="unknown"):
        parse_adapter_policy(policy)


def test_adapter_policy_rejects_wrong_contract_version():
    policy = json.loads(
        (ROOT / "config" / "ros2_adapter_policy.json").read_text(encoding="utf-8")
    )
    policy["adapter_policy_version"] = "2.0"
    with pytest.raises(AdapterConfigurationError, match="expected '1.0'"):
        parse_adapter_policy(policy)


@pytest.mark.parametrize("value", ["80", True, float("nan"), float("inf")])
def test_canonical_battery_rejects_non_numeric_or_non_finite_values(value):
    raw = full_uav_state()
    raw["payload"]["battery_percent"] = value
    with pytest.raises(AdapterEventError):
        parse_event(raw)


def test_event_rejects_unknown_property_and_wrong_version():
    raw = mission_event()
    raw["unknown"] = "not allowed"
    with pytest.raises(AdapterEventError, match="unknown"):
        parse_event(raw)
    raw.pop("unknown")
    raw["schema_version"] = "2.0"
    with pytest.raises(AdapterEventError, match="expected '1.0'"):
        parse_event(raw)


def test_stream_requires_strict_monotonic_sequence_and_matching_mission():
    first = mission_event()
    second = full_uav_state(sequence=1)
    with pytest.raises(AdapterEventError, match="strictly increasing"):
        parse_event_stream(
            {"schema_version": "1.0", "mission_id": "test-mission", "events": [first, second]}
        )
    second["sequence"] = 2
    second["mission_id"] = "different"
    with pytest.raises(AdapterEventError, match="does not match stream"):
        parse_event_stream(
            {"schema_version": "1.0", "mission_id": "test-mission", "events": [first, second]}
        )


def test_event_and_policy_examples_match_their_strict_schemas():
    event_schema = json.loads(
        (ROOT / "schemas" / "adapter_event.schema.json").read_text(encoding="utf-8")
    )
    policy_schema = json.loads(
        (ROOT / "schemas" / "ros2_adapter_policy.schema.json").read_text(
            encoding="utf-8"
        )
    )
    stream = json.loads(
        (ROOT / "scenarios" / "adapter" / "basic_adapter_events.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_schema(stream, event_schema, event_schema)
    raw_policy = json.loads(
        (ROOT / "config" / "ros2_adapter_policy.json").read_text(encoding="utf-8")
    )
    _validate_schema(raw_policy, policy_schema, policy_schema)


def test_task_reference_contract_rejects_wrong_reference_type():
    stream = json.loads(
        (ROOT / "scenarios" / "adapter" / "basic_adapter_events.json").read_text(
            encoding="utf-8"
        )
    )
    raw = deepcopy(stream["events"][4])
    raw["payload"]["target_id"] = None
    raw["payload"]["region_id"] = "some-region"
    with pytest.raises(AdapterEventError, match="target task"):
        parse_event(raw)


def test_lost_link_requires_zero_quality_in_same_event():
    raw = full_uav_state()
    raw["payload"]["link_state"] = "LOST"
    raw["payload"].pop("link_quality")
    with pytest.raises(AdapterEventError, match="requires link_quality 0.0"):
        parse_event(raw)
