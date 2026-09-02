import hashlib
import json

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.ros2_adapter.adapter import MissionStateAdapter
from agentic_autonomy.ros2_adapter.normalized_events import load_event_stream
from agentic_autonomy.ros2_adapter.serialization import write_canonical_json
from agentic_autonomy.ros2_adapter.validation import validate_phase2_history
from agentic_autonomy.state_history import load_mission_history, parse_mission_history
from test_serialization import _validate_schema

from adapter_test_helpers import ROOT, adapter_policy, planner_policy


def _run_basic():
    policy = adapter_policy()
    events = load_event_stream(
        ROOT / "scenarios" / "adapter" / "basic_adapter_events.json", policy
    )
    adapter = MissionStateAdapter(policy, planner_policy())
    return adapter.process_events(events)


def test_basic_offline_history_matches_expected_fixture_byte_for_byte(tmp_path):
    history = _run_basic()
    actual = tmp_path / "history.json"
    write_canonical_json(history, actual)
    expected = ROOT / "scenarios" / "expected" / "basic_adapter_history.json"
    assert actual.read_bytes() == expected.read_bytes()


def test_independent_adapter_runs_are_deterministic():
    first = json.dumps(_run_basic(), sort_keys=True, separators=(",", ":")).encode()
    second = json.dumps(_run_basic(), sort_keys=True, separators=(",", ":")).encode()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_adapter_output_is_strict_phase2_schema_compatible():
    history = _run_basic()
    schema = json.loads(
        (ROOT / "schemas" / "mission_state_sequence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    scenario_schema = json.loads(
        (ROOT / "schemas" / "mission_scenario.schema.json").read_text(encoding="utf-8")
    )
    for snapshot in history["snapshots"]:
        _validate_schema(snapshot["scenario"], scenario_schema, scenario_schema)
    schema["$defs"]["snapshot"]["properties"]["scenario"] = {"type": "object"}
    _validate_schema(history, schema, schema)
    parsed = validate_phase2_history(history)
    assert parsed.mission_id == "adapter-basic-mission"


def test_file_and_in_memory_phase2_parsers_are_equivalent():
    source = ROOT / "scenarios" / "replanning" / "battery_degradation.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    assert parse_mission_history(raw) == load_mission_history(source)


def test_link_loss_demo_reassigns_task_through_existing_replanner():
    policy = adapter_policy()
    events = load_event_stream(
        ROOT / "scenarios" / "adapter" / "stale_link_loss_events.json", policy
    )
    history = MissionStateAdapter(policy, planner_policy()).process_events(events)
    result = replan_history(validate_phase2_history(history), planner_policy())
    second = result["snapshots"][1]
    assert second["decisions"][0]["change"]["change_type"] == "REASSIGNED"
    assert second["decisions"][0]["change"]["previous_uav_id"] == "uav-alpha"
    assert second["decisions"][0]["change"]["current_uav_id"] == "uav-charlie"
    assert second["return_home"][0]["reason"] == "LINK_BELOW_MINIMUM"
    rejected = next(
        item
        for item in second["plan"]["candidate_evaluations"][
            "task-001-request-track-delta"
        ]
        if item["uav_id"] == "uav-alpha"
    )
    assert rejected["decision"] == "REJECTED"
    assert any("LINK_QUALITY failed" in reason for reason in rejected["reasons"])
