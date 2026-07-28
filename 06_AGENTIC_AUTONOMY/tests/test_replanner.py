from copy import deepcopy
from pathlib import Path

import pytest

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.state_history import load_mission_history

ROOT = Path(__file__).parents[1]


def _run(name, policy):
    history = load_mission_history(ROOT / f"scenarios/replanning/{name}.json")
    return replan_history(history, policy)


def test_battery_degradation_reassigns_and_recommends_return(policy):
    result = _run("battery_degradation", policy)
    second = result["snapshots"][1]
    assert second["decisions"][0]["change"]["change_type"] == "REASSIGNED"
    assert second["decisions"][0]["change"]["current_uav_id"] == "uav-b"
    assert second["return_home"][0]["reason"] == "BATTERY_BELOW_RESERVE"


def test_battery_degradation_matches_expected_fixture(policy):
    import json
    expected = json.loads((ROOT / "scenarios/expected/battery_degradation_result.json").read_text(encoding="utf-8"))
    result = _run("battery_degradation", policy)
    actual = []
    for snapshot in result["snapshots"]:
        change = snapshot["decisions"][0]["change"]
        actual.append({"sequence": snapshot["sequence"], "change_type": change["change_type"],
                       "previous_uav_id": change["previous_uav_id"], "current_uav_id": change["current_uav_id"],
                       "advisory_reason": snapshot["return_home"][0]["reason"] if snapshot["return_home"] else None})
    assert actual == expected["expected_snapshots"]


def test_critical_battery_is_distinct_from_reserve(policy):
    result = _run("battery_degradation", policy)
    assert result["snapshots"][1]["return_home"][0]["recommendation"] == "RETURN_HOME_RECOMMENDED"
    assert result["snapshots"][2]["return_home"][0]["reason"] == "BATTERY_CRITICAL"
    assert result["snapshots"][2]["return_home"][0]["recommendation"] == "URGENT_SAFE_RETURN_OR_LANDING_RECOMMENDED"


def test_uav_failure_reassigns(policy):
    result = _run("uav_failure", policy)
    assert result["snapshots"][1]["decisions"][0]["change"]["change_type"] == "REASSIGNED"


def test_link_degradation_reassigns_with_execution_caveat(policy):
    result = _run("link_degradation", policy)
    second = result["snapshots"][1]
    assert second["decisions"][0]["change"]["current_uav_id"] == "uav-b"
    assert "may not be executable" in second["return_home"][0]["explanation"]


def test_task_completion_removes_assignment(policy):
    result = _run("task_completion", policy)
    assert result["snapshots"][1]["decisions"][0]["change"]["change_type"] == "COMPLETED"
    assert result["snapshots"][1]["plan"]["assignments"] == []


def test_no_meaningful_change_is_exclusive(policy):
    result = _run("no_meaningful_change", policy)
    for snapshot in result["snapshots"]:
        kinds = [item["trigger_type"] for item in snapshot["triggers"]]
        if "NO_MEANINGFUL_CHANGE" in kinds:
            assert kinds == ["NO_MEANINGFUL_CHANGE"]
    assert result["snapshots"][1]["decisions"][0]["change"]["change_type"] == "UNCHANGED"


def test_hysteresis_below_threshold_retains_incumbent(policy):
    probe_policy = deepcopy(policy)
    probe_policy["replanning"]["minimum_base_score_improvement_to_switch"] = 10000
    probe = _run("target_movement", probe_policy)
    comparison = probe["snapshots"][1]["decisions"][0]["score_comparison"]
    improvement = comparison["improvement"]
    test_policy = deepcopy(policy)
    test_policy["replanning"]["minimum_base_score_improvement_to_switch"] = improvement + 1
    result = _run("target_movement", test_policy)
    decision = result["snapshots"][1]["decisions"][0]
    assert decision["change"]["change_type"] == "UNCHANGED"
    assert "below the configurable" in decision["reasons"][-1]


def test_hysteresis_exact_threshold_replaces_incumbent(policy):
    probe_policy = deepcopy(policy)
    probe_policy["replanning"]["minimum_base_score_improvement_to_switch"] = 10000
    improvement = _run("target_movement", probe_policy)["snapshots"][1]["decisions"][0]["score_comparison"]["improvement"]
    exact_policy = deepcopy(policy)
    exact_policy["replanning"]["minimum_base_score_improvement_to_switch"] = improvement
    decision = _run("target_movement", exact_policy)["snapshots"][1]["decisions"][0]
    assert decision["change"]["change_type"] == "REASSIGNED"
    assert any(item["trigger_type"] == "HYSTERESIS_THRESHOLD_EXCEEDED"
               for item in _run("target_movement", exact_policy)["snapshots"][1]["triggers"])


def test_unsafe_incumbent_replaced_regardless_of_margin(policy):
    strict = deepcopy(policy)
    strict["replanning"]["minimum_base_score_improvement_to_switch"] = 100000
    decision = _run("battery_degradation", strict)["snapshots"][1]["decisions"][0]
    assert decision["change"]["change_type"] == "REASSIGNED"
    assert decision["score_comparison"]["hysteresis_applied"] is False


def test_snapshot_workload_is_not_counted_twice(tmp_path, policy):
    import json
    data = json.loads((ROOT / "scenarios/replanning/no_meaningful_change.json").read_text(encoding="utf-8"))
    data["snapshots"] = [data["snapshots"][0]]
    scenario = data["snapshots"][0]["scenario"]
    scenario["targets"].append({"id":"target-2","position":{"x":150,"y":0},"priority":"MEDIUM","status":"DETECTED","required_capabilities":[],"continuity_uav_id":None})
    scenario["mission_requests"].append({"id":"track-second","task_type":"TRACK_TARGET","priority":"MEDIUM","required_capabilities":[],"target_id":"target-2","region_id":None})
    data["snapshots"][0]["task_lifecycle"].append({"request_id":"track-second","state":"ACTIVE"})
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    plan = replan_history(load_mission_history(path), policy)["snapshots"][0]["plan"]
    assert len(plan["assignments"]) == 2
    assert "from 1 to 2" in plan["assignments"][1]["reasons"][4]


@pytest.mark.parametrize(("field", "value", "trigger"), [
    ("capabilities", [], "UAV_CAPABILITIES_CHANGED"),
    ("current_workload", 2, "UAV_WORKLOAD_CHANGED"),
])
def test_capability_or_external_workload_can_make_task_unassigned(tmp_path, policy, field, value, trigger):
    import json
    data = json.loads((ROOT / "scenarios/replanning/no_meaningful_change.json").read_text(encoding="utf-8"))
    data["snapshots"][1]["scenario"]["uavs"][0][field] = value
    path = tmp_path / f"{field}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    second = replan_history(load_mission_history(path), policy)["snapshots"][1]
    assert second["decisions"][0]["change"]["change_type"] == "NEWLY_UNASSIGNED"
    assert any(item["trigger_type"] == trigger for item in second["triggers"])


def test_cancelled_task_is_reported(tmp_path, policy):
    import json
    data = json.loads((ROOT / "scenarios/replanning/task_completion.json").read_text(encoding="utf-8"))
    data["snapshots"][1]["task_lifecycle"][0]["state"] = "CANCELLED"
    data["snapshots"][1]["task_lifecycle"][0]["reason"] = "Operator cancelled task"
    path = tmp_path / "cancelled.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    decision = replan_history(load_mission_history(path), policy)["snapshots"][1]["decisions"][0]
    assert decision["change"]["change_type"] == "CANCELLED"
