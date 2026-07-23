import json
from pathlib import Path

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.state_history import load_mission_history

ROOT = Path(__file__).parents[1]


def test_request_identity_survives_task_number_change(tmp_path, policy):
    data = json.loads((ROOT / "scenarios/replanning/no_meaningful_change.json").read_text(encoding="utf-8"))
    snapshot = data["snapshots"][1]
    snapshot["scenario"]["targets"].append({"id":"target-critical","position":{"x":50,"y":0},"priority":"CRITICAL","status":"DETECTED","required_capabilities":[],"continuity_uav_id":None})
    snapshot["scenario"]["mission_requests"].append({"id":"critical-new","task_type":"TRACK_TARGET","priority":"CRITICAL","required_capabilities":[],"target_id":"target-critical","region_id":None})
    snapshot["task_lifecycle"].append({"request_id":"critical-new","state":"ACTIVE"})
    path = tmp_path / "renumber.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = replan_history(load_mission_history(path), policy)
    decisions = {item["change"]["request_id"]: item for item in result["snapshots"][1]["decisions"]}
    old = decisions["track-target"]["change"]
    assert old["change_type"] == "UNCHANGED"
    assert old["previous_task_id"] != old["current_task_id"]
    assert decisions["critical-new"]["change"]["change_type"] == "NEW_ASSIGNMENT"

