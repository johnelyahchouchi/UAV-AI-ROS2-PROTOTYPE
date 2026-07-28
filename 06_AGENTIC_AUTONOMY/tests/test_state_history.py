import json
from pathlib import Path

import pytest

from agentic_autonomy.errors import ScenarioError
from agentic_autonomy.replanning_domain import TaskLifecycleState
from agentic_autonomy.state_history import load_mission_history

ROOT = Path(__file__).parents[1]


def _sequence(name="no_meaningful_change"):
    return json.loads((ROOT / f"scenarios/replanning/{name}.json").read_text(encoding="utf-8"))


def _write(tmp_path, data):
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_ordered_history():
    history = load_mission_history(ROOT / "scenarios/replanning/no_meaningful_change.json")
    assert [item.sequence for item in history.snapshots] == [1, 2]


def test_sequence_must_strictly_increase(tmp_path):
    data = _sequence()
    data["snapshots"][1]["sequence"] = 1
    with pytest.raises(ScenarioError, match="strictly increasing"):
        load_mission_history(_write(tmp_path, data))


def test_missing_request_is_rejected(tmp_path):
    data = _sequence()
    data["snapshots"][1]["scenario"]["mission_requests"] = []
    data["snapshots"][1]["task_lifecycle"] = []
    with pytest.raises(ScenarioError, match="may not disappear"):
        load_mission_history(_write(tmp_path, data))


def test_terminal_task_cannot_reactivate(tmp_path):
    data = _sequence("task_completion")
    third = json.loads(json.dumps(data["snapshots"][1]))
    third["snapshot_id"], third["sequence"] = "complete-003", 3
    third["task_lifecycle"][0]["state"] = "ACTIVE"
    data["snapshots"].append(third)
    with pytest.raises(ScenarioError, match="terminal task"):
        load_mission_history(_write(tmp_path, data))


def test_new_request_may_begin_active(tmp_path):
    data = _sequence()
    snapshot = data["snapshots"][1]
    snapshot["scenario"]["targets"].append({"id":"target-2","position":{"x":150,"y":0},"priority":"CRITICAL","status":"DETECTED","required_capabilities":[],"continuity_uav_id":None})
    snapshot["scenario"]["mission_requests"].append({"id":"new-active","task_type":"TRACK_TARGET","priority":"CRITICAL","required_capabilities":[],"target_id":"target-2","region_id":None})
    snapshot["task_lifecycle"].append({"request_id":"new-active","state":"ACTIVE","reason":"Urgent new target"})
    history = load_mission_history(_write(tmp_path, data))
    assert history.snapshots[1].lifecycle_by_request()["new-active"].state == TaskLifecycleState.ACTIVE

