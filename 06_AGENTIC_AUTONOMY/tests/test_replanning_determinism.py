import hashlib
from copy import deepcopy
from pathlib import Path

from agentic_autonomy.planner import build_plan
from agentic_autonomy.replanner import canonical_fingerprint, replan_history
from agentic_autonomy.scenario_loader import load_scenario
from agentic_autonomy.serialization import canonical_json
from agentic_autonomy.state_history import load_mission_history

ROOT = Path(__file__).parents[1]


def test_replanning_is_byte_deterministic(policy):
    history = load_mission_history(ROOT / "scenarios/replanning/battery_degradation.json")
    outputs = [canonical_json(replan_history(history, policy)) for _ in range(10)]
    assert len(set(outputs)) == 1


def test_canonical_fingerprint_excludes_all_fingerprint_fields(policy):
    history = load_mission_history(ROOT / "scenarios/replanning/no_meaningful_change.json")
    result = replan_history(history, policy)
    assert canonical_fingerprint(result) == result["deterministic_fingerprint"]
    changed_fingerprints = deepcopy(result)
    changed_fingerprints["deterministic_fingerprint"] = "0" * 64
    changed_fingerprints["snapshots"][0]["deterministic_fingerprint"] = "1" * 64
    changed_fingerprints["snapshots"][0]["plan"]["deterministic_fingerprint"] = "2" * 64
    assert canonical_fingerprint(changed_fingerprints) == result["deterministic_fingerprint"]
    changed_data = deepcopy(result)
    changed_data["mission_id"] = "different"
    assert canonical_fingerprint(changed_data) != result["deterministic_fingerprint"]


def test_phase1_output_remains_byte_identical(policy):
    scenario = load_scenario(ROOT / "scenarios/basic_reconnaissance.json")
    output_hash = hashlib.sha256(canonical_json(build_plan(scenario, policy)).encode("utf-8")).hexdigest()
    assert output_hash == "8ba518b3f4b77cea0e76698dc9ab8f7a96ad117085aa2ac7d54bd91f15bd7fa3"

