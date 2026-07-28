from pathlib import Path

import pytest

from agentic_autonomy.planner import build_plan
from agentic_autonomy.scenario_loader import load_scenario

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["basic_reconnaissance", "capability_and_battery", "target_continuity", "infeasible_mission"])
def test_example_scenarios(name, policy):
    plan = build_plan(load_scenario(ROOT / "scenarios" / f"{name}.json"), policy)
    assert plan["summary"]["generated_task_count"] >= 1
    assert len(plan["deterministic_fingerprint"]) == 64


def test_basic_matches_expected(policy):
    actual = build_plan(load_scenario(ROOT / "scenarios/basic_reconnaissance.json"), policy)
    expected = __import__("json").loads((ROOT / "scenarios/expected/basic_reconnaissance_plan.json").read_text())
    assert actual["scenario_id"] == expected["scenario_id"]
    assert actual["tasks"] == expected["tasks"]
    fields = lambda x: {key: x[key] for key in ("task_id", "uav_id", "decision", "reasons", "score_breakdown")}
    assert [fields(x) for x in actual["assignments"]] == [fields(x) for x in expected["assignments"]]
    assert actual["summary"] == expected["summary"]
    assert actual["deterministic_fingerprint"] == expected["deterministic_fingerprint"]
