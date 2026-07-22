from pathlib import Path

from agentic_autonomy.planner import build_plan
from agentic_autonomy.scenario_loader import load_scenario

ROOT = Path(__file__).parents[1]


def test_infeasible_task_stays_unassigned(policy):
    plan = build_plan(load_scenario(ROOT / "scenarios/infeasible_mission.json"), policy)
    assert plan["assignments"] == []
    assert len(plan["unassigned_tasks"]) == 1
    assert plan["unassigned_tasks"][0]["candidate_score_breakdowns"]


def test_every_assignment_passes_every_check(basic, policy):
    plan = build_plan(basic, policy)
    assert all(all(c["passed"] for c in a["safety_checks"]) for a in plan["assignments"])

