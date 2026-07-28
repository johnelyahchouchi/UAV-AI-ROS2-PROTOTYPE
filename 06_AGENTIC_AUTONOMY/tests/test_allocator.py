from pathlib import Path

from agentic_autonomy.allocator import allocate
from agentic_autonomy.scenario_loader import load_scenario
from agentic_autonomy.task_generator import generate_tasks

ROOT = Path(__file__).parents[1]


def test_capability_and_battery_rules(policy):
    scenario = load_scenario(ROOT / "scenarios/capability_and_battery.json")
    assignments, _, evaluations = allocate(generate_tasks(scenario), scenario, policy)
    assert assignments[0]["uav_id"] == "uav-capable-safe"
    decisions = {x["uav_id"]: x for x in next(iter(evaluations.values()))}
    assert decisions["uav-capable-low"]["decision"] == "REJECTED"
    assert decisions["uav-near-wrong-capability"]["score_breakdown"]["capability_score"] == 0


def test_continuity_is_scored(policy):
    scenario = load_scenario(ROOT / "scenarios/target_continuity.json")
    _, _, evaluations = allocate(generate_tasks(scenario), scenario, policy)
    candidates = {x["uav_id"]: x for x in next(iter(evaluations.values()))}
    assert candidates["uav-continuity"]["score_breakdown"]["target_continuity_score"] == 1000
    assert candidates["uav-nearby"]["score_breakdown"]["target_continuity_score"] == 0

