from copy import deepcopy
from pathlib import Path

from agentic_autonomy.explanation import console_report
from agentic_autonomy.planner import build_plan
from agentic_autonomy.scenario_loader import load_scenario

ROOT = Path(__file__).parents[1]


def test_console_report_is_readable_and_stable(basic, policy):
    plan = build_plan(basic, policy)
    report = console_report(plan, "output.json")
    assert "Mission: basic-reconnaissance" in report
    assert "[APPROVED]" in report
    assert "Fingerprint:" in report
    assert "0x" not in report
    reasons = [reason for assignment in plan["assignments"] for reason in assignment["reasons"]]
    assert any("Required capabilities matched" in reason for reason in reasons)
    assert any("Battery" in reason and "margin" in reason for reason in reasons)
    assert any("Link quality" in reason for reason in reasons)
    assert any("Distance to task" in reason for reason in reasons)
    assert any("changes workload" in reason for reason in reasons)
    assert any("runner-up" in reason for reason in reasons)
    assert "All deterministic eligibility and safety rules passed." not in report


def test_continuity_winner_explains_bonus(policy):
    continuity_policy = deepcopy(policy)
    continuity_policy["allocation_weights"] = {
        "capability": 30, "battery": 10, "distance": 5,
        "link_quality": 5, "workload": 0, "target_continuity": 50,
    }
    scenario = load_scenario(ROOT / "scenarios/target_continuity.json")
    assignment = build_plan(scenario, continuity_policy)["assignments"][0]
    assert assignment["uav_id"] == "uav-continuity"
    assert any("Target continuity matched" in reason for reason in assignment["reasons"])
