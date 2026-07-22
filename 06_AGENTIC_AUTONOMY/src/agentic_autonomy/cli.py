from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import ScenarioError
from .explanation import console_report
from .planner import build_plan
from .scenario_loader import load_policy, load_scenario
from .serialization import write_plan


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Deterministic offline UAV mission planner")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", default=str(root / "config" / "default_policy.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        policy = load_policy(args.policy)
        if args.validate_only:
            print(f"Valid scenario: {scenario.scenario_id}")
            return 0
        plan = build_plan(scenario, policy)
        write_plan(plan, args.output)
        print(console_report(plan, args.output))
        return 4 if args.require_complete and plan["unassigned_tasks"] else 0
    except ScenarioError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 3

