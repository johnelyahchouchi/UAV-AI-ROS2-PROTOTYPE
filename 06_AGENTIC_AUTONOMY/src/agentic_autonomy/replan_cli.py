from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import ScenarioError
from .replanner import replan_history
from .scenario_loader import load_policy
from .serialization import write_plan
from .state_history import load_mission_history


def _report(result: dict, output: str, verbose: bool) -> str:
    lines = [f"Mission: {result['mission_id']}", f"Snapshots processed: {len(result['snapshots'])}"]
    for snapshot in result["snapshots"]:
        lines.append(f"\nSnapshot {snapshot['sequence']}: {snapshot['snapshot_id']}")
        for decision in snapshot["decisions"]:
            change = decision["change"]
            lines.append(f"  [{change['change_type']}] {change['request_id']}: "
                         f"{change['previous_uav_id'] or '-'} -> {change['current_uav_id'] or '-'}")
            if verbose:
                lines.extend(f"    - {reason}" for reason in decision["reasons"])
        for advisory in snapshot["return_home"]:
            lines.append(f"  [ADVISORY] {advisory['uav_id']}: {advisory['recommendation']} ({advisory['reason']})")
    lines.extend(["", f"Fingerprint: {result['deterministic_fingerprint']}", f"Output: {output}"])
    return "\n".join(lines)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Deterministic file-based dynamic mission replanner")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--policy", default=str(root / "config/default_policy.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        history = load_mission_history(args.sequence)
        policy = load_policy(args.policy)
        result = replan_history(history, policy)
        write_plan(result, args.output)
        print(_report(result, args.output, args.verbose))
        return 0
    except ScenarioError as exc:
        print(f"Sequence error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Replanner error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
