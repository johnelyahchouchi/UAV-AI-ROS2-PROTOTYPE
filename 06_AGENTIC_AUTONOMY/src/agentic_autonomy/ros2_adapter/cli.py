from __future__ import annotations

import argparse
import sys

from agentic_autonomy.errors import ScenarioError
from agentic_autonomy.replanner import replan_history
from agentic_autonomy.scenario_loader import load_policy

from .adapter import MissionStateAdapter
from .adapter_configuration import load_adapter_policy
from .errors import AdapterError
from .normalized_events import load_event_stream
from .serialization import write_canonical_json
from .validation import validate_phase2_history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline deterministic ROS 2 mission-state adapter replay"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--adapter-policy", required=True)
    parser.add_argument("--planner-policy", required=True)
    parser.add_argument("--snapshot-output", required=True)
    parser.add_argument("--diagnostics-output", required=True)
    parser.add_argument("--replanning-output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        adapter_policy = load_adapter_policy(args.adapter_policy)
        planner_policy = load_policy(args.planner_policy)
        events = load_event_stream(args.events, adapter_policy)
        adapter = MissionStateAdapter(adapter_policy, planner_policy)
        history = adapter.process_events(events)
        write_canonical_json(history, args.snapshot_output)
        diagnostics = adapter.diagnostics_document()
        write_canonical_json(diagnostics, args.diagnostics_output)
        if args.replanning_output:
            result = replan_history(validate_phase2_history(history), planner_policy)
            write_canonical_json(result, args.replanning_output)
        print(f"Mission: {history['mission_id']}")
        print(f"Events processed: {len(events)}")
        print(f"Snapshots emitted: {len(history['snapshots'])}")
        print(f"Diagnostics: {len(diagnostics['diagnostics'])}")
        print(f"Snapshot output: {args.snapshot_output}")
        if args.replanning_output:
            print(f"Replanning output: {args.replanning_output}")
        if args.verbose:
            for snapshot in history["snapshots"]:
                print(
                    f"  Snapshot {snapshot['sequence']}: {snapshot['snapshot_id']} "
                    f"at {snapshot['timestamp']}"
                )
            for diagnostic in diagnostics["diagnostics"]:
                print(f"  [{diagnostic['severity']}] {diagnostic['code']}: {diagnostic['message']}")
        return 0
    except (AdapterError, ScenarioError) as exc:
        print(f"Adapter error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected adapter error: {exc}", file=sys.stderr)
        return 3
