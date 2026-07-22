from __future__ import annotations

from . import __version__
from .allocator import allocate
from .serialization import fingerprint, task_dict
from .task_generator import generate_tasks


def build_plan(scenario, policy) -> dict:
    tasks = generate_tasks(scenario)
    assignments, unassigned, evaluations = allocate(tasks, scenario, policy)
    counts = {u.id: sum(a["uav_id"] == u.id for a in assignments) for u in sorted(scenario.uavs, key=lambda x: x.id)}
    plan = {"schema_version": "1.0", "planner_version": __version__, "policy_version": policy["policy_version"],
            "scenario_id": scenario.scenario_id, "tasks": [task_dict(t) for t in tasks],
            "assignments": assignments, "unassigned_tasks": unassigned, "candidate_evaluations": evaluations,
            "summary": {"generated_task_count": len(tasks), "assigned_task_count": len(assignments),
                        "unassigned_task_count": len(unassigned), "uav_assignment_counts": counts,
                        "all_assignments_safe": all(all(c["passed"] for c in a["safety_checks"]) for a in assignments)}}
    plan["deterministic_fingerprint"] = fingerprint(plan)
    return plan

