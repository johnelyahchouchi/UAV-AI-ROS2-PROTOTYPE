from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

from .domain import Scenario, Task, UAV, UAVStatus
from .errors import ScenarioError
from .geometry import distance, point_in_polygon


def _round(value: float) -> int:
    if isinstance(value, bool) or not isfinite(value):
        raise ScenarioError(f"calculated score must be finite, received {value!r}")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _clamp(value: int, scale: int) -> int:
    return max(0, min(scale, value))


def _score(task: Task, uav: UAV, workload: int, scenario: Scenario, policy: dict) -> dict:
    scale = policy["score_scale"]
    t = policy["safety_thresholds"]
    w = policy["allocation_weights"]
    d = distance(uav.position, task.location)
    battery_denominator = max(1.0, 100.0 - t["minimum_battery_reserve_percent"])
    link_denominator = max(1e-12, 1.0 - t["minimum_link_quality"])
    effective_max = min(uav.max_workload, t["maximum_tasks_per_uav"])
    target = next((x for x in scenario.targets if x.id == task.target_id), None)
    components = {
        "capability": scale if task.required_capabilities <= uav.capabilities else 0,
        "battery": _clamp(_round(scale * (uav.battery_percent - t["minimum_battery_reserve_percent"]) / battery_denominator), scale),
        "distance": _clamp(_round(scale * (1.0 - d / t["distance_normalization"])), scale),
        "link_quality": _clamp(_round(scale * (uav.link_quality - t["minimum_link_quality"]) / link_denominator), scale),
        "workload": _clamp(_round(scale * (1.0 - workload / effective_max)), scale),
        "target_continuity": scale if target and target.continuity_uav_id == uav.id else 0,
    }
    base = _round(sum(components[k] * w[k] for k in w) / 100)
    multiplier = policy["priority_multipliers"][task.priority.value]
    final = _round(base * multiplier / scale)
    return {**{f"{k}_score": v for k, v in components.items()}, "priority_multiplier": multiplier,
            "base_total": base, "final_total": final, "distance": round(d, 6)}


def _checks(task: Task, uav: UAV, workload: int, scenario: Scenario, policy: dict) -> list[dict]:
    t = policy["safety_thresholds"]
    regions = {x.id: x for x in scenario.regions}
    target = next((x for x in scenario.targets if x.id == task.target_id), None)
    effective_max = min(uav.max_workload, t["maximum_tasks_per_uav"])
    status_ok = uav.status == UAVStatus.AVAILABLE or (uav.status == UAVStatus.BUSY and target and uav.current_target_id == target.id)
    pairs = [
        ("UAV_AVAILABLE", status_ok, uav.status.value, "AVAILABLE or active continuity UAV"),
        ("CAPABILITY_MATCH", task.required_capabilities <= uav.capabilities,
         sorted(x.value for x in uav.capabilities), sorted(x.value for x in task.required_capabilities)),
        ("BATTERY_RESERVE", uav.battery_percent >= t["minimum_battery_reserve_percent"], uav.battery_percent, t["minimum_battery_reserve_percent"]),
        ("LINK_QUALITY", uav.link_quality >= t["minimum_link_quality"], uav.link_quality, t["minimum_link_quality"]),
        ("WORKLOAD_LIMIT", workload < effective_max, workload, f"less than {effective_max}"),
        ("TASK_CAPACITY", uav.max_task_distance is None or distance(uav.position, task.location) <= uav.max_task_distance, round(distance(uav.position, task.location), 6), uav.max_task_distance),
    ]
    if scenario.operating_region_id:
        region = regions[scenario.operating_region_id]
        pairs.append(("OPERATING_BOUNDARY", point_in_polygon(task.location, region.vertices), {"x": task.location.x, "y": task.location.y}, scenario.operating_region_id))
    outside = all(not point_in_polygon(task.location, regions[rid].vertices) for rid in scenario.exclusion_region_ids)
    pairs.append(("EXCLUSION_ZONE", outside, {"x": task.location.x, "y": task.location.y}, list(scenario.exclusion_region_ids)))
    continuity_ok = not (t["continuity_required"] and target and target.continuity_uav_id) or target.continuity_uav_id == uav.id
    pairs.append(("TARGET_CONTINUITY", continuity_ok, uav.id, target.continuity_uav_id if target else None))
    return [{"rule": name, "passed": bool(ok), "observed_value": observed, "required_value": required,
             "reason": f"{name} {'passed' if ok else 'failed'}: observed {observed!r}; required {required!r}."}
            for name, ok, observed, required in pairs]


def _approval_reasons(task: Task, uav: UAV, workload: int, score: dict, policy: dict,
                      runner_up: dict | None = None) -> list[str]:
    thresholds = policy["safety_thresholds"]
    effective_max = min(uav.max_workload, thresholds["maximum_tasks_per_uav"])
    capabilities = ", ".join(sorted(item.value for item in task.required_capabilities)) or "none"
    reasons = [
        f"Required capabilities matched: {capabilities}.",
        f"Battery {uav.battery_percent:.2f}% passed the {thresholds['minimum_battery_reserve_percent']:.2f}% reserve with a {uav.battery_percent - thresholds['minimum_battery_reserve_percent']:.2f}-point margin.",
        f"Link quality {uav.link_quality:.3f} passed the {thresholds['minimum_link_quality']:.3f} threshold with a {uav.link_quality - thresholds['minimum_link_quality']:.3f} margin.",
        f"Distance to task is {score['distance']:.6f} scenario units; distance score is {score['distance_score']}.",
        f"Assignment changes workload from {workload} to {workload + 1} of {effective_max}; workload score is {score['workload_score']}.",
    ]
    if score["target_continuity_score"]:
        contribution = _round(score["target_continuity_score"] * policy["allocation_weights"]["target_continuity"] / 100)
        reasons.append(f"Target continuity matched this UAV, contributing {contribution} base-score points.")
    if runner_up is not None:
        difference = score["final_total"] - runner_up["score_breakdown"]["final_total"]
        reasons.append(f"Final score {score['final_total']} is {difference} points above eligible runner-up {runner_up['uav_id']} ({runner_up['score_breakdown']['final_total']}).")
    return reasons


def allocate(tasks: tuple[Task, ...], scenario: Scenario, policy: dict) -> tuple[list[dict], list[dict], dict]:
    workloads = {u.id: u.current_workload for u in scenario.uavs}
    assignments, unassigned, evaluations = [], [], {}
    for task in tasks:
        candidates = []
        for uav in sorted(scenario.uavs, key=lambda x: x.id):
            score = _score(task, uav, workloads[uav.id], scenario, policy)
            checks = _checks(task, uav, workloads[uav.id], scenario, policy)
            passed = all(x["passed"] for x in checks)
            reasons = ([x["reason"] for x in checks if not x["passed"]]
                       if not passed else _approval_reasons(task, uav, workloads[uav.id], score, policy))
            candidates.append({"uav_id": uav.id, "decision": "APPROVED" if passed else "REJECTED",
                               "reasons": reasons, "score_breakdown": score, "safety_checks": checks})
        candidates.sort(key=lambda c: (-c["score_breakdown"]["final_total"], -c["score_breakdown"]["target_continuity_score"],
                                       -c["score_breakdown"]["capability_score"], -c["score_breakdown"]["battery_score"],
                                       -c["score_breakdown"]["distance_score"], -c["score_breakdown"]["link_quality_score"],
                                       -c["score_breakdown"]["workload_score"], c["uav_id"]))
        evaluations[task.id] = candidates
        winner = next((c for c in candidates if c["decision"] == "APPROVED"), None)
        if winner:
            eligible = [candidate for candidate in candidates if candidate["decision"] == "APPROVED"]
            runner_up = next((candidate for candidate in eligible if candidate is not winner), None)
            winner_uav = next(uav for uav in scenario.uavs if uav.id == winner["uav_id"])
            winner["reasons"] = _approval_reasons(task, winner_uav, workloads[winner_uav.id],
                                                   winner["score_breakdown"], policy, runner_up)
            workloads[winner["uav_id"]] += 1
            assignments.append({"task_id": task.id, "uav_id": winner["uav_id"], "decision": "APPROVED",
                                "reasons": winner["reasons"], "score_breakdown": winner["score_breakdown"],
                                "safety_checks": winner["safety_checks"]})
        else:
            reasons = [f"{c['uav_id']}: {reason}" for c in candidates for reason in c["reasons"]]
            unassigned.append({"task_id": task.id, "decision": "REJECTED", "reasons": reasons,
                               "candidate_score_breakdowns": {c["uav_id"]: c["score_breakdown"] for c in candidates}})
    return assignments, unassigned, evaluations
