from __future__ import annotations

import hashlib

from . import __version__
from .domain import UAVStatus
from .lifecycle import active_scenario
from .plan_diff import assignments_by_request, compare_plans, decision_to_dict
from .planner import build_plan
from .replanning_domain import (AssignmentChangeType, MissionHistory, ReplanningTrigger, ReplanningTriggerType,
                                ReturnHomeReason, ReturnHomeRecommendation)
from .serialization import canonical_json
from .state_history import assign_trigger_ids, detect_state_triggers


def canonical_fingerprint(value: object) -> str:
    def without_fingerprints(item):
        if isinstance(item, dict):
            return {key: without_fingerprints(child) for key, child in item.items()
                    if key != "deterministic_fingerprint"}
        if isinstance(item, list):
            return [without_fingerprints(child) for child in item]
        return item
    canonical = canonical_json(without_fingerprints(value))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _selector(previous_assignments: dict, margin: int, metadata: dict, events: list):
    def select(task, candidates):
        approved = [item for item in candidates if item["decision"] == "APPROVED"]
        challenger = approved[0]
        old = previous_assignments.get(task.request_id)
        if old is None:
            return challenger, []
        incumbent = next((item for item in candidates if item["uav_id"] == old.uav_id), None)
        if incumbent is None or incumbent["decision"] != "APPROVED":
            failed = incumbent["reasons"] if incumbent else ["The previous UAV is absent from the current fleet."]
            metadata[task.request_id] = {
                "incumbent_uav_id": old.uav_id, "incumbent_base_score": None,
                "challenger_uav_id": challenger["uav_id"],
                "challenger_base_score": challenger["score_breakdown"]["base_total"],
                "improvement": None, "required_improvement": margin, "hysteresis_applied": False,
            }
            events.append((ReplanningTriggerType.SAFETY_ELIGIBILITY_CHANGED, task.request_id,
                           f"Previous assignee {old.uav_id} is no longer safe or eligible: {'; '.join(failed)}"))
            return challenger, [f"Unsafe incumbent {old.uav_id} was not retained; {challenger['uav_id']} is the highest-ranked safe replacement."]
        if challenger["uav_id"] == incumbent["uav_id"]:
            metadata[task.request_id] = {
                "incumbent_uav_id": old.uav_id, "incumbent_base_score": incumbent["score_breakdown"]["base_total"],
                "challenger_uav_id": None, "challenger_base_score": None, "improvement": 0,
                "required_improvement": margin, "hysteresis_applied": True,
            }
            return incumbent, [f"Assignment continuity retained {old.uav_id}, which remains the highest-ranked safe candidate."]
        improvement = challenger["score_breakdown"]["base_total"] - incumbent["score_breakdown"]["base_total"]
        metadata[task.request_id] = {
            "incumbent_uav_id": old.uav_id, "incumbent_base_score": incumbent["score_breakdown"]["base_total"],
            "challenger_uav_id": challenger["uav_id"], "challenger_base_score": challenger["score_breakdown"]["base_total"],
            "improvement": improvement, "required_improvement": margin, "hysteresis_applied": True,
        }
        if improvement >= margin:
            events.append((ReplanningTriggerType.HYSTERESIS_THRESHOLD_EXCEEDED, task.request_id,
                           f"{challenger['uav_id']} improved the base score by {improvement}, meeting the {margin}-point threshold."))
            return challenger, [f"Challenger {challenger['uav_id']} improved the base score by {improvement}, meeting the configurable {margin}-point switching threshold."]
        return incumbent, [f"Assignment continuity retained {old.uav_id}: challenger {challenger['uav_id']} improved the base score by only {improvement}, below the configurable {margin}-point switching threshold."]
    return select


def _return_home(snapshot, policy):
    records = []
    reserve = policy["safety_thresholds"]["minimum_battery_reserve_percent"]
    minimum_link = policy["safety_thresholds"]["minimum_link_quality"]
    critical = policy["replanning"]["critical_battery_percent"]
    link_enabled = policy["replanning"]["return_home_on_link_below_minimum"]
    for uav in sorted(snapshot.scenario.uavs, key=lambda item: item.id):
        if uav.status == UAVStatus.UNAVAILABLE:
            continue
        if uav.battery_percent < critical:
            records.append({"uav_id": uav.id, "advisory": True,
                            "recommendation": ReturnHomeRecommendation.URGENT_RETURN_OR_LAND.value,
                            "reason": ReturnHomeReason.BATTERY_CRITICAL.value, "observed_value": uav.battery_percent,
                            "threshold": critical,
                            "explanation": f"{uav.id} battery is critical; urgent safe return or landing is recommended. This record does not command the aircraft."})
        elif uav.battery_percent < reserve:
            records.append({"uav_id": uav.id, "advisory": True,
                            "recommendation": ReturnHomeRecommendation.RETURN_HOME.value,
                            "reason": ReturnHomeReason.BATTERY_BELOW_RESERVE.value, "observed_value": uav.battery_percent,
                            "threshold": reserve,
                            "explanation": f"{uav.id} battery is below reserve; return home is recommended. This record does not command the aircraft."})
        if link_enabled and uav.link_quality < minimum_link:
            records.append({"uav_id": uav.id, "advisory": True,
                            "recommendation": ReturnHomeRecommendation.RETURN_HOME.value,
                            "reason": ReturnHomeReason.LINK_BELOW_MINIMUM.value, "observed_value": uav.link_quality,
                            "threshold": minimum_link,
                            "explanation": f"{uav.id} link is below minimum; return home is recommended, but the recommendation may not be executable because communication is already degraded or lost. No command is generated."})
    return records


def _trigger_to_dict(trigger):
    return {"id": trigger.id, "trigger_type": trigger.trigger_type.value, "entity_type": trigger.entity_type,
            "entity_id": trigger.entity_id, "previous_value": trigger.previous_value,
            "current_value": trigger.current_value, "explanation": trigger.explanation}


def replan_history(history: MissionHistory, policy: dict) -> dict:
    snapshot_outputs = []
    previous_snapshot = None
    previous_plan = None
    for snapshot in history.snapshots:
        metadata = {}
        selection_events = []
        previous_assignments = assignments_by_request(previous_plan, previous_snapshot.sequence if previous_snapshot else -1)
        selector = _selector(previous_assignments,
                             policy["replanning"]["minimum_base_score_improvement_to_switch"],
                             metadata, selection_events)
        plan = build_plan(active_scenario(snapshot), policy, candidate_selector=selector)
        triggers = detect_state_triggers(previous_snapshot, snapshot, policy)
        for kind, request_id, explanation in selection_events:
            triggers.append(ReplanningTrigger("", kind, "TASK", request_id, None, None, explanation))
        if not triggers:
            triggers.append(ReplanningTrigger("", ReplanningTriggerType.NO_MEANINGFUL_CHANGE, "MISSION",
                                              history.mission_id, None, None,
                                              "No meaningful state or planning trigger was detected."))
        triggers = assign_trigger_ids(triggers, snapshot.sequence)
        decisions = compare_plans(previous_snapshot, snapshot, previous_plan, plan, triggers, metadata)
        return_home = _return_home(snapshot, policy)
        counts = {kind.value: 0 for kind in AssignmentChangeType}
        for decision in decisions:
            counts[decision.change.change_type.value] += 1
        record = {
            "snapshot_id": snapshot.snapshot_id, "sequence": snapshot.sequence, "timestamp": snapshot.timestamp,
            "triggers": [_trigger_to_dict(item) for item in triggers], "plan": plan,
            "decisions": [decision_to_dict(item) for item in decisions], "return_home": return_home,
            "summary": {"assignment_changes": counts, "return_home_recommendations": len(return_home)},
        }
        record["deterministic_fingerprint"] = canonical_fingerprint(record)
        snapshot_outputs.append(record)
        previous_snapshot, previous_plan = snapshot, plan
    result = {"schema_version": "2.0", "replanner_version": __version__, "mission_id": history.mission_id,
              "policy_version": policy["policy_version"], "snapshots": snapshot_outputs}
    result["deterministic_fingerprint"] = canonical_fingerprint(result)
    return result
