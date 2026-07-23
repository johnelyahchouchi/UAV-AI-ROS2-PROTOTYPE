from __future__ import annotations

from .replanning_domain import (AssignmentChange, AssignmentChangeType, MissionStateSnapshot,
                                PreviousAssignment, ReplanningDecision, ReplanningDecisionType,
                                ReplanningTrigger, ReplanningTriggerType, TaskLifecycleState)


def assignments_by_request(plan: dict | None, sequence: int) -> dict[str, PreviousAssignment]:
    if not plan:
        return {}
    tasks = {task["id"]: task for task in plan["tasks"]}
    result = {}
    for assignment in plan["assignments"]:
        task = tasks[assignment["task_id"]]
        score = assignment["score_breakdown"]
        result[task["request_id"]] = PreviousAssignment(task["request_id"], task["id"], assignment["uav_id"],
                                                        score["base_total"], score["final_total"], sequence)
    return result


def _unassigned_reasons(plan: dict, request_id: str) -> tuple[str, ...]:
    task = next((item for item in plan["tasks"] if item["request_id"] == request_id), None)
    if not task:
        return ()
    item = next((entry for entry in plan["unassigned_tasks"] if entry["task_id"] == task["id"]), None)
    return tuple(item["reasons"]) if item else ()


def _related_trigger_ids(triggers: list[ReplanningTrigger], request_id: str, previous_uav: str | None,
                         current_uav: str | None, snapshot: MissionStateSnapshot) -> tuple[str, ...]:
    related = {request_id}
    if previous_uav:
        related.add(previous_uav)
    if current_uav:
        related.add(current_uav)
    request = next((item for item in snapshot.scenario.requests if item.id == request_id), None)
    if request:
        related.update(value for value in (request.target_id, request.region_id) if value)
    ids = [trigger.id for trigger in triggers
           if trigger.entity_id in related or trigger.trigger_type in {
               ReplanningTriggerType.INITIAL_SNAPSHOT, ReplanningTriggerType.NO_MEANINGFUL_CHANGE}]
    return tuple(ids)


def compare_plans(previous_snapshot: MissionStateSnapshot | None, current_snapshot: MissionStateSnapshot,
                  previous_plan: dict | None, current_plan: dict, triggers: list[ReplanningTrigger],
                  selection_metadata: dict[str, dict]) -> list[ReplanningDecision]:
    previous_sequence = previous_snapshot.sequence if previous_snapshot else -1
    previous = assignments_by_request(previous_plan, previous_sequence)
    current = assignments_by_request(current_plan, current_snapshot.sequence)
    current_tasks = {task["request_id"]: task for task in current_plan["tasks"]}
    previous_lifecycle = previous_snapshot.lifecycle_by_request() if previous_snapshot else {}
    decisions = []
    for index, lifecycle in enumerate(current_snapshot.task_lifecycle, 1):
        request_id = lifecycle.request_id
        old_assignment = previous.get(request_id)
        new_assignment = current.get(request_id)
        old_state = previous_lifecycle.get(request_id)
        old_state_value = old_state.state if old_state else None
        current_task_id = current_tasks.get(request_id, {}).get("id")
        score_comparison = selection_metadata.get(request_id)
        if lifecycle.state == TaskLifecycleState.COMPLETED:
            change_type, decision_type = AssignmentChangeType.COMPLETED, ReplanningDecisionType.REMOVE_COMPLETED
            reasons = (lifecycle.reason or f"Task {request_id} was marked completed.",)
        elif lifecycle.state == TaskLifecycleState.CANCELLED:
            change_type, decision_type = AssignmentChangeType.CANCELLED, ReplanningDecisionType.REMOVE_CANCELLED
            reasons = (lifecycle.reason or f"Task {request_id} was cancelled.",)
        elif lifecycle.state == TaskLifecycleState.PENDING:
            change_type, decision_type = AssignmentChangeType.REMAINS_UNASSIGNED, ReplanningDecisionType.KEEP_UNASSIGNED
            reasons = (f"Task {request_id} remains PENDING and is excluded from active allocation.",)
        elif new_assignment and not old_assignment:
            change_type, decision_type = AssignmentChangeType.NEW_ASSIGNMENT, ReplanningDecisionType.CREATE_ASSIGNMENT
            assignment = next(item for item in current_plan["assignments"] if item["task_id"] == new_assignment.task_id)
            reasons = (f"Task {request_id} received a new assignment to {new_assignment.uav_id}.", *assignment["reasons"])
        elif new_assignment and old_assignment and new_assignment.uav_id == old_assignment.uav_id:
            change_type, decision_type = AssignmentChangeType.UNCHANGED, ReplanningDecisionType.KEEP_ASSIGNMENT
            assignment = next(item for item in current_plan["assignments"] if item["task_id"] == new_assignment.task_id)
            reasons = (f"Task {request_id} remains assigned to {new_assignment.uav_id}.", *assignment["reasons"])
        elif new_assignment and old_assignment:
            change_type, decision_type = AssignmentChangeType.REASSIGNED, ReplanningDecisionType.TRANSFER_ASSIGNMENT
            assignment = next(item for item in current_plan["assignments"] if item["task_id"] == new_assignment.task_id)
            reasons = (f"Task {request_id} transferred from {old_assignment.uav_id} to {new_assignment.uav_id}.",
                       *assignment["reasons"])
        elif old_assignment:
            change_type, decision_type = AssignmentChangeType.NEWLY_UNASSIGNED, ReplanningDecisionType.UNASSIGN
            reasons = (f"Task {request_id} is newly unassigned because no current candidate passed selection and safety.",
                       *_unassigned_reasons(current_plan, request_id))
        else:
            change_type, decision_type = AssignmentChangeType.REMAINS_UNASSIGNED, ReplanningDecisionType.KEEP_UNASSIGNED
            reasons = (f"Task {request_id} remains unassigned.", *_unassigned_reasons(current_plan, request_id))
        change = AssignmentChange(change_type, request_id, old_assignment.task_id if old_assignment else None,
                                  current_task_id, old_assignment.uav_id if old_assignment else None,
                                  new_assignment.uav_id if new_assignment else None,
                                  old_assignment.final_score if old_assignment else None,
                                  new_assignment.final_score if new_assignment else None)
        trigger_ids = _related_trigger_ids(triggers, request_id, change.previous_uav_id,
                                           change.current_uav_id, current_snapshot)
        decisions.append(ReplanningDecision(f"decision-{current_snapshot.sequence:03d}-{index:03d}",
                                            current_snapshot.sequence, decision_type, change, trigger_ids,
                                            score_comparison, tuple(reasons)))
    return decisions


def decision_to_dict(decision: ReplanningDecision) -> dict:
    change = decision.change
    return {
        "id": decision.id, "sequence": decision.sequence, "decision_type": decision.decision_type.value,
        "change": {"change_type": change.change_type.value, "request_id": change.request_id,
                   "previous_task_id": change.previous_task_id, "current_task_id": change.current_task_id,
                   "previous_uav_id": change.previous_uav_id, "current_uav_id": change.current_uav_id,
                   "previous_score": change.previous_score, "current_score": change.current_score},
        "trigger_ids": list(decision.trigger_ids), "score_comparison": decision.score_comparison,
        "reasons": list(decision.reasons),
    }
