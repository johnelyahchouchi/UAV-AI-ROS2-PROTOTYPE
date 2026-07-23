from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Scenario


class TaskLifecycleState(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ReplanningTriggerType(StrEnum):
    INITIAL_SNAPSHOT = "INITIAL_SNAPSHOT"
    TASK_CREATED = "TASK_CREATED"
    TASK_ACTIVATED = "TASK_ACTIVATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_CANCELLED = "TASK_CANCELLED"
    BATTERY_CHANGED = "BATTERY_CHANGED"
    BATTERY_BELOW_RESERVE = "BATTERY_BELOW_RESERVE"
    BATTERY_CRITICAL = "BATTERY_CRITICAL"
    LINK_QUALITY_CHANGED = "LINK_QUALITY_CHANGED"
    LINK_BELOW_MINIMUM = "LINK_BELOW_MINIMUM"
    UAV_AVAILABILITY_CHANGED = "UAV_AVAILABILITY_CHANGED"
    UAV_CAPABILITIES_CHANGED = "UAV_CAPABILITIES_CHANGED"
    UAV_WORKLOAD_CHANGED = "UAV_WORKLOAD_CHANGED"
    UAV_POSITION_CHANGED = "UAV_POSITION_CHANGED"
    TARGET_MOVED = "TARGET_MOVED"
    TARGET_STATE_CHANGED = "TARGET_STATE_CHANGED"
    REGION_CHANGED = "REGION_CHANGED"
    SAFETY_ELIGIBILITY_CHANGED = "SAFETY_ELIGIBILITY_CHANGED"
    HYSTERESIS_THRESHOLD_EXCEEDED = "HYSTERESIS_THRESHOLD_EXCEEDED"
    NO_MEANINGFUL_CHANGE = "NO_MEANINGFUL_CHANGE"


class AssignmentChangeType(StrEnum):
    UNCHANGED = "UNCHANGED"
    NEW_ASSIGNMENT = "NEW_ASSIGNMENT"
    REASSIGNED = "REASSIGNED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    NEWLY_UNASSIGNED = "NEWLY_UNASSIGNED"
    REMAINS_UNASSIGNED = "REMAINS_UNASSIGNED"


class ReplanningDecisionType(StrEnum):
    KEEP_ASSIGNMENT = "KEEP_ASSIGNMENT"
    CREATE_ASSIGNMENT = "CREATE_ASSIGNMENT"
    TRANSFER_ASSIGNMENT = "TRANSFER_ASSIGNMENT"
    REMOVE_COMPLETED = "REMOVE_COMPLETED"
    REMOVE_CANCELLED = "REMOVE_CANCELLED"
    UNASSIGN = "UNASSIGN"
    KEEP_UNASSIGNED = "KEEP_UNASSIGNED"


class ReturnHomeReason(StrEnum):
    BATTERY_BELOW_RESERVE = "BATTERY_BELOW_RESERVE"
    BATTERY_CRITICAL = "BATTERY_CRITICAL"
    LINK_BELOW_MINIMUM = "LINK_BELOW_MINIMUM"


class ReturnHomeRecommendation(StrEnum):
    RETURN_HOME = "RETURN_HOME_RECOMMENDED"
    URGENT_RETURN_OR_LAND = "URGENT_SAFE_RETURN_OR_LANDING_RECOMMENDED"


@dataclass(frozen=True)
class TaskLifecycleRecord:
    request_id: str
    state: TaskLifecycleState
    reason: str | None


@dataclass(frozen=True)
class MissionStateSnapshot:
    snapshot_id: str
    sequence: int
    timestamp: str | None
    scenario: Scenario
    task_lifecycle: tuple[TaskLifecycleRecord, ...]

    def lifecycle_by_request(self) -> dict[str, TaskLifecycleRecord]:
        return {item.request_id: item for item in self.task_lifecycle}


@dataclass(frozen=True)
class MissionHistory:
    mission_id: str
    snapshots: tuple[MissionStateSnapshot, ...]


@dataclass(frozen=True)
class PreviousAssignment:
    request_id: str
    task_id: str
    uav_id: str
    base_score: int
    final_score: int
    snapshot_sequence: int


@dataclass(frozen=True)
class ReplanningTrigger:
    id: str
    trigger_type: ReplanningTriggerType
    entity_type: str
    entity_id: str
    previous_value: object
    current_value: object
    explanation: str


@dataclass(frozen=True)
class AssignmentChange:
    change_type: AssignmentChangeType
    request_id: str
    previous_task_id: str | None
    current_task_id: str | None
    previous_uav_id: str | None
    current_uav_id: str | None
    previous_score: int | None
    current_score: int | None


@dataclass(frozen=True)
class ReplanningDecision:
    id: str
    sequence: int
    decision_type: ReplanningDecisionType
    change: AssignmentChange
    trigger_ids: tuple[str, ...]
    score_comparison: dict | None
    reasons: tuple[str, ...]

