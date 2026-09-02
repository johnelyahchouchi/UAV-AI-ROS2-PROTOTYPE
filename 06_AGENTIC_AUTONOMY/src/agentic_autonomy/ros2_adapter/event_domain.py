from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AdapterEventType(StrEnum):
    MISSION_CONFIGURED = "MISSION_CONFIGURED"
    REGION_UPDATED = "REGION_UPDATED"
    UAV_CONFIGURED = "UAV_CONFIGURED"
    UAV_STATE_UPDATED = "UAV_STATE_UPDATED"
    TARGET_OBSERVED = "TARGET_OBSERVED"
    TARGET_STATE_UPDATED = "TARGET_STATE_UPDATED"
    TASK_CREATED = "TASK_CREATED"
    TASK_UPDATED = "TASK_UPDATED"
    TASK_LIFECYCLE_CHANGED = "TASK_LIFECYCLE_CHANGED"
    SNAPSHOT_TICK = "SNAPSHOT_TICK"
    SNAPSHOT_REQUESTED = "SNAPSHOT_REQUESTED"


class AdapterAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class LinkState(StrEnum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    UNKNOWN = "UNKNOWN"


class MissionStatus(StrEnum):
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    RETURNING = "RETURNING"
    LANDED = "LANDED"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class FreshnessState(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"


class DiagnosticSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, order=True)
class EventTimestamp:
    clock_id: str
    sec: int
    nanosec: int

    @property
    def total_nanoseconds(self) -> int:
        return self.sec * 1_000_000_000 + self.nanosec

    @property
    def label(self) -> str:
        return f"{self.clock_id}:{self.sec}.{self.nanosec:09d}"


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    source_node: str | None
    topic: str | None
    message_type: str | None
    source_uav_id: str | None
    source_session_id: str | None
    source_timestamp: str | None
    upstream_sequence: int | None


@dataclass(frozen=True)
class NormalizedEvent:
    schema_version: str
    mission_id: str
    event_id: str
    sequence: int
    event_type: AdapterEventType
    observed_at: EventTimestamp
    source: SourceMetadata
    payload: dict[str, Any]


@dataclass(frozen=True)
class Position3D:
    x: float
    y: float
    z: float | None
    frame_id: str


@dataclass(frozen=True)
class TimestampedValue:
    value: Any
    observed_at: EventTimestamp


@dataclass
class UAVAdapterState:
    id: str
    capabilities: frozenset[str]
    max_workload: int
    max_task_distance: float | None
    position: TimestampedValue | None = None
    availability: TimestampedValue | None = None
    battery_percent: TimestampedValue | None = None
    link_state: TimestampedValue | None = None
    link_quality: TimestampedValue | None = None
    external_workload: TimestampedValue | None = None
    mission_status: TimestampedValue | None = None
    current_target_id: TimestampedValue | None = None


@dataclass
class TargetAdapterState:
    id: str
    source_uav_id: str
    source_session_id: str | None
    local_track_id: str | None
    class_name: TimestampedValue
    confidence: TimestampedValue
    status: TimestampedValue
    position: TimestampedValue | None
    priority: TimestampedValue | None
    required_capabilities: frozenset[str]
    continuity_uav_id: str | None
    first_observed_at: EventTimestamp
    last_observed_at: EventTimestamp


@dataclass
class TaskAdapterState:
    request_id: str
    task_type: str
    priority: str
    required_capabilities: frozenset[str]
    target_id: str | None
    region_id: str | None
    lifecycle_state: str
    reason: str | None
    created_at: EventTimestamp
    updated_at: EventTimestamp


@dataclass(frozen=True)
class AdapterDiagnostic:
    id: str
    severity: DiagnosticSeverity
    code: str
    message: str
    event_sequence: int | None
    entity_type: str | None
    entity_id: str | None


@dataclass
class AdapterMissionState:
    mission_id: str | None = None
    scenario_id: str | None = None
    clock_id: str | None = None
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    operating_region_id: str | None = None
    exclusion_region_ids: tuple[str, ...] = ()
    uavs: dict[str, UAVAdapterState] = field(default_factory=dict)
    targets: dict[str, TargetAdapterState] = field(default_factory=dict)
    tasks: dict[str, TaskAdapterState] = field(default_factory=dict)


@dataclass(frozen=True)
class EventApplicationResult:
    accepted: bool
    state_changed: bool
    duplicate: bool
    diagnostics: tuple[AdapterDiagnostic, ...]


@dataclass(frozen=True)
class SnapshotBuildResult:
    snapshot: dict[str, Any]
    diagnostics: tuple[AdapterDiagnostic, ...]
