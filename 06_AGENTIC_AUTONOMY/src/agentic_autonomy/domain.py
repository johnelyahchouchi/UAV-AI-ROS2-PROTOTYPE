from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class UAVStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    UNAVAILABLE = "UNAVAILABLE"


class Capability(StrEnum):
    RECONNAISSANCE = "RECONNAISSANCE"
    SURVEILLANCE = "SURVEILLANCE"
    TARGET_TRACKING = "TARGET_TRACKING"
    AREA_SEARCH = "AREA_SEARCH"
    COMMUNICATION_RELAY = "COMMUNICATION_RELAY"


class Priority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TargetStatus(StrEnum):
    UNCONFIRMED = "UNCONFIRMED"
    DETECTED = "DETECTED"
    TRACKED = "TRACKED"
    LOST = "LOST"


class RegionType(StrEnum):
    SEARCH_AREA = "SEARCH_AREA"
    OBSERVATION_AREA = "OBSERVATION_AREA"
    EXCLUSION_ZONE = "EXCLUSION_ZONE"
    OPERATING_AREA = "OPERATING_AREA"


class TaskType(StrEnum):
    SEARCH_REGION = "SEARCH_REGION"
    OBSERVE_REGION = "OBSERVE_REGION"
    INVESTIGATE_TARGET = "INVESTIGATE_TARGET"
    TRACK_TARGET = "TRACK_TARGET"
    RELAY_COMMUNICATIONS = "RELAY_COMMUNICATIONS"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class UAV:
    id: str
    position: Point
    status: UAVStatus
    capabilities: frozenset[Capability]
    battery_percent: float
    link_quality: float
    current_workload: int
    max_workload: int
    max_task_distance: float | None
    current_target_id: str | None


@dataclass(frozen=True)
class Target:
    id: str
    position: Point
    priority: Priority
    status: TargetStatus
    required_capabilities: frozenset[Capability]
    continuity_uav_id: str | None


@dataclass(frozen=True)
class Region:
    id: str
    region_type: RegionType
    vertices: tuple[Point, ...]
    priority: Priority
    required_capabilities: frozenset[Capability]


@dataclass(frozen=True)
class MissionRequest:
    id: str
    task_type: TaskType
    priority: Priority
    required_capabilities: frozenset[Capability]
    target_id: str | None
    region_id: str | None


@dataclass(frozen=True)
class Task:
    id: str
    request_id: str
    task_type: TaskType
    priority: Priority
    required_capabilities: frozenset[Capability]
    location: Point
    target_id: str | None
    region_id: str | None
    sequence: int


@dataclass(frozen=True)
class Scenario:
    schema_version: str
    scenario_id: str
    uavs: tuple[UAV, ...]
    targets: tuple[Target, ...]
    regions: tuple[Region, ...]
    requests: tuple[MissionRequest, ...]
    operating_region_id: str | None
    exclusion_region_ids: tuple[str, ...]

