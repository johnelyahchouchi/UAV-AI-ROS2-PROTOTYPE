from __future__ import annotations

import re

from .domain import Capability, Priority, Scenario, Task, TaskType
from .geometry import polygon_centroid

DEFAULT_CAPABILITY = {
    TaskType.SEARCH_REGION: Capability.AREA_SEARCH,
    TaskType.OBSERVE_REGION: Capability.SURVEILLANCE,
    TaskType.INVESTIGATE_TARGET: Capability.RECONNAISSANCE,
    TaskType.TRACK_TARGET: Capability.TARGET_TRACKING,
    TaskType.RELAY_COMMUNICATIONS: Capability.COMMUNICATION_RELAY,
}
PRIORITY_ORDER = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
TARGET_TASKS = {TaskType.INVESTIGATE_TARGET, TaskType.TRACK_TARGET}


def generate_tasks(scenario: Scenario) -> tuple[Task, ...]:
    targets = {x.id: x for x in scenario.targets}
    regions = {x.id: x for x in scenario.regions}
    ordered = sorted(scenario.requests, key=lambda r: (-PRIORITY_ORDER[r.priority], r.id))
    tasks = []
    for sequence, req in enumerate(ordered, 1):
        if req.task_type in TARGET_TASKS:
            entity = targets[req.target_id]
            location = entity.position
        else:
            entity = regions[req.region_id]
            location = polygon_centroid(entity.vertices)
        capabilities = req.required_capabilities | entity.required_capabilities | {DEFAULT_CAPABILITY[req.task_type]}
        slug = re.sub(r"[^a-z0-9]+", "-", req.id.lower()).strip("-")
        tasks.append(Task(f"task-{sequence:03d}-{slug}", req.id, req.task_type, req.priority,
                          frozenset(capabilities), location, req.target_id, req.region_id, sequence))
    return tuple(tasks)
