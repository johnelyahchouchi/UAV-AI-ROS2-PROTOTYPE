from __future__ import annotations

from math import hypot
from pathlib import Path

from .errors import ScenarioError
from .replanning_domain import (MissionHistory, MissionStateSnapshot, ReplanningTrigger,
                                ReplanningTriggerType, TaskLifecycleRecord, TaskLifecycleState)
from .scenario_loader import (_array, _enum, _integer, _keys, _object, _read_json, _string,
                              parse_scenario)

SEQUENCE_KEYS = {"schema_version", "mission_id", "snapshots"}
SNAPSHOT_KEYS = {"snapshot_id", "sequence", "timestamp", "scenario", "task_lifecycle"}
LIFECYCLE_KEYS = {"request_id", "state", "reason"}
TRIGGER_ORDER = {value: index for index, value in enumerate(ReplanningTriggerType)}


def load_mission_history(path: str | Path) -> MissionHistory:
    raw = _object(_read_json(path, "mission state sequence"), "sequence")
    _keys(raw, SEQUENCE_KEYS, SEQUENCE_KEYS, "sequence")
    version = _string(raw["schema_version"], "sequence.schema_version")
    if version != "2.0":
        raise ScenarioError(f"unsupported sequence.schema_version {version!r}; expected '2.0'")
    mission_id = _string(raw["mission_id"], "sequence.mission_id")
    items = _array(raw["snapshots"], "sequence.snapshots")
    if not items:
        raise ScenarioError("sequence.snapshots must contain at least one snapshot")
    snapshots = []
    seen_snapshot_ids = set()
    previous_sequence = None
    previous_scenario_id = None
    previous_requests = set()
    previous_lifecycle = {}
    for index, item in enumerate(items):
        path_name = f"sequence.snapshots[{index}]"
        data = _object(item, path_name)
        _keys(data, SNAPSHOT_KEYS, SNAPSHOT_KEYS - {"timestamp"}, path_name)
        snapshot_id = _string(data["snapshot_id"], f"{path_name}.snapshot_id")
        if snapshot_id in seen_snapshot_ids:
            raise ScenarioError(f"duplicate snapshot_id: {snapshot_id}")
        seen_snapshot_ids.add(snapshot_id)
        sequence = _integer(data["sequence"], f"{path_name}.sequence")
        if sequence < 0 or (previous_sequence is not None and sequence <= previous_sequence):
            raise ScenarioError("snapshot sequence values must be nonnegative and strictly increasing in input order")
        previous_sequence = sequence
        timestamp = _string(data.get("timestamp"), f"{path_name}.timestamp", nullable=True)
        scenario = parse_scenario(data["scenario"])
        if previous_scenario_id is not None and scenario.scenario_id != previous_scenario_id:
            raise ScenarioError("scenario_id must remain constant across mission snapshots")
        previous_scenario_id = scenario.scenario_id

        lifecycle_items = _array(data["task_lifecycle"], f"{path_name}.task_lifecycle")
        lifecycle = []
        lifecycle_ids = set()
        for life_index, life_item in enumerate(lifecycle_items):
            life_path = f"{path_name}.task_lifecycle[{life_index}]"
            life_data = _object(life_item, life_path)
            _keys(life_data, LIFECYCLE_KEYS, LIFECYCLE_KEYS - {"reason"}, life_path)
            request_id = _string(life_data["request_id"], f"{life_path}.request_id")
            if request_id in lifecycle_ids:
                raise ScenarioError(f"duplicate lifecycle record for request {request_id}")
            lifecycle_ids.add(request_id)
            lifecycle.append(TaskLifecycleRecord(
                request_id, _enum(TaskLifecycleState, life_data["state"], f"{life_path}.state"),
                _string(life_data.get("reason"), f"{life_path}.reason", nullable=True)))
        request_ids = {request.id for request in scenario.requests}
        if lifecycle_ids != request_ids:
            missing = sorted(request_ids - lifecycle_ids)
            extra = sorted(lifecycle_ids - request_ids)
            raise ScenarioError(f"lifecycle records must exactly match mission requests; missing={missing}, extra={extra}")
        if previous_requests - request_ids:
            raise ScenarioError(f"mission requests may not disappear: {sorted(previous_requests - request_ids)}")
        current_lifecycle = {record.request_id: record.state for record in lifecycle}
        for request_id in sorted(request_ids):
            current = current_lifecycle[request_id]
            previous = previous_lifecycle.get(request_id)
            if previous is None and current not in {TaskLifecycleState.PENDING, TaskLifecycleState.ACTIVE}:
                raise ScenarioError(f"new request {request_id} must begin as PENDING or ACTIVE")
            if previous == TaskLifecycleState.PENDING and current not in {TaskLifecycleState.PENDING, TaskLifecycleState.ACTIVE, TaskLifecycleState.CANCELLED}:
                raise ScenarioError(f"invalid lifecycle transition for {request_id}: {previous.value} -> {current.value}")
            if previous == TaskLifecycleState.ACTIVE and current not in {TaskLifecycleState.ACTIVE, TaskLifecycleState.COMPLETED, TaskLifecycleState.CANCELLED}:
                raise ScenarioError(f"invalid lifecycle transition for {request_id}: {previous.value} -> {current.value}")
            if previous in {TaskLifecycleState.COMPLETED, TaskLifecycleState.CANCELLED} and current != previous:
                raise ScenarioError(f"terminal task {request_id} cannot transition from {previous.value} to {current.value}")
        previous_requests = request_ids
        previous_lifecycle = current_lifecycle
        snapshots.append(MissionStateSnapshot(snapshot_id, sequence, timestamp, scenario,
                                               tuple(sorted(lifecycle, key=lambda record: record.request_id))))
    return MissionHistory(mission_id, tuple(snapshots))


def _trigger(kind, entity_type, entity_id, previous, current, explanation):
    return ReplanningTrigger("", kind, entity_type, entity_id, previous, current, explanation)


def detect_state_triggers(previous: MissionStateSnapshot | None, current: MissionStateSnapshot,
                          policy: dict) -> list[ReplanningTrigger]:
    triggers = []
    if previous is None:
        triggers.append(_trigger(ReplanningTriggerType.INITIAL_SNAPSHOT, "MISSION", current.scenario.scenario_id,
                                 None, current.sequence, "Initial mission-state snapshot."))
    old_lifecycle = previous.lifecycle_by_request() if previous else {}
    for record in current.task_lifecycle:
        old = old_lifecycle.get(record.request_id)
        if old is None:
            triggers.append(_trigger(ReplanningTriggerType.TASK_CREATED, "TASK", record.request_id, None,
                                     record.state.value, f"Task {record.request_id} was introduced as {record.state.value}."))
        elif old.state != record.state:
            kind = {TaskLifecycleState.ACTIVE: ReplanningTriggerType.TASK_ACTIVATED,
                    TaskLifecycleState.COMPLETED: ReplanningTriggerType.TASK_COMPLETED,
                    TaskLifecycleState.CANCELLED: ReplanningTriggerType.TASK_CANCELLED}[record.state]
            triggers.append(_trigger(kind, "TASK", record.request_id, old.state.value, record.state.value,
                                     f"Task {record.request_id} changed from {old.state.value} to {record.state.value}."))

    old_uavs = {item.id: item for item in previous.scenario.uavs} if previous else {}
    reserve = policy["safety_thresholds"]["minimum_battery_reserve_percent"]
    minimum_link = policy["safety_thresholds"]["minimum_link_quality"]
    critical = policy["replanning"]["critical_battery_percent"]
    for uav in current.scenario.uavs:
        old = old_uavs.get(uav.id)
        if old and old.battery_percent != uav.battery_percent:
            triggers.append(_trigger(ReplanningTriggerType.BATTERY_CHANGED, "UAV", uav.id, old.battery_percent,
                                     uav.battery_percent, f"{uav.id} battery changed from {old.battery_percent:.2f}% to {uav.battery_percent:.2f}%."))
        if uav.battery_percent < critical:
            triggers.append(_trigger(ReplanningTriggerType.BATTERY_CRITICAL, "UAV", uav.id, reserve,
                                     uav.battery_percent, f"{uav.id} battery is critically low at {uav.battery_percent:.2f}%."))
        elif uav.battery_percent < reserve:
            triggers.append(_trigger(ReplanningTriggerType.BATTERY_BELOW_RESERVE, "UAV", uav.id, reserve,
                                     uav.battery_percent, f"{uav.id} battery is below the {reserve:.2f}% reserve."))
        if old and old.link_quality != uav.link_quality:
            triggers.append(_trigger(ReplanningTriggerType.LINK_QUALITY_CHANGED, "UAV", uav.id, old.link_quality,
                                     uav.link_quality, f"{uav.id} link quality changed from {old.link_quality:.3f} to {uav.link_quality:.3f}."))
        if uav.link_quality < minimum_link:
            triggers.append(_trigger(ReplanningTriggerType.LINK_BELOW_MINIMUM, "UAV", uav.id, minimum_link,
                                     uav.link_quality, f"{uav.id} link quality is below the {minimum_link:.3f} minimum."))
        if old and old.status != uav.status:
            triggers.append(_trigger(ReplanningTriggerType.UAV_AVAILABILITY_CHANGED, "UAV", uav.id,
                                     old.status.value, uav.status.value, f"{uav.id} status changed from {old.status.value} to {uav.status.value}."))
        if old and old.capabilities != uav.capabilities:
            triggers.append(_trigger(ReplanningTriggerType.UAV_CAPABILITIES_CHANGED, "UAV", uav.id,
                                     sorted(old.capabilities), sorted(uav.capabilities), f"{uav.id} capabilities changed."))
        if old and old.current_workload != uav.current_workload:
            triggers.append(_trigger(ReplanningTriggerType.UAV_WORKLOAD_CHANGED, "UAV", uav.id,
                                     old.current_workload, uav.current_workload, f"{uav.id} external workload changed."))
        if old and old.position != uav.position:
            triggers.append(_trigger(ReplanningTriggerType.UAV_POSITION_CHANGED, "UAV", uav.id,
                                     {"x": old.position.x, "y": old.position.y}, {"x": uav.position.x, "y": uav.position.y},
                                     f"{uav.id} position changed."))
    for missing_id in sorted(set(old_uavs) - {item.id for item in current.scenario.uavs}):
        triggers.append(_trigger(ReplanningTriggerType.UAV_AVAILABILITY_CHANGED, "UAV", missing_id,
                                 old_uavs[missing_id].status.value, "ABSENT", f"{missing_id} is absent from the current fleet snapshot."))

    old_targets = {item.id: item for item in previous.scenario.targets} if previous else {}
    movement_threshold = policy["replanning"]["target_movement_trigger_distance"]
    for target in current.scenario.targets:
        old = old_targets.get(target.id)
        if old:
            movement = hypot(target.position.x - old.position.x, target.position.y - old.position.y)
            if movement >= movement_threshold and movement > 0:
                triggers.append(_trigger(ReplanningTriggerType.TARGET_MOVED, "TARGET", target.id, 0,
                                         round(movement, 6), f"{target.id} moved {movement:.6f} scenario units."))
            if (old.status, old.priority, old.required_capabilities) != (target.status, target.priority, target.required_capabilities):
                triggers.append(_trigger(ReplanningTriggerType.TARGET_STATE_CHANGED, "TARGET", target.id,
                                         old.status.value, target.status.value, f"{target.id} state or planning requirements changed."))
    old_regions = {item.id: item for item in previous.scenario.regions} if previous else {}
    for region in current.scenario.regions:
        if region.id in old_regions and region != old_regions[region.id]:
            triggers.append(_trigger(ReplanningTriggerType.REGION_CHANGED, "REGION", region.id, "previous", "current",
                                     f"Region {region.id} changed."))
    return triggers


def assign_trigger_ids(triggers: list[ReplanningTrigger], sequence: int) -> list[ReplanningTrigger]:
    ordered = sorted(triggers, key=lambda item: (TRIGGER_ORDER[item.trigger_type], item.entity_type, item.entity_id,
                                                 str(item.previous_value), str(item.current_value)))
    return [ReplanningTrigger(f"trigger-{sequence:03d}-{index:03d}", item.trigger_type, item.entity_type,
                              item.entity_id, item.previous_value, item.current_value, item.explanation)
            for index, item in enumerate(ordered, 1)]
