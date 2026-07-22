from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .domain import (Capability, MissionRequest, Point, Priority, Region, RegionType,
                     Scenario, Target, TargetStatus, TaskType, UAV, UAVStatus)
from .errors import ScenarioError
from .geometry import polygon_centroid

SCENARIO_KEYS = {"schema_version", "scenario_id", "uavs", "targets", "regions", "mission_requests",
                 "operating_region_id", "exclusion_region_ids"}
UAV_KEYS = {"id", "position", "status", "capabilities", "battery_percent", "link_quality",
            "current_workload", "max_workload", "max_task_distance", "current_target_id"}
TARGET_KEYS = {"id", "position", "priority", "status", "required_capabilities", "continuity_uav_id"}
REGION_KEYS = {"id", "region_type", "vertices", "priority", "required_capabilities"}
REQUEST_KEYS = {"id", "task_type", "priority", "required_capabilities", "target_id", "region_id"}
POINT_KEYS = {"x", "y"}
POLICY_KEYS = {"policy_version", "safety_thresholds", "allocation_weights", "priority_multipliers", "score_scale"}
THRESHOLD_KEYS = {"minimum_battery_reserve_percent", "minimum_link_quality", "maximum_tasks_per_uav",
                  "distance_normalization", "continuity_required"}
WEIGHT_KEYS = {"capability", "battery", "distance", "link_quality", "workload", "target_continuity"}
PRIORITY_KEYS = {x.value for x in Priority}
TARGET_TASKS = {TaskType.INVESTIGATE_TARGET, TaskType.TRACK_TARGET}
REGION_TASKS = {TaskType.SEARCH_REGION, TaskType.OBSERVE_REGION, TaskType.RELAY_COMMUNICATIONS}


def _reject_constant(value: str) -> None:
    raise ScenarioError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScenarioError(f"duplicate JSON property is not allowed: {key}")
        result[key] = value
    return result


def _read_json(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_reject_duplicate_keys)
    except ScenarioError:
        raise
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"malformed {label} JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise ScenarioError(f"could not read {label}: {exc}") from exc


def _object(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ScenarioError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise ScenarioError(f"{path} must be an array")
    return value


def _keys(value: dict, allowed: set[str], required: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ScenarioError(f"{path} contains unknown properties: {', '.join(unknown)}")
    if missing:
        raise ScenarioError(f"{path} is missing required properties: {', '.join(missing)}")


def _string(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{path} must be a non-empty string" + (" or null" if nullable else ""))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{path} must be a real number, not {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ScenarioError(f"{path} must be finite")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioError(f"{path} must be an integer, not {type(value).__name__}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ScenarioError(f"{path} must be a boolean")
    return value


def _enum(enum_type, value: Any, path: str):
    if not isinstance(value, str):
        raise ScenarioError(f"{path} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(x.value for x in enum_type)
        raise ScenarioError(f"{path} has invalid value {value!r}; expected one of: {allowed}") from exc


def _enum_list(enum_type, value: Any, path: str) -> frozenset:
    values = _array(value, path)
    result = tuple(_enum(enum_type, item, f"{path}[{index}]") for index, item in enumerate(values))
    if len(result) != len(set(result)):
        raise ScenarioError(f"{path} must not contain duplicate values")
    return frozenset(result)


def _point(value: Any, path: str) -> Point:
    data = _object(value, path)
    _keys(data, POINT_KEYS, POINT_KEYS, path)
    return Point(_number(data["x"], f"{path}.x"), _number(data["y"], f"{path}.y"))


def load_policy(path: str | Path) -> dict:
    raw = _object(_read_json(path, "policy"), "policy")
    _keys(raw, POLICY_KEYS, POLICY_KEYS, "policy")
    policy_version = _string(raw["policy_version"], "policy.policy_version")
    thresholds = _object(raw["safety_thresholds"], "policy.safety_thresholds")
    weights = _object(raw["allocation_weights"], "policy.allocation_weights")
    multipliers = _object(raw["priority_multipliers"], "policy.priority_multipliers")
    _keys(thresholds, THRESHOLD_KEYS, THRESHOLD_KEYS, "policy.safety_thresholds")
    _keys(weights, WEIGHT_KEYS, WEIGHT_KEYS, "policy.allocation_weights")
    _keys(multipliers, PRIORITY_KEYS, PRIORITY_KEYS, "policy.priority_multipliers")

    battery = _number(thresholds["minimum_battery_reserve_percent"], "policy.safety_thresholds.minimum_battery_reserve_percent")
    link = _number(thresholds["minimum_link_quality"], "policy.safety_thresholds.minimum_link_quality")
    max_tasks = _integer(thresholds["maximum_tasks_per_uav"], "policy.safety_thresholds.maximum_tasks_per_uav")
    distance_norm = _number(thresholds["distance_normalization"], "policy.safety_thresholds.distance_normalization")
    continuity = _boolean(thresholds["continuity_required"], "policy.safety_thresholds.continuity_required")
    if not 0 <= battery < 100:
        raise ScenarioError("policy minimum battery reserve must be in [0, 100)")
    if not 0 <= link < 1:
        raise ScenarioError("policy minimum link quality must be in [0, 1)")
    if max_tasks < 1 or distance_norm <= 0:
        raise ScenarioError("policy maximum tasks and distance normalization must be positive")

    clean_weights = {key: _integer(weights[key], f"policy.allocation_weights.{key}") for key in sorted(WEIGHT_KEYS)}
    if any(value < 0 for value in clean_weights.values()) or sum(clean_weights.values()) != 100:
        raise ScenarioError("allocation weights must be nonnegative integers totaling 100")
    clean_multipliers = {key: _integer(multipliers[key], f"policy.priority_multipliers.{key}") for key in sorted(PRIORITY_KEYS)}
    if any(value <= 0 for value in clean_multipliers.values()):
        raise ScenarioError("priority multipliers must be positive integers")
    scale = _integer(raw["score_scale"], "policy.score_scale")
    if scale <= 0:
        raise ScenarioError("policy score scale must be a positive integer")
    return {"policy_version": policy_version,
            "safety_thresholds": {"minimum_battery_reserve_percent": battery, "minimum_link_quality": link,
                                  "maximum_tasks_per_uav": max_tasks, "distance_normalization": distance_norm,
                                  "continuity_required": continuity},
            "allocation_weights": clean_weights, "priority_multipliers": clean_multipliers, "score_scale": scale}


def load_scenario(path: str | Path) -> Scenario:
    raw = _object(_read_json(path, "scenario"), "scenario")
    required = {"schema_version", "scenario_id", "uavs", "targets", "regions", "mission_requests"}
    _keys(raw, SCENARIO_KEYS, required, "scenario")
    version = _string(raw["schema_version"], "scenario.schema_version")
    if version != "1.0":
        raise ScenarioError(f"unsupported scenario.schema_version {version!r}; expected '1.0'")
    scenario_id = _string(raw["scenario_id"], "scenario.scenario_id")

    uavs = []
    for index, item in enumerate(_array(raw["uavs"], "scenario.uavs")):
        path_name = f"scenario.uavs[{index}]"
        data = _object(item, path_name)
        required_uav = UAV_KEYS - {"max_task_distance", "current_target_id"}
        _keys(data, UAV_KEYS, required_uav, path_name)
        max_distance = None
        if data.get("max_task_distance") is not None:
            max_distance = _number(data["max_task_distance"], f"{path_name}.max_task_distance")
            if max_distance <= 0:
                raise ScenarioError(f"{path_name}.max_task_distance must be greater than zero")
        uavs.append(UAV(
            id=_string(data["id"], f"{path_name}.id"), position=_point(data["position"], f"{path_name}.position"),
            status=_enum(UAVStatus, data["status"], f"{path_name}.status"),
            capabilities=_enum_list(Capability, data["capabilities"], f"{path_name}.capabilities"),
            battery_percent=_number(data["battery_percent"], f"{path_name}.battery_percent"),
            link_quality=_number(data["link_quality"], f"{path_name}.link_quality"),
            current_workload=_integer(data["current_workload"], f"{path_name}.current_workload"),
            max_workload=_integer(data["max_workload"], f"{path_name}.max_workload"), max_task_distance=max_distance,
            current_target_id=_string(data.get("current_target_id"), f"{path_name}.current_target_id", nullable=True)))

    targets = []
    for index, item in enumerate(_array(raw["targets"], "scenario.targets")):
        path_name = f"scenario.targets[{index}]"
        data = _object(item, path_name)
        required_target = TARGET_KEYS - {"required_capabilities", "continuity_uav_id"}
        _keys(data, TARGET_KEYS, required_target, path_name)
        targets.append(Target(
            id=_string(data["id"], f"{path_name}.id"), position=_point(data["position"], f"{path_name}.position"),
            priority=_enum(Priority, data["priority"], f"{path_name}.priority"),
            status=_enum(TargetStatus, data["status"], f"{path_name}.status"),
            required_capabilities=_enum_list(Capability, data.get("required_capabilities", []), f"{path_name}.required_capabilities"),
            continuity_uav_id=_string(data.get("continuity_uav_id"), f"{path_name}.continuity_uav_id", nullable=True)))

    regions = []
    for index, item in enumerate(_array(raw["regions"], "scenario.regions")):
        path_name = f"scenario.regions[{index}]"
        data = _object(item, path_name)
        required_region = REGION_KEYS - {"required_capabilities"}
        _keys(data, REGION_KEYS, required_region, path_name)
        vertices = tuple(_point(point, f"{path_name}.vertices[{point_index}]")
                         for point_index, point in enumerate(_array(data["vertices"], f"{path_name}.vertices")))
        if len(vertices) < 3:
            raise ScenarioError(f"{path_name}.vertices must contain at least three points")
        try:
            center = polygon_centroid(vertices)
        except ValueError as exc:
            raise ScenarioError(f"{path_name} has invalid geometry: {exc}") from exc
        if not math.isfinite(center.x) or not math.isfinite(center.y):
            raise ScenarioError(f"{path_name} geometry produces a non-finite centroid")
        regions.append(Region(
            id=_string(data["id"], f"{path_name}.id"),
            region_type=_enum(RegionType, data["region_type"], f"{path_name}.region_type"), vertices=vertices,
            priority=_enum(Priority, data["priority"], f"{path_name}.priority"),
            required_capabilities=_enum_list(Capability, data.get("required_capabilities", []), f"{path_name}.required_capabilities")))

    requests = []
    for index, item in enumerate(_array(raw["mission_requests"], "scenario.mission_requests")):
        path_name = f"scenario.mission_requests[{index}]"
        data = _object(item, path_name)
        required_request = REQUEST_KEYS - {"required_capabilities", "target_id", "region_id"}
        _keys(data, REQUEST_KEYS, required_request, path_name)
        requests.append(MissionRequest(
            id=_string(data["id"], f"{path_name}.id"), task_type=_enum(TaskType, data["task_type"], f"{path_name}.task_type"),
            priority=_enum(Priority, data["priority"], f"{path_name}.priority"),
            required_capabilities=_enum_list(Capability, data.get("required_capabilities", []), f"{path_name}.required_capabilities"),
            target_id=_string(data.get("target_id"), f"{path_name}.target_id", nullable=True),
            region_id=_string(data.get("region_id"), f"{path_name}.region_id", nullable=True)))

    scenario = Scenario(version, scenario_id, tuple(uavs), tuple(targets), tuple(regions), tuple(requests),
                        _string(raw.get("operating_region_id"), "scenario.operating_region_id", nullable=True),
                        tuple(_string(value, f"scenario.exclusion_region_ids[{index}]")
                              for index, value in enumerate(_array(raw.get("exclusion_region_ids", []), "scenario.exclusion_region_ids"))))
    validate_scenario(scenario)
    return scenario


def validate_scenario(s: Scenario) -> None:
    if not s.uavs:
        raise ScenarioError("scenario.uavs must contain at least one UAV")
    for label, entities in (("UAV", s.uavs), ("target", s.targets), ("region", s.regions), ("request", s.requests)):
        ids = [item.id for item in entities]
        if len(ids) != len(set(ids)):
            raise ScenarioError(f"{label} identifiers must be unique")
    uav_ids, target_ids, region_ids = ({item.id for item in values} for values in (s.uavs, s.targets, s.regions))
    targets = {item.id: item for item in s.targets}
    regions = {item.id: item for item in s.regions}
    for uav in s.uavs:
        if not 0 <= uav.battery_percent <= 100 or not 0 <= uav.link_quality <= 1:
            raise ScenarioError(f"invalid battery or link quality for UAV {uav.id}")
        if uav.current_workload < 0 or uav.max_workload < 1 or uav.current_workload > uav.max_workload:
            raise ScenarioError(f"invalid workload for UAV {uav.id}")
        if uav.current_target_id is not None and uav.current_target_id not in target_ids:
            raise ScenarioError(f"UAV {uav.id} references unknown current target {uav.current_target_id}")
    for target in s.targets:
        if target.continuity_uav_id is not None and target.continuity_uav_id not in uav_ids:
            raise ScenarioError(f"target {target.id} references unknown continuity UAV {target.continuity_uav_id}")
    if s.operating_region_id:
        if s.operating_region_id not in region_ids:
            raise ScenarioError("scenario.operating_region_id references an unknown region")
        if regions[s.operating_region_id].region_type != RegionType.OPERATING_AREA:
            raise ScenarioError("scenario.operating_region_id must reference an OPERATING_AREA")
    for region_id in s.exclusion_region_ids:
        if region_id not in region_ids:
            raise ScenarioError(f"exclusion region {region_id} does not exist")
        if regions[region_id].region_type != RegionType.EXCLUSION_ZONE:
            raise ScenarioError(f"exclusion region {region_id} must have type EXCLUSION_ZONE")
    for request in s.requests:
        has_target = request.target_id is not None
        has_region = request.region_id is not None
        if request.task_type in TARGET_TASKS:
            if not has_target or has_region:
                raise ScenarioError(f"request {request.id} ({request.task_type.value}) requires exactly one target_id and no region_id")
            if request.target_id not in targets:
                raise ScenarioError(f"request {request.id} references unknown target {request.target_id}")
        elif request.task_type in REGION_TASKS:
            if not has_region or has_target:
                raise ScenarioError(f"request {request.id} ({request.task_type.value}) requires exactly one region_id and no target_id")
            if request.region_id not in regions:
                raise ScenarioError(f"request {request.id} references unknown region {request.region_id}")
        elif has_target or has_region:
            raise ScenarioError(f"request {request.id} does not permit target_id or region_id")
