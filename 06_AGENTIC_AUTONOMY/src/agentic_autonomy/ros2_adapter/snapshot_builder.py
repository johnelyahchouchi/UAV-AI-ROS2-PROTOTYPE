from __future__ import annotations

from agentic_autonomy.domain import TargetStatus
from agentic_autonomy.errors import ScenarioError
from agentic_autonomy.scenario_loader import parse_scenario

from .errors import AdapterSnapshotError
from .event_domain import (
    AdapterAvailability,
    AdapterDiagnostic,
    DiagnosticSeverity,
    EventTimestamp,
    FreshnessState,
    LinkState,
    MissionStatus,
    SnapshotBuildResult,
)
from .freshness import freshness_state


class SnapshotBuilder:
    """Project richer adapter state into the strict Phase 2 snapshot contract."""

    def __init__(self, policy: dict):
        self.policy = policy

    def build(self, store, snapshot_time: EventTimestamp, event_sequence: int) -> SnapshotBuildResult:
        state = store.state
        if state.scenario_id is None or state.mission_id is None:
            raise AdapterSnapshotError("mission configuration is not available")
        diagnostics: list[AdapterDiagnostic] = []
        uavs = self._project_uavs(store, snapshot_time, event_sequence, diagnostics)
        if not uavs:
            raise AdapterSnapshotError(
                "no UAV has enough validated state to produce a Phase 2 snapshot"
            )
        projected_uav_ids = {item["id"] for item in uavs}
        targets = self._project_targets(
            store, snapshot_time, event_sequence, projected_uav_ids, diagnostics
        )
        projected_target_ids = {item["id"] for item in targets}
        for uav in uavs:
            current_target_id = uav["current_target_id"]
            if (
                current_target_id is not None
                and current_target_id not in projected_target_ids
            ):
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "CURRENT_TARGET_NOT_PROJECTED",
                        f"UAV {uav['id']} current target {current_target_id} is not projectable and was cleared.",
                        event_sequence,
                        "UAV",
                        uav["id"],
                    )
                )
                uav["current_target_id"] = None
        requests = []
        lifecycle = []
        for task in sorted(state.tasks.values(), key=lambda item: item.request_id):
            if task.target_id is not None and task.target_id not in projected_target_ids:
                raise AdapterSnapshotError(
                    f"task {task.request_id} references target {task.target_id}, which has no valid planning position"
                )
            if task.region_id is not None and task.region_id not in state.regions:
                raise AdapterSnapshotError(
                    f"task {task.request_id} references unavailable region {task.region_id}"
                )
            requests.append(
                {
                    "id": task.request_id,
                    "task_type": task.task_type,
                    "priority": task.priority,
                    "required_capabilities": sorted(task.required_capabilities),
                    "target_id": task.target_id,
                    "region_id": task.region_id,
                }
            )
            record = {"request_id": task.request_id, "state": task.lifecycle_state}
            if task.reason is not None:
                record["reason"] = task.reason
            lifecycle.append(record)

        scenario_dict = {
            "schema_version": "1.0",
            "scenario_id": state.scenario_id,
            "uavs": uavs,
            "targets": targets,
            "regions": [state.regions[key] for key in sorted(state.regions)],
            "mission_requests": requests,
            "operating_region_id": state.operating_region_id,
            "exclusion_region_ids": sorted(state.exclusion_region_ids),
        }
        try:
            parse_scenario(scenario_dict)
        except ScenarioError as exc:
            raise AdapterSnapshotError(
                f"generated Phase 1 scenario is invalid: {exc}"
            ) from exc
        next_sequence = getattr(store, "next_snapshot_sequence", 1)
        snapshot = {
            "snapshot_id": f"{state.scenario_id}-snapshot-{next_sequence:06d}",
            "sequence": next_sequence,
            "timestamp": snapshot_time.label,
            "scenario": scenario_dict,
            "task_lifecycle": lifecycle,
        }
        return SnapshotBuildResult(snapshot, tuple(diagnostics))

    def _project_uavs(
        self, store, snapshot_time, event_sequence, diagnostics
    ) -> list[dict]:
        output = []
        thresholds = self.policy["freshness_thresholds_seconds"]
        required = {
            "position": "position",
            "availability": "availability",
            "battery_percent": "battery",
            "link_state": "link",
            "link_quality": "link",
            "external_workload": "workload",
            "mission_status": "availability",
        }
        for uav in sorted(store.state.uavs.values(), key=lambda item: item.id):
            missing = [
                field
                for field in required
                if getattr(uav, field) is None
            ]
            if missing:
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "UAV_OMITTED_MISSING_STATE",
                        f"UAV {uav.id} is missing required dynamic fields: {', '.join(missing)}.",
                        event_sequence,
                        "UAV",
                        uav.id,
                    )
                )
                continue
            position = uav.position.value
            if position["frame_id"] != self.policy["planning_frame"]:
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.ERROR,
                        "UNSUPPORTED_COORDINATE_FRAME",
                        f"UAV {uav.id} position frame {position['frame_id']!r} does not match planning frame {self.policy['planning_frame']!r}.",
                        event_sequence,
                        "UAV",
                        uav.id,
                    )
                )
                continue
            stale_fields = []
            for field, threshold_key in required.items():
                status = freshness_state(
                    getattr(uav, field),
                    snapshot_time,
                    thresholds[threshold_key],
                )
                if status == FreshnessState.STALE:
                    stale_fields.append(field)
            availability = AdapterAvailability(uav.availability.value)
            link_state = LinkState(uav.link_state.value)
            mission_status = MissionStatus(uav.mission_status.value)
            eligible_status = (
                availability in {
                    AdapterAvailability.AVAILABLE,
                    AdapterAvailability.BUSY,
                }
                and link_state != LinkState.UNKNOWN
                and mission_status not in {
                    MissionStatus.RETURNING,
                    MissionStatus.LANDED,
                    MissionStatus.FAULT,
                    MissionStatus.UNKNOWN,
                }
                and not stale_fields
            )
            if not eligible_status:
                projected_status = "UNAVAILABLE"
            elif availability == AdapterAvailability.BUSY or mission_status == MissionStatus.EXECUTING:
                projected_status = "BUSY"
            else:
                projected_status = "AVAILABLE"
            if stale_fields:
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "UAV_PROJECTED_UNAVAILABLE_STALE_STATE",
                        f"UAV {uav.id} is projected UNAVAILABLE because fields are stale: {', '.join(stale_fields)}.",
                        event_sequence,
                        "UAV",
                        uav.id,
                    )
                )
            elif projected_status == "UNAVAILABLE":
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.INFO,
                        "UAV_PROJECTED_UNAVAILABLE",
                        f"UAV {uav.id} is conservatively projected UNAVAILABLE.",
                        event_sequence,
                        "UAV",
                        uav.id,
                    )
                )
            current_target_id = (
                uav.current_target_id.value if uav.current_target_id is not None else None
            )
            if (
                current_target_id is not None
                and current_target_id not in store.state.targets
            ):
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "UNKNOWN_CURRENT_TARGET_CLEARED",
                        f"UAV {uav.id} current target {current_target_id} is unknown and was not projected.",
                        event_sequence,
                        "UAV",
                        uav.id,
                    )
                )
                current_target_id = None
            output.append(
                {
                    "id": uav.id,
                    "position": {"x": position["x"], "y": position["y"]},
                    "status": projected_status,
                    "capabilities": sorted(uav.capabilities),
                    "battery_percent": uav.battery_percent.value,
                    "link_quality": uav.link_quality.value,
                    "current_workload": uav.external_workload.value,
                    "max_workload": uav.max_workload,
                    "max_task_distance": uav.max_task_distance,
                    "current_target_id": current_target_id,
                }
            )
        return output

    def _project_targets(
        self,
        store,
        snapshot_time,
        event_sequence,
        projected_uav_ids,
        diagnostics,
    ) -> list[dict]:
        output = []
        threshold = self.policy["freshness_thresholds_seconds"]["target"]
        referenced = {
            task.target_id
            for task in store.state.tasks.values()
            if task.target_id is not None
        }
        for target in sorted(store.state.targets.values(), key=lambda item: item.id):
            if target.position is None or target.priority is None:
                if target.id in referenced:
                    raise AdapterSnapshotError(
                        f"referenced target {target.id} is missing position or planning priority"
                    )
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "TARGET_OMITTED_INCOMPLETE",
                        f"Target {target.id} lacks a planning position or priority and was omitted.",
                        event_sequence,
                        "TARGET",
                        target.id,
                    )
                )
                continue
            position = target.position.value
            if position["frame_id"] != self.policy["planning_frame"]:
                if target.id in referenced:
                    raise AdapterSnapshotError(
                        f"referenced target {target.id} uses unsupported frame {position['frame_id']!r}"
                    )
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.ERROR,
                        "UNSUPPORTED_COORDINATE_FRAME",
                        f"Target {target.id} frame {position['frame_id']!r} does not match planning frame {self.policy['planning_frame']!r}.",
                        event_sequence,
                        "TARGET",
                        target.id,
                    )
                )
                continue
            target_freshness = freshness_state(
                target.status, snapshot_time, threshold
            )
            status = target.status.value
            if target_freshness == FreshnessState.STALE:
                status = TargetStatus.LOST.value
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "TARGET_PROJECTED_LOST_STALE_OBSERVATION",
                        f"Target {target.id} is stale and was projected LOST.",
                        event_sequence,
                        "TARGET",
                        target.id,
                    )
                )
            continuity = target.continuity_uav_id
            if continuity is not None and continuity not in projected_uav_ids:
                diagnostics.append(
                    store.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "TARGET_CONTINUITY_CLEARED",
                        f"Target {target.id} continuity UAV {continuity} is not projected.",
                        event_sequence,
                        "TARGET",
                        target.id,
                    )
                )
                continuity = None
            output.append(
                {
                    "id": target.id,
                    "position": {"x": position["x"], "y": position["y"]},
                    "priority": target.priority.value,
                    "status": status,
                    "required_capabilities": sorted(target.required_capabilities),
                    "continuity_uav_id": continuity,
                }
            )
        return output
