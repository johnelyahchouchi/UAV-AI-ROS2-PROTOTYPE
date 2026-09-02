from __future__ import annotations

import hashlib

from agentic_autonomy.replanning_domain import TaskLifecycleState
from agentic_autonomy.serialization import canonical_json

from .errors import AdapterEventError, AdapterLimitError
from .event_domain import (
    AdapterDiagnostic,
    AdapterEventType,
    AdapterMissionState,
    DiagnosticSeverity,
    EventApplicationResult,
    NormalizedEvent,
    TargetAdapterState,
    TaskAdapterState,
    TimestampedValue,
    UAVAdapterState,
)
from .serialization import event_to_dict


class MissionStateStore:
    """Apply normalized events in one deterministic total order."""

    def __init__(self, policy: dict):
        self.policy = policy
        self.state = AdapterMissionState()
        self.last_sequence = -1
        self._event_signatures: dict[str, str] = {}
        self.diagnostics: list[AdapterDiagnostic] = []
        self._diagnostic_counter = 0

    def apply(self, event: NormalizedEvent) -> EventApplicationResult:
        if event.sequence <= self.last_sequence:
            raise AdapterEventError(
                f"event sequence {event.sequence} is not greater than last applied sequence {self.last_sequence}"
            )
        signature = hashlib.sha256(
            canonical_json(event_to_dict(event, include_sequence=False)).encode("utf-8")
        ).hexdigest()
        previous_signature = self._event_signatures.get(event.event_id)
        if previous_signature is not None:
            if previous_signature != signature:
                raise AdapterEventError(
                    f"duplicate event_id {event.event_id!r} has conflicting content"
                )
            diagnostic = self.add_diagnostic(
                DiagnosticSeverity.INFO,
                "DUPLICATE_EVENT_IGNORED",
                f"Duplicate event {event.event_id} was ignored idempotently.",
                event.sequence,
            )
            self.last_sequence = event.sequence
            return EventApplicationResult(True, False, True, (diagnostic,))

        self._validate_envelope(event)
        diagnostics: list[AdapterDiagnostic] = []
        changed = self._apply_new_event(event, diagnostics)
        self.last_sequence = event.sequence
        self._event_signatures[event.event_id] = signature
        return EventApplicationResult(True, changed, False, tuple(diagnostics))

    def add_diagnostic(
        self,
        severity: DiagnosticSeverity,
        code: str,
        message: str,
        event_sequence: int | None,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> AdapterDiagnostic:
        maximum = self.policy["limits"]["maximum_diagnostics"]
        if len(self.diagnostics) >= maximum:
            raise AdapterLimitError(
                f"maximum_diagnostics limit {maximum} would be exceeded"
            )
        self._diagnostic_counter += 1
        item = AdapterDiagnostic(
            id=f"diagnostic-{self._diagnostic_counter:06d}",
            severity=severity,
            code=code,
            message=message,
            event_sequence=event_sequence,
            entity_type=entity_type,
            entity_id=entity_id,
        )
        self.diagnostics.append(item)
        return item

    def _validate_envelope(self, event: NormalizedEvent) -> None:
        state = self.state
        if state.mission_id is not None and event.mission_id != state.mission_id:
            raise AdapterEventError(
                f"event mission_id {event.mission_id!r} does not match {state.mission_id!r}"
            )
        if state.clock_id is not None and event.observed_at.clock_id != state.clock_id:
            raise AdapterEventError(
                f"event clock_id {event.observed_at.clock_id!r} does not match {state.clock_id!r}"
            )
        if (
            state.scenario_id is None
            and event.event_type != AdapterEventType.MISSION_CONFIGURED
        ):
            raise AdapterEventError("MISSION_CONFIGURED must be the first accepted event")

    def _apply_new_event(
        self, event: NormalizedEvent, diagnostics: list[AdapterDiagnostic]
    ) -> bool:
        handlers = {
            AdapterEventType.MISSION_CONFIGURED: self._mission_configured,
            AdapterEventType.REGION_UPDATED: self._region_updated,
            AdapterEventType.UAV_CONFIGURED: self._uav_configured,
            AdapterEventType.UAV_STATE_UPDATED: self._uav_state_updated,
            AdapterEventType.TARGET_OBSERVED: self._target_observed,
            AdapterEventType.TARGET_STATE_UPDATED: self._target_state_updated,
            AdapterEventType.TASK_CREATED: self._task_created,
            AdapterEventType.TASK_UPDATED: self._task_updated,
            AdapterEventType.TASK_LIFECYCLE_CHANGED: self._task_lifecycle_changed,
            AdapterEventType.SNAPSHOT_TICK: lambda *_: False,
            AdapterEventType.SNAPSHOT_REQUESTED: lambda *_: False,
        }
        return handlers[event.event_type](event, diagnostics)

    def _mission_configured(self, event, diagnostics) -> bool:
        if self.state.scenario_id is not None:
            raise AdapterEventError("MISSION_CONFIGURED may only be accepted once")
        payload = event.payload
        self.state.mission_id = event.mission_id
        self.state.scenario_id = payload["scenario_id"]
        self.state.clock_id = event.observed_at.clock_id
        self.state.regions = {item["id"]: item for item in payload["regions"]}
        self.state.operating_region_id = payload["operating_region_id"]
        self.state.exclusion_region_ids = tuple(payload["exclusion_region_ids"])
        return True

    def _region_updated(self, event, diagnostics) -> bool:
        region = event.payload["region"]
        self.state.regions[region["id"]] = region
        return True

    def _uav_configured(self, event, diagnostics) -> bool:
        payload = event.payload
        uav_id = payload["uav_id"]
        if uav_id in self.state.uavs:
            raise AdapterEventError(f"UAV {uav_id} is already configured")
        if len(self.state.uavs) >= self.policy["limits"]["maximum_uavs"]:
            raise AdapterLimitError("maximum_uavs limit would be exceeded")
        self.state.uavs[uav_id] = UAVAdapterState(
            id=uav_id,
            capabilities=frozenset(payload["capabilities"]),
            max_workload=payload["max_workload"],
            max_task_distance=payload["max_task_distance"],
        )
        return True

    def _uav_state_updated(self, event, diagnostics) -> bool:
        payload = event.payload
        uav_id = payload["uav_id"]
        uav = self.state.uavs.get(uav_id)
        if uav is None:
            raise AdapterEventError(f"UAV_STATE_UPDATED references unknown UAV {uav_id}")
        if (
            event.source.source_uav_id is not None
            and event.source.source_uav_id != uav_id
        ):
            raise AdapterEventError(
                "UAV_STATE_UPDATED uav_id conflicts with event source metadata"
            )
        field_map = {
            "position": "position",
            "availability": "availability",
            "battery_percent": "battery_percent",
            "link_state": "link_state",
            "link_quality": "link_quality",
            "external_workload": "external_workload",
            "mission_status": "mission_status",
            "current_target_id": "current_target_id",
        }
        for payload_key, attribute in field_map.items():
            if payload_key in payload:
                self._ensure_timestamp_is_current(
                    getattr(uav, attribute), event, "UAV", uav_id, payload_key
                )
        if (
            "external_workload" in payload
            and payload["external_workload"] > uav.max_workload
        ):
            raise AdapterEventError(
                f"UAV {uav_id} external_workload exceeds max_workload"
            )
        normalized_battery = None
        if "battery_percent" in payload:
            normalized_battery = self._normalize_battery(
                payload["battery_percent"], event, diagnostics, uav_id
            )

        for payload_key, attribute in field_map.items():
            if payload_key not in payload:
                continue
            value = payload[payload_key]
            if payload_key == "battery_percent":
                value = normalized_battery
            setattr(uav, attribute, TimestampedValue(value, event.observed_at))
        return True

    def _normalize_battery(self, value, event, diagnostics, uav_id) -> float:
        tolerance = self.policy["battery"]["clamp_tolerance_percent"]
        if 0 <= value <= 100:
            return value
        if -tolerance <= value < 0:
            clamped = 0.0
        elif 100 < value <= 100 + tolerance:
            clamped = 100.0
        else:
            raise AdapterEventError(
                f"UAV {uav_id} battery_percent must be in [0, 100]"
            )
        diagnostics.append(
            self.add_diagnostic(
                DiagnosticSeverity.WARNING,
                "BATTERY_CLAMPED_WITHIN_TOLERANCE",
                f"UAV {uav_id} battery_percent {value} was clamped to {clamped}.",
                event.sequence,
                "UAV",
                uav_id,
            )
        )
        return clamped

    def _target_observed(self, event, diagnostics) -> bool:
        payload = event.payload
        source_uav_id = payload["source_uav_id"]
        if source_uav_id not in self.state.uavs:
            raise AdapterEventError(
                f"TARGET_OBSERVED references unknown source UAV {source_uav_id}"
            )
        if (
            event.source.source_uav_id is not None
            and event.source.source_uav_id != source_uav_id
        ):
            raise AdapterEventError(
                "TARGET_OBSERVED source_uav_id conflicts with event source metadata"
            )
        target_id = payload["global_target_id"]
        if target_id is None:
            session_id = event.source.source_session_id
            local_track_id = payload["local_track_id"]
            if session_id is None or local_track_id is None:
                diagnostics.append(
                    self.add_diagnostic(
                        DiagnosticSeverity.WARNING,
                        "TARGET_IDENTITY_INSUFFICIENT",
                        "Legacy detection lacks a global target ID or source session/local track identity; it cannot become a planner target.",
                        event.sequence,
                        "TARGET_OBSERVATION",
                        None,
                    )
                )
                return False
            target_id = f"target:{source_uav_id}:{session_id}:{local_track_id}"

        current = self.state.targets.get(target_id)
        if current is None:
            if len(self.state.targets) >= self.policy["limits"]["maximum_targets"]:
                raise AdapterLimitError("maximum_targets limit would be exceeded")
            timestamp = event.observed_at
            self.state.targets[target_id] = TargetAdapterState(
                id=target_id,
                source_uav_id=source_uav_id,
                source_session_id=event.source.source_session_id,
                local_track_id=payload["local_track_id"],
                class_name=TimestampedValue(payload["class_name"], timestamp),
                confidence=TimestampedValue(payload["confidence"], timestamp),
                status=TimestampedValue(payload["status"], timestamp),
                position=(
                    TimestampedValue(payload["position"], timestamp)
                    if payload["position"] is not None
                    else None
                ),
                priority=(
                    TimestampedValue(payload["priority"], timestamp)
                    if payload["priority"] is not None
                    else None
                ),
                required_capabilities=frozenset(payload["required_capabilities"]),
                continuity_uav_id=payload["continuity_uav_id"],
                first_observed_at=timestamp,
                last_observed_at=timestamp,
            )
            return True
        if event.observed_at.total_nanoseconds < current.last_observed_at.total_nanoseconds:
            raise AdapterEventError(
                f"TARGET_OBSERVED for {target_id} has an out-of-order timestamp"
            )
        timestamp = event.observed_at
        current.class_name = TimestampedValue(payload["class_name"], timestamp)
        current.confidence = TimestampedValue(payload["confidence"], timestamp)
        current.status = TimestampedValue(payload["status"], timestamp)
        if payload["position"] is not None:
            current.position = TimestampedValue(payload["position"], timestamp)
        if payload["priority"] is not None:
            current.priority = TimestampedValue(payload["priority"], timestamp)
        current.required_capabilities = frozenset(payload["required_capabilities"])
        current.continuity_uav_id = payload["continuity_uav_id"]
        current.last_observed_at = timestamp
        return True

    def _target_state_updated(self, event, diagnostics) -> bool:
        payload = event.payload
        target_id = payload["target_id"]
        target = self.state.targets.get(target_id)
        if target is None:
            raise AdapterEventError(
                f"TARGET_STATE_UPDATED references unknown target {target_id}"
            )
        if event.observed_at.total_nanoseconds < target.last_observed_at.total_nanoseconds:
            raise AdapterEventError(
                f"TARGET_STATE_UPDATED for {target_id} has an out-of-order timestamp"
            )
        timestamp = event.observed_at
        target.status = TimestampedValue(payload["status"], timestamp)
        if "position" in payload:
            target.position = TimestampedValue(payload["position"], timestamp)
        if "priority" in payload:
            target.priority = TimestampedValue(payload["priority"], timestamp)
        if "required_capabilities" in payload:
            target.required_capabilities = frozenset(payload["required_capabilities"])
        if "continuity_uav_id" in payload:
            target.continuity_uav_id = payload["continuity_uav_id"]
        target.last_observed_at = timestamp
        return True

    def _task_created(self, event, diagnostics) -> bool:
        payload = event.payload
        request_id = payload["request_id"]
        if request_id in self.state.tasks:
            raise AdapterEventError(f"task {request_id} already exists")
        if len(self.state.tasks) >= self.policy["limits"]["maximum_tasks"]:
            raise AdapterLimitError("maximum_tasks limit would be exceeded")
        if payload["target_id"] is not None and payload["target_id"] not in self.state.targets:
            raise AdapterEventError(
                f"task {request_id} references unknown target {payload['target_id']}"
            )
        if payload["region_id"] is not None and payload["region_id"] not in self.state.regions:
            raise AdapterEventError(
                f"task {request_id} references unknown region {payload['region_id']}"
            )
        self.state.tasks[request_id] = TaskAdapterState(
            request_id=request_id,
            task_type=payload["task_type"],
            priority=payload["priority"],
            required_capabilities=frozenset(payload["required_capabilities"]),
            target_id=payload["target_id"],
            region_id=payload["region_id"],
            lifecycle_state=payload["lifecycle_state"],
            reason=payload["reason"],
            created_at=event.observed_at,
            updated_at=event.observed_at,
        )
        return True

    def _task_updated(self, event, diagnostics) -> bool:
        payload = event.payload
        task = self.state.tasks.get(payload["request_id"])
        if task is None:
            raise AdapterEventError(
                f"TASK_UPDATED references unknown task {payload['request_id']}"
            )
        if task.lifecycle_state in {
            TaskLifecycleState.COMPLETED.value,
            TaskLifecycleState.CANCELLED.value,
        }:
            raise AdapterEventError(f"terminal task {task.request_id} cannot be updated")
        self._ensure_task_timestamp(task, event)
        if "priority" in payload:
            task.priority = payload["priority"]
        if "required_capabilities" in payload:
            task.required_capabilities = frozenset(payload["required_capabilities"])
        task.updated_at = event.observed_at
        return True

    def _task_lifecycle_changed(self, event, diagnostics) -> bool:
        payload = event.payload
        task = self.state.tasks.get(payload["request_id"])
        if task is None:
            raise AdapterEventError(
                f"TASK_LIFECYCLE_CHANGED references unknown task {payload['request_id']}"
            )
        self._ensure_task_timestamp(task, event)
        previous = TaskLifecycleState(task.lifecycle_state)
        current = TaskLifecycleState(payload["state"])
        previous_reason = task.reason
        valid = {
            TaskLifecycleState.PENDING: {
                TaskLifecycleState.PENDING,
                TaskLifecycleState.ACTIVE,
                TaskLifecycleState.CANCELLED,
            },
            TaskLifecycleState.ACTIVE: {
                TaskLifecycleState.ACTIVE,
                TaskLifecycleState.COMPLETED,
                TaskLifecycleState.CANCELLED,
            },
            TaskLifecycleState.COMPLETED: {TaskLifecycleState.COMPLETED},
            TaskLifecycleState.CANCELLED: {TaskLifecycleState.CANCELLED},
        }
        if current not in valid[previous]:
            raise AdapterEventError(
                f"invalid lifecycle transition for {task.request_id}: {previous.value} -> {current.value}"
            )
        task.lifecycle_state = current.value
        task.reason = payload["reason"]
        task.updated_at = event.observed_at
        return current != previous or previous_reason != task.reason

    @staticmethod
    def _ensure_timestamp_is_current(current, event, entity_type, entity_id, field) -> None:
        if (
            current is not None
            and event.observed_at.total_nanoseconds
            < current.observed_at.total_nanoseconds
        ):
            raise AdapterEventError(
                f"{entity_type} {entity_id} field {field} has an out-of-order timestamp"
            )

    @staticmethod
    def _ensure_task_timestamp(task, event) -> None:
        if event.observed_at.total_nanoseconds < task.updated_at.total_nanoseconds:
            raise AdapterEventError(
                f"task {task.request_id} has an out-of-order timestamp"
            )
