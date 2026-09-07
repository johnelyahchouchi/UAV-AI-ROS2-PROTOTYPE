from __future__ import annotations

from threading import Lock

from agentic_autonomy.replanning_domain import TaskLifecycleState
from agentic_autonomy.serialization import canonical_json

from . import ADAPTER_VERSION
from .errors import AdapterLimitError, AdapterSnapshotError
from .event_domain import (
    AdapterEventType,
    DiagnosticSeverity,
    LinkState,
    NormalizedEvent,
)
from .serialization import diagnostic_to_dict
from .snapshot_builder import SnapshotBuilder
from .state_store import MissionStateStore
from .validation import validate_phase2_history


class MissionStateAdapter:
    """Serialized event-to-snapshot pipeline independent of ROS 2."""

    def __init__(self, adapter_policy: dict, planner_policy: dict):
        self.adapter_policy = adapter_policy
        self.planner_policy = planner_policy
        self.store = MissionStateStore(adapter_policy)
        self.builder = SnapshotBuilder(adapter_policy)
        self.snapshots: list[dict] = []
        self._dirty = False
        self._last_snapshot_time = None
        self._lock = Lock()
        self.store.next_snapshot_sequence = 1

    def process_event(self, event: NormalizedEvent) -> dict | None:
        """Apply one event atomically and optionally emit a complete snapshot."""
        with self._lock:
            result = self.store.apply(event)
            if result.state_changed:
                self._dirty = True
            if not self._should_snapshot(event):
                return None
            try:
                return self._emit_snapshot(event)
            except AdapterSnapshotError as exc:
                self.store.add_diagnostic(
                    DiagnosticSeverity.WARNING,
                    "SNAPSHOT_NOT_READY",
                    str(exc),
                    event.sequence,
                    "MISSION",
                    self.store.state.mission_id,
                )
                return None

    def process_events(self, events) -> dict:
        for event in events:
            self.process_event(event)
        if not self.snapshots:
            raise AdapterSnapshotError(
                "event stream completed without producing a valid mission-state snapshot"
            )
        return self.history_document()

    def history_document(self) -> dict:
        mission_id = self.store.state.mission_id
        if mission_id is None:
            raise AdapterSnapshotError("mission is not configured")
        return {
            "schema_version": "2.0",
            "mission_id": mission_id,
            "snapshots": list(self.snapshots),
        }

    def diagnostics_document(self) -> dict:
        return {
            "schema_version": "1.0",
            "adapter_version": ADAPTER_VERSION,
            "mission_id": self.store.state.mission_id,
            "diagnostics": [
                diagnostic_to_dict(item) for item in self.store.diagnostics
            ],
        }

    def _should_snapshot(self, event: NormalizedEvent) -> bool:
        if event.event_type == AdapterEventType.SNAPSHOT_REQUESTED:
            return True
        if event.event_type == AdapterEventType.SNAPSHOT_TICK:
            if self._dirty:
                return True
            if self._last_snapshot_time is None:
                return True
            elapsed = (
                event.observed_at.total_nanoseconds
                - self._last_snapshot_time.total_nanoseconds
            )
            heartbeat = int(
                self.adapter_policy["snapshot_trigger"][
                    "unchanged_heartbeat_seconds"
                ]
                * 1_000_000_000
            )
            return elapsed >= heartbeat
        if not self.adapter_policy["snapshot_trigger"]["emit_on_safety_change"]:
            return False
        if event.event_type == AdapterEventType.TASK_LIFECYCLE_CHANGED:
            return event.payload["state"] in {
                TaskLifecycleState.COMPLETED.value,
                TaskLifecycleState.CANCELLED.value,
            }
        if event.event_type == AdapterEventType.UAV_STATE_UPDATED:
            if event.payload.get("availability") == "UNAVAILABLE":
                return True
            if event.payload.get("link_state") == LinkState.LOST.value:
                return True
            battery = event.payload.get("battery_percent")
            if battery is not None:
                reserve = self.planner_policy["safety_thresholds"][
                    "minimum_battery_reserve_percent"
                ]
                return battery < reserve
        return False

    def _emit_snapshot(self, event: NormalizedEvent) -> dict:
        maximum_snapshots = self.adapter_policy["limits"]["maximum_snapshots"]
        if len(self.snapshots) >= maximum_snapshots:
            raise AdapterLimitError(
                f"maximum_snapshots limit {maximum_snapshots} would be exceeded"
            )
        self.store.next_snapshot_sequence = len(self.snapshots) + 1
        result = self.builder.build(
            self.store, event.observed_at, event.sequence
        )
        candidate = {
            "schema_version": "2.0",
            "mission_id": self.store.state.mission_id,
            "snapshots": [*self.snapshots, result.snapshot],
        }
        encoded_size = len(canonical_json(candidate).encode("utf-8"))
        maximum_bytes = self.adapter_policy["limits"]["maximum_message_bytes"]
        if encoded_size > maximum_bytes:
            raise AdapterLimitError(
                f"mission history would be {encoded_size} bytes; maximum_message_bytes is {maximum_bytes}"
            )
        validate_phase2_history(candidate)
        self.snapshots.append(result.snapshot)
        self._last_snapshot_time = event.observed_at
        self._dirty = False
        return result.snapshot
