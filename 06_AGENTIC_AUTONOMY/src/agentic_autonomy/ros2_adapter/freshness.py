from __future__ import annotations

from .errors import AdapterSnapshotError
from .event_domain import EventTimestamp, FreshnessState, TimestampedValue


def freshness_state(
    value: TimestampedValue | None,
    snapshot_time: EventTimestamp,
    threshold_seconds: float,
) -> FreshnessState:
    """Classify one field without reading a wall clock."""
    if value is None:
        return FreshnessState.MISSING
    if value.observed_at.clock_id != snapshot_time.clock_id:
        raise AdapterSnapshotError(
            f"cannot compare clock {value.observed_at.clock_id!r} with {snapshot_time.clock_id!r}"
        )
    age = snapshot_time.total_nanoseconds - value.observed_at.total_nanoseconds
    if age < 0:
        raise AdapterSnapshotError("snapshot timestamp precedes stored state")
    if age > int(threshold_seconds * 1_000_000_000):
        return FreshnessState.STALE
    return FreshnessState.FRESH
