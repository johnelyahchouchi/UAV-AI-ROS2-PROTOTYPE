from __future__ import annotations

from agentic_autonomy.errors import ScenarioError
from agentic_autonomy.replanning_domain import MissionHistory
from agentic_autonomy.state_history import parse_mission_history

from .errors import AdapterSnapshotError


def validate_phase2_history(value: object) -> MissionHistory:
    """Validate adapter output through the existing Phase 2 runtime contract."""
    try:
        return parse_mission_history(value)
    except ScenarioError as exc:
        raise AdapterSnapshotError(f"generated Phase 2 history is invalid: {exc}") from exc
