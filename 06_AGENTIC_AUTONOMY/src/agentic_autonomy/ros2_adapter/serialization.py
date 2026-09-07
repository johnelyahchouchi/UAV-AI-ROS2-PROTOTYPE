from __future__ import annotations

from pathlib import Path

from agentic_autonomy.serialization import canonical_json

from .event_domain import AdapterDiagnostic, NormalizedEvent


def event_to_dict(event: NormalizedEvent, *, include_sequence: bool = True) -> dict:
    data = {
        "schema_version": event.schema_version,
        "mission_id": event.mission_id,
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "observed_at": {
            "clock_id": event.observed_at.clock_id,
            "sec": event.observed_at.sec,
            "nanosec": event.observed_at.nanosec,
        },
        "source": {
            "source_id": event.source.source_id,
            "source_node": event.source.source_node,
            "topic": event.source.topic,
            "message_type": event.source.message_type,
            "source_uav_id": event.source.source_uav_id,
            "source_session_id": event.source.source_session_id,
            "source_timestamp": event.source.source_timestamp,
            "upstream_sequence": event.source.upstream_sequence,
        },
        "payload": event.payload,
    }
    if include_sequence:
        data["sequence"] = event.sequence
    return data


def diagnostic_to_dict(item: AdapterDiagnostic) -> dict:
    return {
        "id": item.id,
        "severity": item.severity.value,
        "code": item.code,
        "message": item.message,
        "event_sequence": item.event_sequence,
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
    }


def write_canonical_json(value: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(value), encoding="utf-8", newline="\n")
