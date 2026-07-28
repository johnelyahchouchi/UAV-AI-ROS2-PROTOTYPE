from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .domain import Task


def task_dict(task: Task) -> dict:
    return {"id": task.id, "request_id": task.request_id, "task_type": task.task_type.value,
            "priority": task.priority.value, "required_capabilities": sorted(task.required_capabilities),
            "location": {"x": task.location.x, "y": task.location.y}, "target_id": task.target_id,
            "region_id": task.region_id, "sequence": task.sequence}


def canonical_json(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def fingerprint(data: dict) -> str:
    clean = dict(data)
    clean.pop("deterministic_fingerprint", None)
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def write_plan(plan: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(plan), encoding="utf-8", newline="\n")

