import json
from copy import deepcopy
from pathlib import Path

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.state_history import load_mission_history
from test_serialization import _validate_schema

ROOT = Path(__file__).parents[1]


def _allow_embedded_phase1_plan(schema):
    if isinstance(schema, dict):
        if schema.get("$ref") == "mission_plan.schema.json":
            return {}
        return {key: _allow_embedded_phase1_plan(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return [_allow_embedded_phase1_plan(value) for value in schema]
    return schema


def test_replanning_output_matches_strict_schema(policy):
    schema = json.loads((ROOT / "schemas/replanning_result.schema.json").read_text(encoding="utf-8"))
    schema = _allow_embedded_phase1_plan(schema)
    result = replan_history(load_mission_history(ROOT / "scenarios/replanning/battery_degradation.json"), policy)
    _validate_schema(result, schema, schema)

