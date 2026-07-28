import json
import re
from pathlib import Path

import pytest

from agentic_autonomy.planner import build_plan
from agentic_autonomy.scenario_loader import load_scenario
from agentic_autonomy.serialization import canonical_json, write_plan

ROOT = Path(__file__).parents[1]


def _schema_type_matches(value, expected):
    if expected == "object": return isinstance(value, dict)
    if expected == "array": return isinstance(value, list)
    if expected == "string": return isinstance(value, str)
    if expected == "boolean": return isinstance(value, bool)
    if expected == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null": return value is None
    raise AssertionError(f"unsupported schema type {expected}")


def _validate_schema(value, schema, root, path="$"):
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return _validate_schema(value, target, root, path)
    if "const" in schema:
        assert value == schema["const"], f"{path}: expected constant {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in enum"
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        assert any(_schema_type_matches(value, item) for item in types), f"{path}: wrong type"
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        assert required <= set(value), f"{path}: missing {sorted(required - set(value))}"
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], root, f"{path}.{key}")
            elif additional is False:
                raise AssertionError(f"{path}: unexpected property {key}")
            elif isinstance(additional, dict):
                _validate_schema(item, additional, root, f"{path}.{key}")
        assert len(value) >= schema.get("minProperties", 0), f"{path}: too few properties"
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0), f"{path}: too few items"
        if schema.get("uniqueItems"):
            assert len({json.dumps(item, sort_keys=True) for item in value}) == len(value), f"{path}: duplicate items"
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(value, str):
        assert len(value) >= schema.get("minLength", 0), f"{path}: string too short"
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), f"{path}: pattern mismatch"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema: assert value >= schema["minimum"], f"{path}: below minimum"
        if "maximum" in schema: assert value <= schema["maximum"], f"{path}: above maximum"


def test_canonical_output_round_trips(tmp_path, basic, policy):
    plan = build_plan(basic, policy)
    path = tmp_path / "plan.json"
    write_plan(plan, path)
    assert path.read_bytes().endswith(b"\n")
    assert json.loads(path.read_text()) == plan
    assert canonical_json(plan) == path.read_text()


@pytest.mark.parametrize("scenario_name", ["basic_reconnaissance", "infeasible_mission"])
def test_generated_plan_matches_strict_output_schema(policy, scenario_name):
    schema = json.loads((ROOT / "schemas/mission_plan.schema.json").read_text(encoding="utf-8"))
    scenario = load_scenario(ROOT / "scenarios" / f"{scenario_name}.json")
    _validate_schema(build_plan(scenario, policy), schema, schema)


def test_expected_basic_plan_matches_strict_output_schema():
    schema = json.loads((ROOT / "schemas/mission_plan.schema.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "scenarios/expected/basic_reconnaissance_plan.json").read_text(encoding="utf-8"))
    _validate_schema(expected, schema, schema)
