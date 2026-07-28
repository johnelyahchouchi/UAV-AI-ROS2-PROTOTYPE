import json
from pathlib import Path

import pytest

from agentic_autonomy.errors import ScenarioError
from agentic_autonomy.scenario_loader import load_policy, load_scenario

ROOT = Path(__file__).parents[1]


def _scenario():
    return json.loads((ROOT / "scenarios/basic_reconnaissance.json").read_text(encoding="utf-8"))


def _policy():
    return json.loads((ROOT / "config/default_policy.json").read_text(encoding="utf-8"))


def _write(tmp_path, data, name="input.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_basic_scenario(basic):
    assert basic.scenario_id == "basic-reconnaissance"
    assert len(basic.uavs) == 2


def test_malformed_json_has_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ScenarioError, match="malformed scenario JSON at line"):
        load_scenario(path)


def test_missing_required_property_has_clear_error(tmp_path):
    source = _scenario()
    del source["scenario_id"]
    with pytest.raises(ScenarioError, match="missing required properties: scenario_id"):
        load_scenario(_write(tmp_path, source))


def test_invalid_enum_has_clear_error(tmp_path):
    source = _scenario()
    source["uavs"][0]["status"] = "FLYING"
    with pytest.raises(ScenarioError, match="invalid value 'FLYING'"):
        load_scenario(_write(tmp_path, source))


def test_duplicate_id_is_rejected(tmp_path):
    source = _scenario()
    source["uavs"].append(source["uavs"][0])
    with pytest.raises(ScenarioError, match="unique"):
        load_scenario(_write(tmp_path, source))


@pytest.mark.parametrize(("path", "value"), [
    (("uavs", 0, "battery_percent"), "82.0"),
    (("uavs", 0, "current_workload"), "0"),
])
def test_numeric_strings_are_rejected(tmp_path, path, value):
    source = _scenario()
    source[path[0]][path[1]][path[2]] = value
    with pytest.raises(ScenarioError, match="must be (a real number|an integer)"):
        load_scenario(_write(tmp_path, source))


@pytest.mark.parametrize("field", ["battery_percent", "current_workload"])
def test_booleans_used_as_numbers_are_rejected(tmp_path, field):
    source = _scenario()
    source["uavs"][0][field] = True
    with pytest.raises(ScenarioError, match="not bool"):
        load_scenario(_write(tmp_path, source))


def test_unknown_properties_are_rejected(tmp_path):
    source = _scenario()
    source["uavs"][0]["unexpected"] = 1
    with pytest.raises(ScenarioError, match="unknown properties: unexpected"):
        load_scenario(_write(tmp_path, source))


def test_wrong_schema_version_is_rejected(tmp_path):
    source = _scenario()
    source["schema_version"] = "2.0"
    with pytest.raises(ScenarioError, match="expected '1.0'"):
        load_scenario(_write(tmp_path, source))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_json_numbers_are_rejected(tmp_path, value):
    source = _scenario()
    source["targets"][0]["position"]["x"] = value
    with pytest.raises(ScenarioError, match="non-finite JSON number"):
        load_scenario(_write(tmp_path, source))


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_max_task_distance_is_rejected(tmp_path, value):
    source = _scenario()
    source["uavs"][0]["max_task_distance"] = value
    with pytest.raises(ScenarioError, match="greater than zero"):
        load_scenario(_write(tmp_path, source))


def test_task_with_both_reference_types_is_rejected(tmp_path):
    source = _scenario()
    source["mission_requests"][0]["target_id"] = "target-01"
    with pytest.raises(ScenarioError, match="exactly one region_id and no target_id"):
        load_scenario(_write(tmp_path, source))


def test_task_with_wrong_reference_type_is_rejected(tmp_path):
    source = _scenario()
    request = next(item for item in source["mission_requests"] if item["task_type"] == "TRACK_TARGET")
    request["target_id"] = None
    request["region_id"] = "search-sector-a"
    with pytest.raises(ScenarioError, match="exactly one target_id and no region_id"):
        load_scenario(_write(tmp_path, source))


def test_empty_uav_fleet_is_rejected(tmp_path):
    source = _scenario()
    source["uavs"] = []
    with pytest.raises(ScenarioError, match="at least one UAV"):
        load_scenario(_write(tmp_path, source))


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_policy_requires_exact_weight_keys(tmp_path, mutation):
    policy = _policy()
    if mutation == "missing":
        policy["allocation_weights"].pop("distance")
    else:
        policy["allocation_weights"]["altitude"] = 0
    with pytest.raises(ScenarioError, match="(missing required|unknown properties)"):
        load_policy(_write(tmp_path, policy, "policy.json"))


def test_policy_rejects_boolean_multiplier(tmp_path):
    policy = _policy()
    policy["priority_multipliers"]["HIGH"] = True
    with pytest.raises(ScenarioError, match="must be an integer, not bool"):
        load_policy(_write(tmp_path, policy, "policy.json"))
