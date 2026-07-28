from pathlib import Path

import pytest

from agentic_autonomy.scenario_loader import load_policy, load_scenario

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return load_policy(ROOT / "config" / "default_policy.json")


@pytest.fixture
def basic():
    return load_scenario(ROOT / "scenarios" / "basic_reconnaissance.json")

