from agentic_autonomy.planner import build_plan
from agentic_autonomy.serialization import canonical_json


def test_repeated_plans_are_byte_identical(basic, policy):
    outputs = [canonical_json(build_plan(basic, policy)) for _ in range(10)]
    assert len(set(outputs)) == 1

