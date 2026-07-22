from agentic_autonomy.task_generator import generate_tasks


def test_priority_order_and_ids_are_deterministic(basic):
    tasks = generate_tasks(basic)
    assert [x.id for x in tasks] == ["task-001-request-track", "task-002-request-search"]
    assert tasks[0].priority.value == "HIGH"
    assert sorted(tasks[0].required_capabilities) == ["TARGET_TRACKING"]

