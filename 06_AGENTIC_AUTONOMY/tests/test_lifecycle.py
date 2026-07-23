from pathlib import Path

from agentic_autonomy.lifecycle import active_scenario
from agentic_autonomy.state_history import load_mission_history

ROOT = Path(__file__).parents[1]


def test_completed_task_is_removed_from_active_scenario():
    history = load_mission_history(ROOT / "scenarios/replanning/task_completion.json")
    assert len(active_scenario(history.snapshots[0]).requests) == 1
    assert len(active_scenario(history.snapshots[1]).requests) == 0

