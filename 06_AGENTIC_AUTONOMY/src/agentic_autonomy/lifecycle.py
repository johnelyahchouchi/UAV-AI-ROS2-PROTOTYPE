from dataclasses import replace

from .replanning_domain import MissionStateSnapshot, TaskLifecycleState


def active_scenario(snapshot: MissionStateSnapshot):
    states = {item.request_id: item.state for item in snapshot.task_lifecycle}
    active_requests = tuple(request for request in snapshot.scenario.requests
                            if states[request.id] == TaskLifecycleState.ACTIVE)
    return replace(snapshot.scenario, requests=active_requests)
