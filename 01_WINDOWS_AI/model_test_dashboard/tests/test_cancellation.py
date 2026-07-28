from __future__ import annotations

import pytest

from uav_model_dashboard.errors import DashboardError, ProcessingCancelled
from uav_model_dashboard.processing_control import ProcessingController


def test_controller_cancels_active_job_and_allows_next_job() -> None:
    controller = ProcessingController()
    token = controller.begin()
    assert controller.is_running is True
    assert controller.request_cancel() is True
    with pytest.raises(ProcessingCancelled):
        token.raise_if_cancelled()
    controller.finish(token)
    assert controller.is_running is False
    next_token = controller.begin()
    controller.finish(next_token)


def test_controller_rejects_parallel_job() -> None:
    controller = ProcessingController()
    token = controller.begin()
    with pytest.raises(DashboardError) as raised:
        controller.begin()
    assert raised.value.code == "PROCESSING_ALREADY_RUNNING"
    controller.finish(token)
