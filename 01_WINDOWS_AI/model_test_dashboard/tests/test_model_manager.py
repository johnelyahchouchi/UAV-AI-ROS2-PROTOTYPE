from __future__ import annotations

import os
from pathlib import Path

import pytest

from uav_model_dashboard.configuration import DeviceChoice
from uav_model_dashboard.errors import DashboardError
from uav_model_dashboard.model_manager import ModelManager


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, _: int) -> str:
        return "Test GPU"


class FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = FakeCuda(available)


class LoadedModel:
    task = "detect"
    predictor = None


def test_cache_uses_canonical_path_size_and_mtime(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"first")
    calls: list[str] = []

    def loader(path: str) -> LoadedModel:
        calls.append(path)
        return LoadedModel()

    manager = ModelManager(
        loader=loader,
        torch_module=FakeTorch(False),
        verifier=lambda _: "verified",
    )
    first = manager.get_model(model_path)
    second = manager.get_model(model_path)
    assert first.model is second.model
    assert first.loaded_now is True
    assert second.loaded_now is False
    assert len(calls) == 1

    model_path.write_bytes(b"changed-size")
    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third = manager.get_model(model_path)
    assert third.model is not first.model
    assert third.loaded_now is True
    assert len(calls) == 2


def test_non_detection_model_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "classifier.pt"
    model_path.write_bytes(b"x")
    classification_model = LoadedModel()
    classification_model.task = "classify"
    manager = ModelManager(
        loader=lambda _: classification_model,
        torch_module=FakeTorch(False),
        verifier=lambda _: "verified",
    )
    with pytest.raises(DashboardError) as raised:
        manager.get_model(model_path)
    assert raised.value.code == "MODEL_TASK_UNSUPPORTED"


def test_auto_and_explicit_device_resolution() -> None:
    gpu_manager = ModelManager(
        loader=lambda _: LoadedModel(),
        torch_module=FakeTorch(True),
        verifier=lambda _: "verified",
    )
    assert gpu_manager.resolve_device(DeviceChoice.AUTO).argument == 0
    assert gpu_manager.resolve_device(DeviceChoice.GPU_0).argument == 0
    assert gpu_manager.resolve_device(DeviceChoice.CPU).argument == "cpu"

    cpu_manager = ModelManager(
        loader=lambda _: LoadedModel(),
        torch_module=FakeTorch(False),
        verifier=lambda _: "verified",
    )
    assert cpu_manager.resolve_device(DeviceChoice.AUTO).argument == "cpu"
    with pytest.raises(DashboardError) as raised:
        cpu_manager.resolve_device(DeviceChoice.GPU_0)
    assert raised.value.code == "GPU_0_UNAVAILABLE"
