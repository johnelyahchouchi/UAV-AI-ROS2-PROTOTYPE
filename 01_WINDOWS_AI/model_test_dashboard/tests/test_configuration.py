from __future__ import annotations

from pathlib import Path

import pytest

from uav_model_dashboard.configuration import (
    DeviceChoice,
    InferenceMode,
    ProcessingSettings,
    default_model_path,
    model_location_warning,
    repository_root,
    validate_model_path,
    validate_video_path,
)
from uav_model_dashboard.errors import DashboardError


def test_default_model_path_uses_environment_override(tmp_path: Path) -> None:
    configured = tmp_path / "models" / "custom.pt"
    result = default_model_path({"UAV_MODEL_PATH": str(configured)})
    assert result == configured.resolve(strict=False)


def test_default_model_path_is_repository_anchored() -> None:
    result = default_model_path({})
    expected = (
        repository_root()
        / "03_MODELS"
        / "active"
        / "detector"
        / "military_kaggle_v1.pt"
    )
    assert result == expected.resolve(strict=False)


def test_processing_settings_defaults_and_valid_values() -> None:
    settings = ProcessingSettings.from_values(
        0.50,
        0.45,
        960,
        DeviceChoice.GPU_0.value,
        InferenceMode.BOTSORT.value,
    )
    assert settings.confidence == 0.50
    assert settings.iou == 0.45
    assert settings.image_size == 960
    assert settings.device is DeviceChoice.GPU_0
    assert settings.mode is InferenceMode.BOTSORT


@pytest.mark.parametrize(
    ("field_values", "code"),
    [
        ((True, 0.45, 640), "INVALID_SETTING"),
        ((0.50, False, 640), "INVALID_SETTING"),
        ((0.0, 0.45, 640), "INVALID_CONFIDENCE"),
        ((0.50, 1.1, 640), "INVALID_IOU"),
        ((0.50, 0.45, True), "INVALID_IMAGE_SIZE"),
        ((0.50, 0.45, 641), "INVALID_IMAGE_SIZE"),
    ],
)
def test_processing_settings_reject_invalid_values(
    field_values: tuple[object, object, object],
    code: str,
) -> None:
    with pytest.raises(DashboardError) as raised:
        ProcessingSettings.from_values(
            *field_values,
            DeviceChoice.AUTO.value,
            InferenceMode.DETECTION.value,
        )
    assert raised.value.code == code


def test_path_validation_and_repository_model_warning(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")

    assert validate_video_path(video) == video
    assert validate_model_path(model) == model
    warning = model_location_warning(model, repo_root=tmp_path)
    assert warning is not None
    assert "must not be committed" in warning


def test_model_validation_requires_pt_file(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    with pytest.raises(DashboardError, match=r"\.pt"):
        validate_model_path(model)
