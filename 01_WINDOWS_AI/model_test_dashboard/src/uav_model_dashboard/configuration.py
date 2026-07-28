"""Configuration resolution and validation for dashboard processing jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
from typing import Mapping

from .errors import DashboardError


DEFAULT_CONFIDENCE = 0.50
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640
SUPPORTED_IMAGE_SIZES = (320, 480, 640, 960, 1280)
DISPLAY_TABLE_LIMIT = 10_000


class DeviceChoice(str, Enum):
    """User-selectable inference device."""

    AUTO = "Auto"
    GPU_0 = "GPU 0"
    CPU = "CPU"


class InferenceMode(str, Enum):
    """Supported inference behaviors."""

    DETECTION = "Detection only"
    BOTSORT = "BoT-SORT tracking"


def dashboard_root() -> Path:
    """Return the isolated dashboard subsystem root."""
    return Path(__file__).resolve().parents[2]


def repository_root() -> Path:
    """Return the repository root derived from this module location."""
    return Path(__file__).resolve().parents[4]


def default_model_path(
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve UAV_MODEL_PATH or the USERPROFILE-based external default."""
    values = os.environ if environ is None else environ
    configured = values.get("UAV_MODEL_PATH", "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()

    if home is None:
        user_profile = values.get("USERPROFILE", "").strip()
        home = Path(user_profile) if user_profile else Path.home()

    return home / "Desktop" / "UAV_MODELS" / "military_kaggle_v1.pt"


def model_location_warning(model_path: Path, repo_root: Path | None = None) -> str | None:
    """Warn when a selected weight is inside the repository without rejecting it."""
    try:
        resolved_model = model_path.resolve(strict=False)
        resolved_repo = (repo_root or repository_root()).resolve(strict=True)
        resolved_model.relative_to(resolved_repo)
    except (ValueError, OSError):
        return None

    return (
        "Warning: this model is inside the repository. Existing local weights may be "
        "tested, but model weights and newly generated checkpoints must not be committed."
    )


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise DashboardError(
            "INVALID_SETTING",
            f"{field_name} must be a number, not a boolean.",
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DashboardError(
            "INVALID_SETTING",
            f"{field_name} must be a real number.",
        ) from error
    if not math.isfinite(number):
        raise DashboardError(
            "INVALID_SETTING",
            f"{field_name} must be finite.",
        )
    return number


@dataclass(frozen=True)
class ProcessingSettings:
    """Validated settings for one uploaded-video processing job."""

    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    image_size: int = DEFAULT_IMAGE_SIZE
    device: DeviceChoice = DeviceChoice.AUTO
    mode: InferenceMode = InferenceMode.DETECTION

    @classmethod
    def from_values(
        cls,
        confidence: object,
        iou: object,
        image_size: object,
        device: object,
        mode: object,
    ) -> "ProcessingSettings":
        """Build and validate settings received from UI components."""
        confidence_number = _finite_number(confidence, "Confidence threshold")
        iou_number = _finite_number(iou, "IoU threshold")

        if not 0.0 < confidence_number <= 1.0:
            raise DashboardError(
                "INVALID_CONFIDENCE",
                "Confidence threshold must be greater than 0 and at most 1.",
            )
        if not 0.0 < iou_number <= 1.0:
            raise DashboardError(
                "INVALID_IOU",
                "IoU threshold must be greater than 0 and at most 1.",
            )
        if isinstance(image_size, bool):
            raise DashboardError(
                "INVALID_IMAGE_SIZE",
                "Inference image size must be one of the supported integer values.",
            )
        try:
            image_size_number = int(image_size)
        except (TypeError, ValueError) as error:
            raise DashboardError(
                "INVALID_IMAGE_SIZE",
                "Inference image size must be an integer.",
            ) from error
        if image_size_number not in SUPPORTED_IMAGE_SIZES:
            raise DashboardError(
                "INVALID_IMAGE_SIZE",
                f"Inference image size must be one of {SUPPORTED_IMAGE_SIZES}.",
            )

        try:
            device_choice = DeviceChoice(str(device))
        except ValueError as error:
            raise DashboardError(
                "INVALID_DEVICE",
                f"Unsupported device selection: {device!r}.",
            ) from error
        try:
            inference_mode = InferenceMode(str(mode))
        except ValueError as error:
            raise DashboardError(
                "INVALID_MODE",
                f"Unsupported inference mode: {mode!r}.",
            ) from error

        return cls(
            confidence=confidence_number,
            iou=iou_number,
            image_size=image_size_number,
            device=device_choice,
            mode=inference_mode,
        )


def validate_video_path(value: str | Path | None) -> Path:
    """Validate a path supplied by the uploaded-video component."""
    if value is None or not str(value).strip():
        raise DashboardError(
            "VIDEO_REQUIRED",
            "Select or drag and drop a video before processing.",
        )
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise DashboardError(
            "VIDEO_NOT_FOUND",
            f"The selected video does not exist: {path}",
            recovery="Select the video again.",
        )
    return path


def validate_model_path(value: str | Path | None) -> Path:
    """Validate an Ultralytics PyTorch model path."""
    if value is None or not str(value).strip():
        raise DashboardError(
            "MODEL_REQUIRED",
            "Select a YOLO .pt model before processing.",
        )
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.suffix.lower() != ".pt":
        raise DashboardError(
            "MODEL_EXTENSION_INVALID",
            "The selected model must be an Ultralytics .pt file.",
        )
    if not path.is_file():
        raise DashboardError(
            "MODEL_NOT_FOUND",
            f"The selected model does not exist: {path}",
            recovery="Check UAV_MODEL_PATH or select an existing .pt file.",
        )
    return path
