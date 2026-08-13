"""Validation and defaults for uncertainty experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import os
from pathlib import Path
from typing import Mapping

from .errors import DashboardError


DEFAULT_IMAGE_SIZE = 960
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS_IOU = 0.45
DEFAULT_MATCH_IOU = 0.50
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 42
DEFAULT_OVERLAP_IOU = 0.80
DEFAULT_VIDEO_INTERVAL_SECONDS = 5.0
DEFAULT_VIDEO_MAX_FRAMES = 20
SAMPLE_COUNT_CHOICES = (5, 10, 20, 30)
IMAGE_SIZE_CHOICES = (320, 480, 640, 960, 1280)
METHOD_INPUT_PERTURBATION_V1 = "Input Perturbation V1"


class DeviceChoice(str, Enum):
    """Supported Ultralytics device selections."""

    AUTO = "Auto"
    GPU_0 = "GPU 0"
    CPU = "CPU"

    @property
    def argument(self) -> int | str | None:
        """Return the corresponding Ultralytics device argument."""
        if self is DeviceChoice.AUTO:
            return None
        if self is DeviceChoice.GPU_0:
            return 0
        return "cpu"


class InputKind(str, Enum):
    """Supported experiment input types."""

    IMAGE = "Image"
    VIDEO = "Video"


class VideoSamplingMode(str, Enum):
    """Supported bounded video-frame selection policies."""

    INTERVAL = "Interval"
    MANUAL = "Manual timestamps"


def dashboard_root() -> Path:
    """Return the dashboard subsystem root."""
    return Path(__file__).resolve().parents[2]


def repository_root() -> Path:
    """Return the repository root derived from the package location."""
    return Path(__file__).resolve().parents[4]


def default_model_path(
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve an environment override or external USERPROFILE-based default."""
    values = os.environ if environ is None else environ
    configured = values.get("UAV_MODEL_PATH", "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    if home is None:
        profile = values.get("USERPROFILE", "").strip()
        home = Path(profile) if profile else Path.home()
    return home / "Desktop" / "UAV_MODELS" / "military_kaggle_v1.pt"


def repository_path_warning(path: Path) -> str | None:
    """Warn, without rejecting, when a runtime input is inside the repository."""
    try:
        path.resolve(strict=False).relative_to(repository_root().resolve(strict=True))
    except (ValueError, OSError):
        return None
    return (
        "Warning: this runtime input is inside the Git repository. Do not commit "
        "model weights, datasets, uploaded media, or generated experiment outputs."
    )


def validate_model_path(value: str | Path | None) -> Path:
    """Validate an existing local Ultralytics detection checkpoint path."""
    if value is None or not str(value).strip():
        raise DashboardError("MODEL_REQUIRED", "Select a YOLO .pt model.")
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.suffix.lower() != ".pt":
        raise DashboardError("MODEL_EXTENSION", "The model must be a .pt file.")
    if not path.is_file():
        raise DashboardError("MODEL_NOT_FOUND", f"Model file does not exist: {path}")
    return path


def validate_input_path(value: str | Path | None, kind: InputKind) -> Path:
    """Validate a local image or video selected by Gradio."""
    if value is None or not str(value).strip():
        raise DashboardError("INPUT_REQUIRED", f"Select a {kind.value.lower()} first.")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise DashboardError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    return path


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise DashboardError("INVALID_SETTING", f"{name} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DashboardError("INVALID_SETTING", f"{name} must be numeric.") from error
    if not math.isfinite(number):
        raise DashboardError("INVALID_SETTING", f"{name} must be finite.")
    return number


@dataclass(frozen=True)
class ExperimentSettings:
    """Validated settings for one V1 experiment."""

    method: str = METHOD_INPUT_PERTURBATION_V1
    sample_count: int = DEFAULT_SAMPLE_COUNT
    seed: int = DEFAULT_SEED
    image_size: int = DEFAULT_IMAGE_SIZE
    confidence: float = DEFAULT_CONFIDENCE
    nms_iou: float = DEFAULT_NMS_IOU
    match_iou: float = DEFAULT_MATCH_IOU
    device: DeviceChoice = DeviceChoice.AUTO
    overlap_iou: float = DEFAULT_OVERLAP_IOU

    @classmethod
    def from_values(
        cls,
        method: object,
        sample_count: object,
        seed: object,
        image_size: object,
        confidence: object,
        nms_iou: object,
        match_iou: object,
        device: object,
        overlap_iou: object = DEFAULT_OVERLAP_IOU,
    ) -> "ExperimentSettings":
        """Build settings from UI values with explicit validation."""
        if str(method) != METHOD_INPUT_PERTURBATION_V1:
            raise DashboardError("METHOD_UNSUPPORTED", f"Unsupported method: {method!r}")
        try:
            samples = int(sample_count)
            seed_value = int(seed)
            size = int(image_size)
        except (TypeError, ValueError) as error:
            raise DashboardError(
                "INVALID_INTEGER_SETTING",
                "Sample count, seed, and image size must be integers.",
            ) from error
        if isinstance(sample_count, bool) or not 1 <= samples <= 100:
            raise DashboardError("INVALID_SAMPLE_COUNT", "Sample count must be 1–100.")
        if isinstance(image_size, bool) or size <= 0:
            raise DashboardError("INVALID_IMAGE_SIZE", "Image size must be positive.")
        confidence_value = _finite(confidence, "Confidence threshold")
        nms_value = _finite(nms_iou, "NMS IoU")
        match_value = _finite(match_iou, "Matching IoU")
        overlap_value = _finite(overlap_iou, "Overlap diagnostic IoU")
        for name, value in (
            ("Confidence threshold", confidence_value),
            ("NMS IoU", nms_value),
            ("Matching IoU", match_value),
            ("Overlap diagnostic IoU", overlap_value),
        ):
            if not 0.0 < value <= 1.0:
                raise DashboardError("INVALID_THRESHOLD", f"{name} must be in (0, 1].")
        try:
            choice = DeviceChoice(str(device))
        except ValueError as error:
            raise DashboardError("INVALID_DEVICE", f"Unsupported device: {device!r}") from error
        return cls(
            method=METHOD_INPUT_PERTURBATION_V1,
            sample_count=samples,
            seed=seed_value,
            image_size=size,
            confidence=confidence_value,
            nms_iou=nms_value,
            match_iou=match_value,
            device=choice,
            overlap_iou=overlap_value,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready configuration mapping."""
        values = asdict(self)
        values["device"] = self.device.value
        values["total_inference_samples"] = self.sample_count + 1
        return values
