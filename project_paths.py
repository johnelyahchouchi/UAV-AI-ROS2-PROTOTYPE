"""Portable filesystem locations shared by repository scripts.

Repository-owned defaults are anchored to this file. Large models, datasets, test
media, and outputs can be redirected with environment variables without editing
source code. Relative override values are interpreted from the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parent


def configured_path(
    variable: str,
    default: str | Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return an environment-configured path or a repository-anchored default."""

    values = os.environ if environ is None else environ
    configured = values.get(variable, "").strip()
    candidate = (
        Path(os.path.expandvars(configured)).expanduser()
        if configured
        else Path(default)
    )
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


MODELS_DIR = configured_path("UAV_MODELS_DIR", PROJECT_ROOT / "03_MODELS")
DATASETS_ROOT = configured_path("UAV_DATASETS_ROOT", PROJECT_ROOT / "datasets")
TEST_MEDIA_DIR = configured_path("UAV_TEST_MEDIA_DIR", PROJECT_ROOT / "06_TEST_MEDIA")
OUTPUTS_DIR = configured_path("UAV_OUTPUTS_DIR", PROJECT_ROOT / "outputs")

ACTIVE_DETECTOR_MODEL = configured_path(
    "UAV_MODEL_PATH",
    MODELS_DIR / "active" / "detector" / "military_kaggle_v1.pt",
)
BASE_YOLO_MODEL = configured_path(
    "UAV_BASE_MODEL_PATH",
    MODELS_DIR / "base_weights" / "yolov8n.pt",
)
YOLOV8S_MODEL = configured_path(
    "UAV_KAGGLE_BASE_MODEL_PATH",
    MODELS_DIR / "base_weights" / "yolov8s.pt",
)
BTR_MODEL = configured_path(
    "UAV_BTR_MODEL_PATH",
    MODELS_DIR / "experimental" / "btr_best_v2.pt",
)

TANK_RECOGNITION_DATASET_DIR = configured_path(
    "UAV_TANK_RECOGNITION_DATASET_DIR",
    DATASETS_ROOT / "07_tank_platform_recognition",
)
BTR_DATASET_DIR = configured_path(
    "UAV_BTR_DATASET_DIR",
    DATASETS_ROOT / "01_detection" / "BTR_v1",
)
KAGGLE_DATASET_DIR = configured_path(
    "UAV_KAGGLE_DATASET_DIR",
    DATASETS_ROOT / "01_detection" / "military_kaggle_v1",
)
AMAD5_DATASET_DIR = configured_path(
    "UAV_AMAD5_DATASET_DIR",
    DATASETS_ROOT / "05_amad5_aerial_military_5class",
)
AMAD5_CLEAN_DATASET_DIR = configured_path(
    "UAV_AMAD5_CLEAN_DATASET_DIR",
    DATASETS_ROOT / "05_amad5_aerial_military_5class_clean",
)
ROBOFLOW_MILITARY_DATASET_DIR = configured_path(
    "UAV_ROBOFLOW_MILITARY_DATASET_DIR",
    DATASETS_ROOT / "03_roboflow_military_footage_8class",
)
ROBOFLOW_TANK_DATASET_DIR = configured_path(
    "UAV_ROBOFLOW_TANK_DATASET_DIR",
    DATASETS_ROOT / "04_roboflow_tank_2class",
)
ROBOFLOW_TANK_CLEAN_DATASET_DIR = configured_path(
    "UAV_ROBOFLOW_TANK_CLEAN_DATASET_DIR",
    DATASETS_ROOT / "04_roboflow_tank_2class_clean",
)
MULTICLASS_DATASET_DIR = configured_path(
    "UAV_MULTICLASS_DATASET_DIR",
    DATASETS_ROOT / "dataset_v3_military_multiclass",
)
DATASET_EXTRACT_DIR = configured_path(
    "UAV_DATASET_EXTRACT_DIR",
    OUTPUTS_DIR / "dataset_extract_cache",
)
