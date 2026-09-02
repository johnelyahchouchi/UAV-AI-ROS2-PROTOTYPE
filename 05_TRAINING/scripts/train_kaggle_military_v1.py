from ultralytics import YOLO
import torch
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import KAGGLE_DATASET_DIR, YOLOV8S_MODEL, configured_path


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data_yaml = KAGGLE_DATASET_DIR / "data.yaml"
    training_output_dir = configured_path(
        "UAV_TRAINING_OUTPUT_DIR", PROJECT_ROOT / "05_TRAINING" / "detection_runs"
    )
    training_output_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Kaggle data.yaml not found: {data_yaml}. Set UAV_KAGGLE_DATASET_DIR."
        )
    if not YOLOV8S_MODEL.is_file():
        raise FileNotFoundError(
            f"Base model not found: {YOLOV8S_MODEL}. Set UAV_KAGGLE_BASE_MODEL_PATH."
        )

    model = YOLO(str(YOLOV8S_MODEL))

    model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(training_output_dir),
        name="military_kaggle_yolov8s_v1",
        exist_ok=True,
        patience=5,
        cos_lr=True,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
