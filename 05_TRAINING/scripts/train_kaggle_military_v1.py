import os
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.model_integrity import load_trusted_yolo


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data_value = os.environ.get("UAV_TRAINING_DATA_YAML", "").strip()
    model_value = os.environ.get("UAV_BASE_MODEL_PATH", "").strip()
    if not data_value or not model_value:
        raise RuntimeError("UAV_TRAINING_DATA_YAML and UAV_BASE_MODEL_PATH are required")
    output_dir = Path(os.environ.get("UAV_TRAINING_OUTPUT_DIR", PROJECT_ROOT / "05_TRAINING" / "local_runs"))
    model = load_trusted_yolo(model_value)

    model.train(
        data=str(Path(data_value).expanduser()),
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(output_dir),
        name="military_kaggle_yolov8s_v1",
        exist_ok=True,
        patience=5,
        cos_lr=True,
        close_mosaic=10
    )


if __name__ == "__main__":
    main()
