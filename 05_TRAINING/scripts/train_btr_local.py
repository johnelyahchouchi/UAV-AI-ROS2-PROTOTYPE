import os
import torch
from pathlib import Path
from multiprocessing import freeze_support
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from uav_security.model_integrity import load_trusted_yolo


def main():
    print("====================================")
    print(" LOCAL BTR / ARMORED MODEL TRAINING")
    print("====================================")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available. Training will be very slow.")

    data_value = os.environ.get("UAV_TRAINING_DATA_YAML", "").strip()
    model_value = os.environ.get("UAV_BASE_MODEL_PATH", "").strip()
    if not data_value or not model_value:
        raise RuntimeError("UAV_TRAINING_DATA_YAML and UAV_BASE_MODEL_PATH are required")
    data_yaml = Path(data_value).expanduser()
    output_dir = Path(
        os.environ.get(
            "UAV_TRAINING_OUTPUT_DIR", PROJECT_DIR / "05_TRAINING" / "local_runs"
        )
    )

    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find data.yaml here: {data_yaml}")

    model = load_trusted_yolo(model_value)

    model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(output_dir),
        name="btr_yolov8n_local_test",
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
    )

    print("====================================")
    print("TRAINING FINISHED")
    print("Best model saved here:")
    print(output_dir / "btr_yolov8n_local_test" / "weights" / "best.pt")
    print("====================================")


if __name__ == "__main__":
    freeze_support()
    main()
