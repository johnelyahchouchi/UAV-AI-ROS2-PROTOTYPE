from ultralytics import YOLO
import torch
from pathlib import Path
from multiprocessing import freeze_support
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import BASE_YOLO_MODEL, BTR_DATASET_DIR, configured_path


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

    data_yaml = BTR_DATASET_DIR / "data.yaml"
    training_output_dir = configured_path(
        "UAV_TRAINING_OUTPUT_DIR", PROJECT_ROOT / "05_TRAINING" / "detection_runs"
    )
    training_output_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"BTR data.yaml not found: {data_yaml}. Set UAV_BTR_DATASET_DIR."
        )
    if not BASE_YOLO_MODEL.is_file():
        raise FileNotFoundError(
            f"Base model not found: {BASE_YOLO_MODEL}. Set UAV_BASE_MODEL_PATH."
        )

    model = YOLO(str(BASE_YOLO_MODEL))

    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(training_output_dir),
        name="btr_yolov8n_local_test",
        exist_ok=True,
        patience=5,
        save=True,
        plots=True
    )

    print("====================================")
    print("TRAINING FINISHED")
    print("Best model saved here:")
    print(training_output_dir / "btr_yolov8n_local_test" / "weights" / "best.pt")
    print("====================================")


if __name__ == "__main__":
    freeze_support()
    main()
