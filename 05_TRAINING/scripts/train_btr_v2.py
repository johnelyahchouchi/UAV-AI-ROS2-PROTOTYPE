from ultralytics import YOLO
import torch
from pathlib import Path
from multiprocessing import freeze_support


def main():
    print("====================================")
    print(" LOCAL BTR MODEL TRAINING V2")
    print("====================================")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available. Training will be very slow.")

    PROJECT_DIR = Path(r"C:\Users\UAVlab\Desktop\uav_ai_company")
    DATA_YAML = PROJECT_DIR / "BTR.v1i.yolov8" / "data.yaml"

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Could not find data.yaml here: {DATA_YAML}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(PROJECT_DIR / "training_runs"),
        name="btr_yolov8n_v2_50epochs",
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        cos_lr=True,
        close_mosaic=10
    )

    print("====================================")
    print("TRAINING V2 FINISHED")
    print("Best model saved here:")
    print(PROJECT_DIR / "training_runs" / "btr_yolov8n_v2_50epochs" / "weights" / "best.pt")
    print("====================================")


if __name__ == "__main__":
    freeze_support()
    main()