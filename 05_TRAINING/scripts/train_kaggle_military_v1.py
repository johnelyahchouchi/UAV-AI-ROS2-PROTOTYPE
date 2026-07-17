from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolov8s.pt")

model.train(
    data=r"C:\Users\UAVlab\Desktop\uav_ai_company\big_datasets\01_kaggle_military_assets\military_object_dataset\data.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    project=r"C:\Users\UAVlab\Desktop\uav_ai_company\training_runs",
    name="military_kaggle_yolov8s_v1",
    exist_ok=True,
    patience=5,
    cos_lr=True,
    close_mosaic=10
)