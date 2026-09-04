from pathlib import Path
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))

datasets = {
    "01_kaggle_military_assets": DATASET_ROOT / "01_kaggle_military_assets" / "military_object_dataset" / "data.yaml",
    "02_roboflow_military_footage": DATASET_ROOT / "02_roboflow_military_footage" / "data_fixed.yaml",
    "03_roboflow_tank_clean": DATASET_ROOT / "03_roboflow_tank_clean" / "data.yaml",
    "04_amad5_aerial": DATASET_ROOT / "04_amad5_aerial" / "data_fixed.yaml",
}

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

print("=" * 80)
print("UAV AI DATASET INVENTORY")
print("=" * 80)

for name, yaml_path in datasets.items():
    print(f"\nDATASET: {name}")
    print(f"YAML: {yaml_path}")

    if not yaml_path.exists():
        print("STATUS: YAML NOT FOUND")
        continue

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data["path"])
    names = data.get("names", [])

    print(f"ROOT: {root}")
    print(f"CLASSES: {names}")

    for split in ["train", "val", "valid", "test"]:
        split_key = split
        if split_key not in data:
            continue

        img_dir = root / data[split_key]
        label_dir = Path(str(img_dir).replace("images", "labels"))

        img_count = 0
        label_count = 0

        if img_dir.exists():
            img_count = sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in image_exts)

        if label_dir.exists():
            label_count = sum(1 for p in label_dir.rglob("*.txt"))

        print(f"{split_key}: images={img_count}, labels={label_count}")

print("\nDone.")
