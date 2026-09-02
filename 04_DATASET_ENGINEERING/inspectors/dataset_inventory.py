from pathlib import Path
import sys
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    AMAD5_CLEAN_DATASET_DIR,
    KAGGLE_DATASET_DIR,
    ROBOFLOW_MILITARY_DATASET_DIR,
    ROBOFLOW_TANK_CLEAN_DATASET_DIR,
)


datasets = {
    "01_kaggle_military_assets": KAGGLE_DATASET_DIR / "data.yaml",
    "02_roboflow_military_footage": ROBOFLOW_MILITARY_DATASET_DIR / "data_fixed.yaml",
    "03_roboflow_tank_clean": ROBOFLOW_TANK_CLEAN_DATASET_DIR / "data.yaml",
    "04_amad5_aerial": AMAD5_CLEAN_DATASET_DIR / "data_fixed.yaml",
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

    root = Path(data["path"]).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve(strict=False)
    names = data.get("names", [])

    print(f"ROOT: {root}")
    print(f"CLASSES: {names}")

    for split in ["train", "val", "valid", "test"]:
        split_key = split
        if split_key not in data:
            continue

        img_dir = root / data[split_key]
        label_dir = img_dir.parent / "labels"

        img_count = 0
        label_count = 0

        if img_dir.exists():
            img_count = sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in image_exts)

        if label_dir.exists():
            label_count = sum(1 for p in label_dir.rglob("*.txt"))

        print(f"{split_key}: images={img_count}, labels={label_count}")

print("\nDone.")
