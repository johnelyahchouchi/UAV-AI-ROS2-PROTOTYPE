import os
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
SRC = DATASET_ROOT / "05_amad5_aerial_military_5class"
DST = DATASET_ROOT / "05_amad5_aerial_military_5class_clean"

CONFIG_DIR = PROJECT_ROOT / "05_TRAINING" / "configs" / "training_configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def long_path(p: Path) -> str:
    p = p.resolve()
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    return "\\\\?\\" + s


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(long_path(src), long_path(dst))


if not SRC.exists():
    raise SystemExit(f"Source dataset not found: {SRC}")

if DST.exists():
    print(f"Removing old clean folder: {DST}")
    shutil.rmtree(long_path(DST))

total_images = 0
total_labels = 0
errors = []

for split in ["train", "val", "test"]:
    src_images = SRC / split / "images"
    src_labels = SRC / split / "labels"

    dst_images = DST / split / "images"
    dst_labels = DST / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    split_images = 0
    split_labels = 0

    if not src_images.exists():
        print(f"Missing split images folder: {src_images}")
        continue

    images = []
    for p in src_images.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    images = sorted(images, key=lambda x: x.name.lower())

    for idx, img_path in enumerate(images, start=1):
        new_stem = f"amad5_{split}_{idx:06d}"
        new_img = dst_images / f"{new_stem}{img_path.suffix.lower()}"
        old_label = src_labels / f"{img_path.stem}.txt"
        new_label = dst_labels / f"{new_stem}.txt"

        try:
            safe_copy(img_path, new_img)
            split_images += 1
            total_images += 1

            if old_label.exists():
                safe_copy(old_label, new_label)
                split_labels += 1
                total_labels += 1
            else:
                new_label.write_text("", encoding="utf-8")

        except Exception as e:
            errors.append((str(img_path), str(e)))

    print(f"{split}: images={split_images}, labels={split_labels}")

yaml_text = """path: C:/uav_datasets_master/05_amad5_aerial_military_5class_clean
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: military_tank
  1: military_vehicle
  2: civilian
  3: soldier
  4: civilian_vehicle
"""

yaml_path = CONFIG_DIR / "05_amad5_aerial_military_5class_clean.yaml"
yaml_path.write_text(yaml_text, encoding="utf-8")

print("\nDONE")
print(f"Clean AMAD5 folder: {DST}")
print(f"Clean YAML: {yaml_path}")
print(f"Total images copied: {total_images}")
print(f"Total labels copied: {total_labels}")
print(f"Errors skipped: {len(errors)}")

if errors:
    error_log = CONFIG_DIR / "amad5_cleaning_errors.txt"
    with open(error_log, "w", encoding="utf-8") as f:
        for file_path, err in errors:
            f.write(file_path + "\n")
            f.write(err + "\n\n")
    print(f"Error log saved: {error_log}")
