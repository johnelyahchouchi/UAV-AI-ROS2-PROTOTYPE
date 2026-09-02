from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import ROBOFLOW_TANK_CLEAN_DATASET_DIR, ROBOFLOW_TANK_DATASET_DIR


SRC = ROBOFLOW_TANK_DATASET_DIR
DST = ROBOFLOW_TANK_CLEAN_DATASET_DIR

splits = ["train", "valid", "test"]
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

if DST.exists():
    shutil.rmtree(DST)

for split in splits:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    src_img_dir = SRC / split / "images"
    src_lbl_dir = SRC / split / "labels"

    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue

        shutil.copy2(img_path, DST / split / "images" / img_path.name)

        label_path = src_lbl_dir / (img_path.stem + ".txt")
        out_label_path = DST / split / "labels" / (img_path.stem + ".txt")

        cleaned_lines = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    # Convert every object in this tank dataset into class 0 = military_vehicle
                    parts[0] = "0"
                    cleaned_lines.append(" ".join(parts))

        with open(out_label_path, "w", encoding="utf-8") as f:
            if cleaned_lines:
                f.write("\n".join(cleaned_lines) + "\n")

data_yaml = f"""path: {DST.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['military_vehicle']
"""

with open(DST / "data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)

print("Clean tank dataset created:")
print(DST)
