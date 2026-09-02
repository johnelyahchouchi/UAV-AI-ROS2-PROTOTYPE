import random
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import TANK_RECOGNITION_DATASET_DIR

DATASET_DIR = TANK_RECOGNITION_DATASET_DIR
RAW_DIR = DATASET_DIR / "00_raw_by_class"
OUT_DIR = DATASET_DIR / "01_tank_type_classifier_dataset_v4_safe_unknown"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = {
    "tank_t72": 250,
    "tank_t80": 120,
    "tank_t90": 120,
    "tank_m1_abrams": 120,
    "tank_leopard2": 150,
    "tank_merkava": 150,
    "tank_unknown": 180,
}

MIN_IMAGES_PER_CLASS = 50

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


def get_images(folder):
    if not folder.exists():
        return []

    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def clean_output():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)


def copy_images(images, split, class_name):
    dest_dir = OUT_DIR / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in images:
        dest = dest_dir / src.name

        if dest.exists():
            counter = 1
            while True:
                new_dest = dest_dir / f"{src.stem}_{counter:03d}{src.suffix}"
                if not new_dest.exists():
                    dest = new_dest
                    break
                counter += 1

        shutil.copy2(src, dest)


def main():
    random.seed(RANDOM_SEED)
    clean_output()

    print("\nBUILDING TANK TYPE CLASSIFIER DATASET V4 - SAFE UNKNOWN")
    print("=" * 90)
    print("Goal: exact tank type when clear, tank_unknown when unclear.")
    print("Important: tank_unknown is capped to avoid unknown spam.")
    print("=" * 90)

    total_train = 0
    total_val = 0
    total_test = 0
    used_classes = []

    for class_name, max_images in CLASSES.items():
        folder = RAW_DIR / class_name
        images = get_images(folder)
        raw_count = len(images)

        if raw_count < MIN_IMAGES_PER_CLASS:
            print(f"[SKIP] {class_name:20s} | raw={raw_count:4d} | not enough images")
            continue

        random.shuffle(images)

        used_images = images[:max_images]
        used_count = len(used_images)

        train_end = int(used_count * TRAIN_RATIO)
        val_end = train_end + int(used_count * VAL_RATIO)

        train_images = used_images[:train_end]
        val_images = used_images[train_end:val_end]
        test_images = used_images[val_end:]

        copy_images(train_images, "train", class_name)
        copy_images(val_images, "val", class_name)
        copy_images(test_images, "test", class_name)

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        used_classes.append(class_name)

        print(
            f"[OK]   {class_name:20s} | "
            f"raw={raw_count:4d} | "
            f"used={used_count:4d} | "
            f"train={len(train_images):4d} | "
            f"val={len(val_images):4d} | "
            f"test={len(test_images):4d}"
        )

    print("=" * 90)
    print(f"Used classes: {len(used_classes)}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Test images:  {total_test}")
    print("=" * 90)
    print("Created dataset:")
    print(OUT_DIR)

    if "tank_unknown" in used_classes:
        print("\nSAFE UNKNOWN ENABLED:")
        print("The model can now say tank_unknown, but unknown is balanced and capped.")

    print("\nNext step after this: train Tank Type Classifier V4.")


if __name__ == "__main__":
    main()
