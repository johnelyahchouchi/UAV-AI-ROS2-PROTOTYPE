import random
import shutil
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"
CLS_DIR = DATASET_DIR / "01_tank_type_classifier_dataset_v3"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INCLUDE_CLASSES = {
    "tank_t72",
    "tank_t80",
    "tank_t90",
    "tank_m1_abrams",
    "tank_leopard2",
    "tank_merkava",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

MIN_IMAGES_PER_CLASS = 50
MAX_IMAGES_PER_CLASS = 180

RANDOM_SEED = 42


def clean_output_folder():
    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)

    for split in ["train", "val", "test"]:
        (CLS_DIR / split).mkdir(parents=True, exist_ok=True)


def get_images(folder):
    images = []

    if not folder.exists():
        return images

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return sorted(images)


def copy_images(images, split_name, class_name):
    dest_dir = CLS_DIR / split_name / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in images:
        dest = dest_dir / src.name

        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1

            while True:
                candidate = dest_dir / f"{stem}_{counter:03d}{suffix}"

                if not candidate.exists():
                    dest = candidate
                    break

                counter += 1

        shutil.copy2(src, dest)


def main():
    random.seed(RANDOM_SEED)

    clean_output_folder()

    print("\nBUILDING TANK TYPE CLASSIFIER DATASET V3")
    print("=" * 80)
    print("This dataset contains ONLY tank types.")
    print("No BMP, no trucks, no artillery, no tank_unknown.")
    print("=" * 80)

    total_train = 0
    total_val = 0
    total_test = 0
    used_classes = []

    for class_name in sorted(INCLUDE_CLASSES):
        class_folder = RAW_DIR / class_name
        images = get_images(class_folder)

        raw_count = len(images)

        if raw_count < MIN_IMAGES_PER_CLASS:
            print(f"[SKIP] {class_name:20s} | {raw_count:4d} images | not enough")
            continue

        random.shuffle(images)

        if len(images) > MAX_IMAGES_PER_CLASS:
            images = images[:MAX_IMAGES_PER_CLASS]

        used_count = len(images)

        train_end = int(used_count * TRAIN_RATIO)
        val_end = train_end + int(used_count * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

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

    print("=" * 80)
    print(f"Used classes: {len(used_classes)}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Test images:  {total_test}")
    print("=" * 80)

    if len(used_classes) < 2:
        print("Not ready. Need at least 2 tank classes.")
    else:
        print("Tank Type Classifier V3 dataset is ready.")

    print("\nCreated here:")
    print(CLS_DIR)


if __name__ == "__main__":
    main()
