import random
import shutil
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"
CLS_DIR = DATASET_DIR / "01_classifier_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXCLUDED_FOLDERS = {
    "98_skipped_review_later",
    "99_uncertain_review",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

MIN_IMAGES_PER_CLASS = 50

RANDOM_SEED = 42


def clean_output_folder():
    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)

    for split in ["train", "val", "test"]:
        (CLS_DIR / split).mkdir(parents=True, exist_ok=True)


def get_images(folder):
    images = []

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

    if not RAW_DIR.exists():
        print(f"[ERROR] Raw folder not found: {RAW_DIR}")
        return

    clean_output_folder()

    class_folders = sorted([
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir() and folder.name not in EXCLUDED_FOLDERS
    ])

    print("\nBUILDING TANK PLATFORM CLASSIFIER DATASET")
    print("=" * 70)
    print(f"Raw source: {RAW_DIR}")
    print(f"Output:     {CLS_DIR}")
    print("=" * 70)

    total_train = 0
    total_val = 0
    total_test = 0
    used_classes = []

    for class_folder in class_folders:
        class_name = class_folder.name
        images = get_images(class_folder)
        count = len(images)

        if count < MIN_IMAGES_PER_CLASS:
            print(f"[SKIP] {class_name:25s} | {count:5d} images | not enough")
            continue

        random.shuffle(images)

        train_end = int(count * TRAIN_RATIO)
        val_end = train_end + int(count * VAL_RATIO)

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
            f"[OK]   {class_name:25s} | "
            f"total={count:5d} | "
            f"train={len(train_images):4d} | "
            f"val={len(val_images):4d} | "
            f"test={len(test_images):4d}"
        )

    print("=" * 70)
    print(f"Used classes: {len(used_classes)}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Test images:  {total_test}")
    print("=" * 70)

    if len(used_classes) < 2:
        print("\nWARNING:")
        print("The classifier dataset has fewer than 2 usable classes.")
        print("This means we are NOT ready to train a real platform classifier yet.")
        print("Add images to tank_t72, tank_t90, tank_m1_abrams, tank_leopard2, etc.")
    else:
        print("\nDataset ready for classifier training.")

    print("\nClassifier dataset created here:")
    print(CLS_DIR)


if __name__ == "__main__":
    main()
