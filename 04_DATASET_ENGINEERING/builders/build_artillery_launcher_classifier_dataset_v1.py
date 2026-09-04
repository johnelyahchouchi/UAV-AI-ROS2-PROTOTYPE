import random
import shutil
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"
OUT_DIR = DATASET_DIR / "01_artillery_launcher_classifier_dataset_v1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = {
    "rocket_launcher_grad": 500,
    "mlrs_unknown": 341,
    "self_propelled_artillery": 500,
    "unknown_artillery": 500,
}

MIN_IMAGES_PER_CLASS = 80

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


def copy_images_short_names(images, split, class_name):
    dest_dir = OUT_DIR / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for index, src in enumerate(images, start=1):
        suffix = src.suffix.lower()

        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"

        dest = dest_dir / f"{class_name}_{split}_{index:06d}{suffix}"

        shutil.copy2(src, dest)


def main():
    random.seed(RANDOM_SEED)
    clean_output()

    print("\nBUILDING ARTILLERY / LAUNCHER CLASSIFIER DATASET V1")
    print("=" * 90)
    print("Using SHORT filenames to avoid Windows path-length errors.")
    print("=" * 90)
    print("Classes:")
    print("  rocket_launcher_grad")
    print("  mlrs_unknown")
    print("  self_propelled_artillery")
    print("  unknown_artillery")
    print("=" * 90)
    print("Skipping for now:")
    print("  rocket_launcher_smerch because it has too few images")
    print("  artillery_cannon because it has zero images")
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
            print(f"[SKIP] {class_name:25s} | raw={raw_count:4d} | not enough images")
            continue

        random.shuffle(images)

        used_images = images[:max_images]
        used_count = len(used_images)

        train_end = int(used_count * TRAIN_RATIO)
        val_end = train_end + int(used_count * VAL_RATIO)

        train_images = used_images[:train_end]
        val_images = used_images[train_end:val_end]
        test_images = used_images[val_end:]

        copy_images_short_names(train_images, "train", class_name)
        copy_images_short_names(val_images, "val", class_name)
        copy_images_short_names(test_images, "test", class_name)

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        used_classes.append(class_name)

        print(
            f"[OK]   {class_name:25s} | "
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

    if len(used_classes) >= 3:
        print("\nArtillery / Launcher Dataset V1 is ready for training.")
    else:
        print("\nNot enough classes. Add more data first.")


if __name__ == "__main__":
    main()
