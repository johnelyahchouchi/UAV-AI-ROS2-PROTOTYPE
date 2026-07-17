import random
import shutil
from pathlib import Path


DATASET_DIR = Path(r"C:\uav_datasets_master\07_tank_platform_recognition")
RAW_DIR = DATASET_DIR / "00_raw_by_class"
CLS_DIR = DATASET_DIR / "01_classifier_dataset_v2_exact_focus"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INCLUDE_CLASSES = {
    "armored_truck",
    "artillery",
    "ifv_bmp",
    "military_truck",

    "tank_t72",
    "tank_t80",
    "tank_t90",
    "tank_m1_abrams",
    "tank_leopard2",
    "tank_merkava",
}

EXCLUDE_CLASSES = {
    "tank_unknown",
    "98_skipped_review_later",
    "99_uncertain_review",
    "apc_btr",
    "tank_challenger2",
    "tank_leclerc",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

MIN_IMAGES_PER_CLASS = 50
MAX_IMAGES_PER_CLASS = 250

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

    print("\nBUILDING CLASSIFIER DATASET V2 - EXACT FOCUS")
    print("=" * 80)
    print(f"Raw source: {RAW_DIR}")
    print(f"Output:     {CLS_DIR}")
    print("=" * 80)
    print("Important: tank_unknown is excluded from this V2 training dataset.")
    print("=" * 80)

    total_train = 0
    total_val = 0
    total_test = 0
    used_classes = []

    for class_name in sorted(INCLUDE_CLASSES):
        class_folder = RAW_DIR / class_name

        if not class_folder.exists():
            print(f"[SKIP] {class_name:25s} | folder missing")
            continue

        images = get_images(class_folder)
        count_original = len(images)

        if count_original < MIN_IMAGES_PER_CLASS:
            print(f"[SKIP] {class_name:25s} | {count_original:5d} images | not enough")
            continue

        random.shuffle(images)

        if len(images) > MAX_IMAGES_PER_CLASS:
            images = images[:MAX_IMAGES_PER_CLASS]

        count_used = len(images)

        train_end = int(count_used * TRAIN_RATIO)
        val_end = train_end + int(count_used * VAL_RATIO)

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
            f"raw={count_original:5d} | "
            f"used={count_used:4d} | "
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
        print("Not ready. Need at least 2 classes.")
    else:
        print("V2 exact-focus classifier dataset is ready.")

    print("\nCreated here:")
    print(CLS_DIR)


if __name__ == "__main__":
    main()