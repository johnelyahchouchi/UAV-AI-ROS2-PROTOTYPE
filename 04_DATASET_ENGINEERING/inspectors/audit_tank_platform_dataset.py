import csv
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.csv_safe import sanitize_csv_rows

DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"
OUTPUT_CSV = DATASET_DIR / "raw_dataset_audit.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

MIN_FOR_TEST = 50
MIN_FOR_PROTOTYPE = 300
MIN_FOR_STRONG_MODEL = 1000


def count_images(folder):
    images = []

    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return images


def get_status(count):
    if count == 0:
        return "EMPTY"

    if count < MIN_FOR_TEST:
        return "TOO SMALL"

    if count < MIN_FOR_PROTOTYPE:
        return "SMALL TEST ONLY"

    if count < MIN_FOR_STRONG_MODEL:
        return "PROTOTYPE READY"

    return "STRONG DATASET"


def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Raw folder not found: {RAW_DIR}")
        return

    class_folders = sorted([
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    ])

    rows = []
    total_images = 0

    print("\nTANK PLATFORM DATASET AUDIT")
    print("=" * 60)
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Raw folder:     {RAW_DIR}")
    print("=" * 60)

    for folder in class_folders:
        images = count_images(folder)
        count = len(images)
        total_images += count
        status = get_status(count)

        rows.append({
            "class_name": folder.name,
            "image_count": count,
            "status": status,
            "folder": str(folder)
        })

        print(f"{folder.name:25s} | {count:5d} images | {status}")

    print("=" * 60)
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)

    empty_classes = [r["class_name"] for r in rows if r["image_count"] == 0]
    weak_classes = [r["class_name"] for r in rows if 0 < r["image_count"] < MIN_FOR_TEST]
    prototype_classes = [r["class_name"] for r in rows if r["image_count"] >= MIN_FOR_PROTOTYPE]

    print("\nSUMMARY")
    print("-" * 60)

    if empty_classes:
        print("Empty classes:")
        for cls in empty_classes:
            print(f"  - {cls}")

    if weak_classes:
        print("\nVery weak classes:")
        for cls in weak_classes:
            print(f"  - {cls}")

    if prototype_classes:
        print("\nPrototype-ready classes:")
        for cls in prototype_classes:
            print(f"  - {cls}")

    print("\nTRAINING READINESS")
    print("-" * 60)

    if total_images == 0:
        print("Not ready. No images found yet.")
    elif total_images < 500:
        print("Very early stage. Good for testing scripts only, not real training.")
    elif total_images < 3000:
        print("Prototype stage. You can train, but expect limited accuracy.")
    else:
        print("Good prototype dataset. Training can produce useful results.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "image_count", "status", "folder"]
        )
        writer.writeheader()
        writer.writerows(sanitize_csv_rows(rows))

    print(f"\nCSV audit saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
