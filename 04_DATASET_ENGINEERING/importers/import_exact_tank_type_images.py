import csv
import hashlib
import os
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.csv_safe import sanitize_csv_rows

DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))

INCOMING_DIR = DATASET_DIR / "02_incoming_exact_type_images"
RAW_DIR = DATASET_DIR / "00_raw_by_class"
LOG_PATH = DATASET_DIR / "exact_type_import_log.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

ALLOWED_CLASSES = {
    "tank_t72",
    "tank_t80",
    "tank_t90",
    "tank_m1_abrams",
    "tank_leopard2",
    "tank_merkava",
    "tank_challenger2",
    "tank_leclerc",
    "ifv_bmp",
    "apc_btr",
    "artillery",
}


def file_md5(path):
    h = hashlib.md5(usedforsecurity=False)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def get_existing_hashes():
    hashes = set()

    for file in RAW_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                hashes.add(file_md5(file))
            except Exception:
                pass

    return hashes


def next_filename(class_name, dest_dir):
    existing = list(dest_dir.glob(f"{class_name}_*.jpg"))

    max_id = 0

    for file in existing:
        stem = file.stem

        try:
            number_part = stem.split("_")[-1]
            n = int(number_part)
            max_id = max(max_id, n)
        except Exception:
            continue

    return dest_dir / f"{class_name}_{max_id + 1:06d}.jpg"


def validate_and_load_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None, "could_not_open"

    h, w = img.shape[:2]

    if w < 40 or h < 40:
        return None, "too_small"

    return img, "ok"


def save_clean_jpg(img, out_path):
    cv2.imwrite(
        str(out_path),
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )


def main():
    if not INCOMING_DIR.exists():
        print(f"[ERROR] Incoming folder not found: {INCOMING_DIR}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes = get_existing_hashes()

    rows = []

    imported = 0
    skipped = 0
    duplicates = 0
    broken = 0

    print("\nIMPORTING EXACT TANK TYPE IMAGES")
    print("=" * 70)
    print(f"Incoming: {INCOMING_DIR}")
    print(f"Raw out:  {RAW_DIR}")
    print("=" * 70)

    class_folders = sorted([
        folder for folder in INCOMING_DIR.iterdir()
        if folder.is_dir()
    ])

    for class_folder in class_folders:
        class_name = class_folder.name

        if class_name not in ALLOWED_CLASSES:
            print(f"[SKIP CLASS] {class_name} is not in allowed classes.")
            continue

        dest_dir = RAW_DIR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            file for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\nClass: {class_name} | incoming images: {len(image_files)}")

        for src in image_files:
            status = "unknown"
            dest = ""

            try:
                src_hash = file_md5(src)

                if src_hash in existing_hashes:
                    duplicates += 1
                    skipped += 1
                    status = "duplicate_skipped"
                    print(f"  [DUP] {src.name}")

                    rows.append({
                        "source_file": str(src),
                        "class_name": class_name,
                        "status": status,
                        "output_file": dest,
                    })
                    continue

                img, reason = validate_and_load_image(src)

                if img is None:
                    broken += 1
                    skipped += 1
                    status = reason
                    print(f"  [BAD] {src.name} | {reason}")

                    rows.append({
                        "source_file": str(src),
                        "class_name": class_name,
                        "status": status,
                        "output_file": dest,
                    })
                    continue

                dest_path = next_filename(class_name, dest_dir)

                save_clean_jpg(img, dest_path)

                existing_hashes.add(src_hash)
                imported += 1
                status = "imported"
                dest = str(dest_path)

                print(f"  [OK] {src.name} -> {dest_path.name}")

                rows.append({
                    "source_file": str(src),
                    "class_name": class_name,
                    "status": status,
                    "output_file": dest,
                })

            except Exception as e:
                skipped += 1
                status = f"error: {e}"
                print(f"  [ERROR] {src.name} | {e}")

                rows.append({
                    "source_file": str(src),
                    "class_name": class_name,
                    "status": status,
                    "output_file": dest,
                })

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "class_name",
                "status",
                "output_file",
            ]
        )
        writer.writeheader()
        writer.writerows(sanitize_csv_rows(rows))

    print("\n" + "=" * 70)
    print("IMPORT FINISHED")
    print(f"Imported:   {imported}")
    print(f"Skipped:    {skipped}")
    print(f"Duplicates: {duplicates}")
    print(f"Broken:     {broken}")
    print("=" * 70)
    print(f"Log saved to: {LOG_PATH}")
    print("\nNext command:")
    print("python audit_tank_platform_dataset.py")


if __name__ == "__main__":
    main()
