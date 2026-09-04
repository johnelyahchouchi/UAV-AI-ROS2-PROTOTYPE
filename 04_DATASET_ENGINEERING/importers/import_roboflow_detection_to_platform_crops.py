import csv
import hashlib
import os
import zipfile
from pathlib import Path
import sys

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.csv_safe import sanitize_csv_rows
from uav_security.safe_zip import safe_extract_zip

DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
ZIP_DIR = DATASET_DIR / "03_downloaded_roboflow_zips"
EXTRACT_DIR = Path(os.environ.get("UAV_DATASET_EXTRACT_ROOT", DATASET_DIR / "extracted"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"

LOG_PATH = DATASET_DIR / "roboflow_detection_import_log.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

CLASS_MAPPING = {
    "t-72": "tank_t72",
    "t72": "tank_t72",
    "tank-t-72": "tank_t72",

    "t-80": "tank_t80",
    "t80": "tank_t80",
    "tank-t-80": "tank_t80",

    "t-90": "tank_t90",
    "t90": "tank_t90",
    "tank-t-90": "tank_t90",

    "tank": "tank_unknown",
    "tank-": "tank_unknown",
    "military-tank": "tank_unknown",

    "btr-80": "apc_btr",
    "btr80": "apc_btr",
    "btr-82": "apc_btr",
    "btr82": "apc_btr",
    "btr-86": "apc_btr",
    "btr86": "apc_btr",
    "btr": "apc_btr",

    "bmp": "ifv_bmp",
    "bmp-1": "ifv_bmp",
    "bmp-2": "ifv_bmp",
    "bmp-3": "ifv_bmp",

    "bm-21": "artillery",
    "bm21": "artillery",
    "grad": "artillery",
    "smerch": "artillery",
    "artillery": "artillery",

    "military-truck": "military_truck",
    "militarytruck": "military_truck",
    "military_vehicle": "military_truck",

    "tiger": "armored_truck",
    "tigr": "armored_truck",

    "mt-lb": "armored_truck",
    "mtlb": "armored_truck",

    "t-64": "tank_unknown",
    "t64": "tank_unknown",
    "t-72": "tank_t72",
    "t72": "tank_t72",
    "t-80": "tank_t80",
    "t80": "tank_t80",

    "m1-abrams": "tank_m1_abrams",
    "m1a1-abrams": "tank_m1_abrams",
    "m1a2-abrams": "tank_m1_abrams",
    "m1a2": "tank_m1_abrams",

    "leopard-2": "tank_leopard2",
    "leopard-2a4m": "tank_leopard2",
    "leopard-2a5": "tank_leopard2",
    "leopard-2a6": "tank_leopard2",
    "leopard-2a7": "tank_leopard2",
    "leopard-2ng": "tank_leopard2",
    "leopard-2p1": "tank_leopard2",

    "merkava-m1": "tank_merkava",
    "merkava-m2": "tank_merkava",
    "merkava-m3": "tank_merkava",
    "merkava-m4": "tank_merkava",
    "merkava-m4-meil-ruach": "tank_merkava",

    "challenger-2": "tank_challenger2",

    "leclerk": "tank_leclerc",
    "leclerc": "tank_leclerc",
}


def normalize_name(name):
    return str(name).lower().strip().replace("_", "-").replace(" ", "-")


def file_md5(path):
    h = hashlib.md5(usedforsecurity=False)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def crop_md5(img):
    return hashlib.md5(img.tobytes(), usedforsecurity=False).hexdigest()


def get_existing_hashes():
    hashes = set()

    if not RAW_DIR.exists():
        return hashes

    for file in RAW_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            try:
                hashes.add(file_md5(file))
            except Exception:
                pass

    return hashes


def unzip_all():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))

    if not zip_files:
        print(f"[ERROR] No ZIP files found in: {ZIP_DIR}")
        return []

    extracted_folders = []

    for zip_path in zip_files:
        out_dir = EXTRACT_DIR / zip_path.stem

        if out_dir.exists():
            print(f"[SKIP UNZIP] Already extracted: {out_dir}")
        else:
            print(f"[UNZIP] {zip_path.name} -> {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_extract_zip(zip_path, out_dir)

        extracted_folders.append(out_dir)

    return extracted_folders


def find_data_yaml(dataset_root):
    candidates = (
        list(dataset_root.rglob("data.yaml")) +
        list(dataset_root.rglob("data.yml")) +
        list(dataset_root.rglob("dataset.yaml")) +
        list(dataset_root.rglob("dataset.yml"))
    )

    if not candidates:
        return None

    return candidates[0]


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)

    if names is None:
        raise ValueError("No names field found in YAML")

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    raise ValueError("Unsupported names format")


def build_image_index(dataset_root):
    """
    Roboflow sometimes stores labels and images in slightly different layouts.
    Instead of guessing the folder path, this builds a dictionary:
    image_stem -> image_path
    """
    image_index = {}

    for file in dataset_root.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image_index[file.stem.lower()] = file

    return image_index


def get_label_files(dataset_root):
    label_files = []

    for file in dataset_root.rglob("*.txt"):
        parts_lower = [p.lower() for p in file.parts]

        if "labels" in parts_lower:
            label_files.append(file)

    return sorted(label_files)


def find_matching_image(label_path, image_index):
    stem = label_path.stem.lower()

    if stem in image_index:
        return image_index[stem]

    # Sometimes Roboflow names contain extra suffixes.
    for image_stem, image_path in image_index.items():
        if image_stem == stem:
            return image_path

        if image_stem.startswith(stem) or stem.startswith(image_stem):
            return image_path

    return None


def yolo_to_xyxy(line, img_w, img_h):
    values = line.strip().split()

    if len(values) < 5:
        return None

    try:
        cls_id = int(float(values[0]))
        cx = float(values[1])
        cy = float(values[2])
        bw = float(values[3])
        bh = float(values[4])
    except Exception:
        return None

    x1 = int((cx - bw / 2.0) * img_w)
    y1 = int((cy - bh / 2.0) * img_h)
    x2 = int((cx + bw / 2.0) * img_w)
    y2 = int((cy + bh / 2.0) * img_h)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return cls_id, x1, y1, x2, y2


def safe_crop(img, x1, y1, x2, y2, pad_ratio=0.12):
    h, w = img.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    crop = img[y1:y2, x1:x2]

    return crop


def next_filename(class_name, dest_dir):
    existing = list(dest_dir.glob(f"{class_name}_roboflow_*.jpg"))
    max_id = 0

    for file in existing:
        try:
            n = int(file.stem.split("_")[-1])
            max_id = max(max_id, n)
        except Exception:
            pass

    return dest_dir / f"{class_name}_roboflow_{max_id + 1:06d}.jpg"


def main():
    print("\nROBOFLOW OBJECT DETECTION IMPORTER V2")
    print("=" * 70)
    print(f"ZIP folder: {ZIP_DIR}")
    print(f"Extract to:  {EXTRACT_DIR}")
    print(f"Raw output:  {RAW_DIR}")
    print("=" * 70)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    extracted_folders = unzip_all()
    existing_hashes = get_existing_hashes()

    rows = []

    total_label_files = 0
    total_yolo_objects = 0
    total_crops = 0

    skipped_unmapped = 0
    skipped_bad = 0
    skipped_duplicate = 0
    skipped_no_image = 0

    for dataset_root in extracted_folders:
        print(f"\n[DATASET] {dataset_root.name}")

        data_yaml = find_data_yaml(dataset_root)

        if data_yaml is None:
            print("  [ERROR] No data.yaml / dataset.yaml found. Skipping.")
            continue

        print(f"  YAML: {data_yaml}")

        try:
            class_names = load_class_names(data_yaml)
        except Exception as e:
            print(f"  [ERROR] Could not load class names: {e}")
            continue

        print("  Classes found:")
        for cls_id, cls_name in class_names.items():
            mapped = CLASS_MAPPING.get(normalize_name(cls_name), "UNMAPPED")
            print(f"    {cls_id}: {cls_name} -> {mapped}")

        image_index = build_image_index(dataset_root)
        label_files = get_label_files(dataset_root)

        print(f"  Images found: {len(image_index)}")
        print(f"  Label files:  {len(label_files)}")

        total_label_files += len(label_files)

        for label_path in label_files:
            image_path = find_matching_image(label_path, image_index)

            if image_path is None:
                skipped_no_image += 1
                rows.append([str(label_path), "", "", "", "image_not_found"])
                continue

            img = cv2.imread(str(image_path))

            if img is None:
                skipped_bad += 1
                rows.append([str(label_path), str(image_path), "", "", "image_could_not_open"])
                continue

            img_h, img_w = img.shape[:2]

            try:
                lines = [
                    line.strip()
                    for line in label_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception:
                skipped_bad += 1
                rows.append([str(label_path), str(image_path), "", "", "label_read_error"])
                continue

            for line in lines:
                parsed = yolo_to_xyxy(line, img_w, img_h)

                if parsed is None:
                    skipped_bad += 1
                    rows.append([str(label_path), str(image_path), "", "", f"bad_label_line: {line[:80]}"])
                    continue

                cls_id, x1, y1, x2, y2 = parsed
                total_yolo_objects += 1

                source_class = class_names.get(cls_id, f"class_{cls_id}")
                target_class = CLASS_MAPPING.get(normalize_name(source_class), None)

                if target_class is None:
                    skipped_unmapped += 1
                    rows.append([str(label_path), str(image_path), source_class, "", "unmapped_class"])
                    continue

                crop = safe_crop(img, x1, y1, x2, y2)

                if crop is None or crop.size == 0:
                    skipped_bad += 1
                    rows.append([str(label_path), str(image_path), source_class, target_class, "empty_crop"])
                    continue

                ch, cw = crop.shape[:2]

                if cw < 35 or ch < 35:
                    skipped_bad += 1
                    rows.append([str(label_path), str(image_path), source_class, target_class, "crop_too_small"])
                    continue

                hsh = crop_md5(crop)

                if hsh in existing_hashes:
                    skipped_duplicate += 1
                    rows.append([str(label_path), str(image_path), source_class, target_class, "duplicate_crop"])
                    continue

                dest_dir = RAW_DIR / target_class
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = next_filename(target_class, dest_dir)

                cv2.imwrite(
                    str(dest_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )

                existing_hashes.add(hsh)
                total_crops += 1

                rows.append([str(label_path), str(image_path), source_class, target_class, str(dest_path)])

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_file",
            "image_file",
            "source_class",
            "target_class",
            "status_or_output",
        ])
        writer.writerows(sanitize_csv_rows(rows))

    print("\n" + "=" * 70)
    print("IMPORT FINISHED")
    print(f"Label files found:      {total_label_files}")
    print(f"YOLO objects read:      {total_yolo_objects}")
    print(f"Imported crops:         {total_crops}")
    print(f"Skipped no image:       {skipped_no_image}")
    print(f"Skipped unmapped:       {skipped_unmapped}")
    print(f"Skipped bad:            {skipped_bad}")
    print(f"Skipped duplicate:      {skipped_duplicate}")
    print("=" * 70)
    print(f"Log saved to: {LOG_PATH}")
    print("\nNext:")
    print("python audit_tank_platform_dataset.py")


if __name__ == "__main__":
    main()
