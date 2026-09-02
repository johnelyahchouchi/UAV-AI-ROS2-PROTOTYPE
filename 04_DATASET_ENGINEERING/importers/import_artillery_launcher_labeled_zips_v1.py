import ast
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import TANK_RECOGNITION_DATASET_DIR

ZIP_DIR = TANK_RECOGNITION_DATASET_DIR / "03_downloaded_roboflow_zips"
RAW_DIR = TANK_RECOGNITION_DATASET_DIR / "00_raw_by_class"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEST_CLASSES = [
    "rocket_launcher_grad",
    "rocket_launcher_smerch",
    "mlrs_unknown",
    "self_propelled_artillery",
    "unknown_artillery",
]


def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_class_names(yaml_text):
    lines = yaml_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:") and "[" in stripped:
            raw = stripped.split("names:", 1)[1].strip()
            names = ast.literal_eval(raw)
            return {i: str(name) for i, name in enumerate(names)}

    names = {}
    inside_names = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("names:"):
            inside_names = True
            continue

        if inside_names:
            match = re.match(r"^(\d+)\s*:\s*(.+)$", stripped)

            if match:
                class_id = int(match.group(1))
                class_name = match.group(2).strip().strip("'").strip('"')
                names[class_id] = class_name

            elif stripped and not stripped[0].isdigit():
                break

    return names


def get_import_rules(zip_name):
    z = zip_name.lower()

    rules = {}

    if "bm-grad" in z or "bm_grad" in z:
        rules["bm grad"] = "rocket_launcher_grad"

    if z.startswith("mlrs") or "mlrs" in z:
        rules["mlrs"] = "mlrs_unknown"

    if "my_equipment" in z:
        rules["grad"] = "rocket_launcher_grad"
        rules["smerch"] = "rocket_launcher_smerch"

    if "military artillery val" in z or "military_artillery_val" in z:
        rules["self propelled artillery"] = "self_propelled_artillery"

    if z.startswith("artillery"):
        rules["artillery"] = "unknown_artillery"

    return rules


def unique_path(path):
    if not path.exists():
        return path

    counter = 1

    while True:
        candidate = path.parent / f"{path.stem}_{counter:04d}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def find_image_for_label(label_path, image_by_stem):
    label_stem = Path(label_path).stem

    candidates = image_by_stem.get(label_stem, [])

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    label_parts = label_path.replace("\\", "/").split("/")

    for candidate in candidates:
        candidate_parts = candidate.replace("\\", "/").split("/")

        if len(label_parts) > 1 and len(candidate_parts) > 1:
            if label_parts[0] == candidate_parts[0]:
                return candidate

    return candidates[0]


def crop_yolo_box(image, xc, yc, bw, bh, pad_ratio=0.08):
    h, w = image.shape[:2]

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    box_w = x2 - x1
    box_h = y2 - y1

    pad = int(max(box_w, box_h) * pad_ratio)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    return crop


def inspect_and_import_zip(zip_path):
    zip_name = zip_path.name
    rules = get_import_rules(zip_name)

    print("\n" + "=" * 100)
    print(f"ZIP: {zip_name}")
    print("=" * 100)

    if not rules:
        print("No import rules for this ZIP. Skipped.")
        return {}

    print("Import rules:")
    for src, dst in rules.items():
        print(f"  {src} -> {dst}")

    imported_counts = defaultdict(int)

    with zipfile.ZipFile(zip_path, "r") as z:
        files = z.namelist()

        yaml_files = [
            f for f in files
            if f.lower().endswith(("data.yaml", "dataset.yaml"))
        ]

        if not yaml_files:
            print("No data.yaml or dataset.yaml found. Skipped.")
            return imported_counts

        yaml_text = z.read(yaml_files[0]).decode("utf-8", errors="ignore")
        class_names = parse_class_names(yaml_text)

        class_id_to_dest = {}

        for class_id, class_name in class_names.items():
            normalized = normalize(class_name)

            if normalized in rules:
                class_id_to_dest[class_id] = rules[normalized]

        if not class_id_to_dest:
            print("No matching classes found inside this ZIP. Skipped.")
            return imported_counts

        print("\nMatched classes:")
        for class_id, dest_class in class_id_to_dest.items():
            print(f"  {class_id}: {class_names[class_id]} -> {dest_class}")

        image_files = [
            f for f in files
            if "/images/" in f.replace("\\", "/").lower()
            and Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]

        label_files = [
            f for f in files
            if "/labels/" in f.replace("\\", "/").lower()
            and f.lower().endswith(".txt")
        ]

        image_by_stem = defaultdict(list)

        for image_path in image_files:
            image_by_stem[Path(image_path).stem].append(image_path)

        safe_zip_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", zip_path.stem)

        for label_path in label_files:
            label_text = z.read(label_path).decode("utf-8", errors="ignore").strip()

            if not label_text:
                continue

            image_path = find_image_for_label(label_path, image_by_stem)

            if image_path is None:
                continue

            image_bytes = np.frombuffer(z.read(image_path), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                continue

            label_stem = Path(label_path).stem

            for line_index, line in enumerate(label_text.splitlines()):
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except Exception:
                    continue

                if class_id not in class_id_to_dest:
                    continue

                dest_class = class_id_to_dest[class_id]
                dest_dir = RAW_DIR / dest_class
                dest_dir.mkdir(parents=True, exist_ok=True)

                crop = crop_yolo_box(image, xc, yc, bw, bh)

                if crop is None:
                    continue

                out_name = f"{safe_zip_stem}_{dest_class}_{label_stem}_{line_index:02d}.jpg"
                out_path = unique_path(dest_dir / out_name)

                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                imported_counts[dest_class] += 1

    print("\nImported from this ZIP:")
    for dest_class, count in sorted(imported_counts.items()):
        print(f"  {dest_class:25s}: {count}")

    return imported_counts


def main():
    print("\nIMPORTING ARTILLERY / LAUNCHER LABELED YOLO ZIPS")
    print("=" * 100)
    print("This script uses trusted YOLO labels.")
    print("It does not manually sort images.")
    print("It crops labeled objects and copies them into raw class folders.")
    print("=" * 100)

    for class_name in DEST_CLASSES:
        (RAW_DIR / class_name).mkdir(parents=True, exist_ok=True)

    total_counts = defaultdict(int)

    zips = sorted(ZIP_DIR.glob("*.zip"))

    for zip_path in zips:
        counts = inspect_and_import_zip(zip_path)

        for class_name, count in counts.items():
            total_counts[class_name] += count

    print("\n" + "=" * 100)
    print("TOTAL IMPORTED COUNTS")
    print("=" * 100)

    for class_name in DEST_CLASSES:
        print(f"{class_name:25s}: {total_counts[class_name]}")

    print("=" * 100)
    print("Import complete.")
    print("Now run the dataset audit.")
    print("=" * 100)


if __name__ == "__main__":
    main()
