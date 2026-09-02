import ast
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import TANK_RECOGNITION_DATASET_DIR, configured_path

ZIP_PATH = configured_path(
    "UAV_ARTILLERY_ZIP_PATH",
    TANK_RECOGNITION_DATASET_DIR
    / "03_downloaded_roboflow_zips"
    / "military_vehicles_126_classes_yolov8.zip",
)

KEYWORDS = [
    "grad",
    "smerch",
    "mlrs",
    "rocket",
    "launcher",
    "artillery",
    "howitzer",
    "cannon",
    "gun",
    "msta",
    "tos",
    "bm-21",
    "bm21",
    "bm-30",
    "bm30",
]


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


def main():
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        return

    print("\nARTILLERY KEYWORD INSPECTION")
    print("=" * 90)
    print(f"ZIP: {ZIP_PATH}")
    print("=" * 90)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        files = z.namelist()

        yaml_files = [
            f for f in files
            if f.lower().endswith(("data.yaml", "dataset.yaml"))
        ]

        if not yaml_files:
            print("No data.yaml or dataset.yaml found.")
            return

        yaml_path = yaml_files[0]
        yaml_text = z.read(yaml_path).decode("utf-8", errors="ignore")
        class_names = parse_class_names(yaml_text)

        print(f"YAML found: {yaml_path}")
        print(f"Total classes found: {len(class_names)}")

        label_files = [
            f for f in files
            if "/labels/" in f.replace("\\", "/").lower()
            and f.lower().endswith(".txt")
        ]

        box_counts = defaultdict(int)
        image_counts = defaultdict(int)

        for label_path in label_files:
            text = z.read(label_path).decode("utf-8", errors="ignore").strip()

            if not text:
                continue

            classes_in_file = set()

            for line in text.splitlines():
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                except Exception:
                    continue

                box_counts[class_id] += 1
                classes_in_file.add(class_id)

            for class_id in classes_in_file:
                image_counts[class_id] += 1

        print(f"Label files found: {len(label_files)}")
        print("\nARTILLERY / LAUNCHER RELATED CLASSES")
        print("-" * 90)

        found_any = False

        for class_id in sorted(class_names.keys()):
            class_name = class_names[class_id]
            lowered = class_name.lower()

            if any(keyword in lowered for keyword in KEYWORDS):
                found_any = True
                print(
                    f"{class_id:3d} | {class_name:35s} | "
                    f"objects={box_counts[class_id]:5d} | "
                    f"images_with_class={image_counts[class_id]:5d}"
                )

        if not found_any:
            print("No artillery/launcher keyword classes found.")

        print("=" * 90)
        print("Inspection complete. No files were changed.")


if __name__ == "__main__":
    main()
