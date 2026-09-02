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

ZIP_DIR = configured_path(
    "UAV_ARTILLERY_ZIPS_DIR",
    TANK_RECOGNITION_DATASET_DIR / "03_downloaded_roboflow_zips",
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
    "tosan",
    "bm-21",
    "bm21",
    "bm-30",
    "bm30",
]


def parse_class_names(yaml_text):
    lines = yaml_text.splitlines()

    # names: ['a', 'b', 'c']
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:") and "[" in stripped:
            raw = stripped.split("names:", 1)[1].strip()
            names = ast.literal_eval(raw)
            return {i: str(name) for i, name in enumerate(names)}

    # names:
    #   0: class
    #   1: class
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


def inspect_zip(zip_path):
    print("\n" + "=" * 100)
    print(f"ZIP: {zip_path.name}")
    print("=" * 100)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
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

            print(f"YAML: {yaml_path}")
            print(f"Classes found: {len(class_names)}")

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

            print("\nAll classes:")
            print("-" * 100)
            for class_id in sorted(class_names.keys()):
                class_name = class_names[class_id]
                print(
                    f"{class_id:3d} | {class_name:35s} | "
                    f"objects={box_counts[class_id]:5d} | "
                    f"images_with_class={image_counts[class_id]:5d}"
                )

            print("\nArtillery / launcher candidate classes:")
            print("-" * 100)

            found_candidate = False

            for class_id in sorted(class_names.keys()):
                class_name = class_names[class_id]
                lowered = class_name.lower()

                if any(keyword in lowered for keyword in KEYWORDS):
                    found_candidate = True
                    print(
                        f"{class_id:3d} | {class_name:35s} | "
                        f"objects={box_counts[class_id]:5d} | "
                        f"images_with_class={image_counts[class_id]:5d}"
                    )

            if not found_candidate:
                print("No artillery/launcher candidate classes found.")

    except Exception as e:
        print(f"ERROR reading ZIP: {e}")


def main():
    if not ZIP_DIR.exists():
        print(f"Folder not found: {ZIP_DIR}")
        return

    zips = sorted(ZIP_DIR.glob("*.zip"))

    print("\nINSPECTING ALL ARTILLERY / LAUNCHER ZIP DATASETS")
    print("=" * 100)
    print(f"Folder: {ZIP_DIR}")
    print(f"ZIP files found: {len(zips)}")
    print("=" * 100)

    for zip_path in zips:
        inspect_zip(zip_path)

    print("\n" + "=" * 100)
    print("Inspection complete. No files were changed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
