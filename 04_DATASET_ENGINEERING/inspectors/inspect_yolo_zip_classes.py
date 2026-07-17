import ast
import re
import zipfile
from collections import defaultdict
from pathlib import Path


ZIP_PATH = Path(
    r"C:\uav_datasets_master\07_tank_platform_recognition\03_downloaded_roboflow_zips\my_equipment_bmp_grad_truck_yolov8.zip"
)


def parse_class_names(yaml_text):
    lines = yaml_text.splitlines()

    # Case 1:
    # names: ['BMP', 'Grad', 'Military truck', ...]
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("names:") and "[" in stripped:
            raw = stripped.split("names:", 1)[1].strip()
            names = ast.literal_eval(raw)
            return {i: str(name) for i, name in enumerate(names)}

    # Case 2:
    # names:
    #   0: BMP
    #   1: Grad
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

    print("\nYOLO ZIP CLASS INSPECTION")
    print("=" * 80)
    print(f"ZIP: {ZIP_PATH}")
    print("=" * 80)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        files = z.namelist()

        yaml_files = [
            f for f in files
            if f.lower().endswith(("data.yaml", "dataset.yaml"))
        ]

        if not yaml_files:
            print("No data.yaml or dataset.yaml found inside ZIP.")
            return

        yaml_path = yaml_files[0]
        yaml_text = z.read(yaml_path).decode("utf-8", errors="ignore")

        class_names = parse_class_names(yaml_text)

        print(f"YAML found: {yaml_path}")
        print("\nClasses found:")
        for class_id, class_name in class_names.items():
            print(f"  {class_id}: {class_name}")

        label_files = [
            f for f in files
            if "/labels/" in f.replace("\\", "/").lower()
            and f.lower().endswith(".txt")
        ]

        box_counts = defaultdict(int)
        image_label_file_counts = defaultdict(int)
        total_boxes = 0

        for label_path in label_files:
            text = z.read(label_path).decode("utf-8", errors="ignore").strip()

            if not text:
                continue

            classes_in_this_file = set()

            for line in text.splitlines():
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                except Exception:
                    continue

                box_counts[class_id] += 1
                classes_in_this_file.add(class_id)
                total_boxes += 1

            for class_id in classes_in_this_file:
                image_label_file_counts[class_id] += 1

        print("\nLabel files found:", len(label_files))
        print("Total labeled objects:", total_boxes)

        print("\nObject count per class:")
        print("-" * 80)

        for class_id in sorted(class_names.keys()):
            class_name = class_names[class_id]
            boxes = box_counts[class_id]
            files_with_class = image_label_file_counts[class_id]

            print(
                f"{class_id:2d} | {class_name:20s} | "
                f"objects={boxes:5d} | images_with_class={files_with_class:5d}"
            )

        print("=" * 80)

        grad_id = None
        smerch_id = None

        for class_id, class_name in class_names.items():
            name = class_name.lower().strip()

            if name == "grad":
                grad_id = class_id

            if name == "smerch":
                smerch_id = class_id

        print("\nARTILLERY USEFUL COUNTS")
        print("-" * 80)

        if grad_id is not None:
            print(f"Grad objects:   {box_counts[grad_id]}")
        else:
            print("Grad class not found.")

        if smerch_id is not None:
            print(f"Smerch objects: {box_counts[smerch_id]}")
        else:
            print("Smerch class not found.")

        print("=" * 80)
        print("Inspection complete. No files were changed.")


if __name__ == "__main__":
    main()