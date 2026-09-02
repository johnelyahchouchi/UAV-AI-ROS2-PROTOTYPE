import cv2
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import MULTICLASS_DATASET_DIR


BASE = MULTICLASS_DATASET_DIR

INPUT_DIR = BASE / "01_extracted_frames"
OUTPUT_DIR = BASE / "02_selected_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    images.extend(INPUT_DIR.rglob(ext))

images = sorted(images)

print(f"Found {len(images)} extracted images.")
print("Controls:")
print("K = keep image")
print("S = skip image")
print("Q = quit")
print("ESC = quit")

kept = 0
skipped = 0

for i, img_path in enumerate(images):
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    display = img.copy()

    h, w = display.shape[:2]
    max_w = 1100
    max_h = 700

    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    cv2.putText(
        display,
        f"{i+1}/{len(images)} | K=keep | S=skip | Q=quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Select useful training images", display)

    key = cv2.waitKey(0) & 0xFF

    if key in [ord("k"), ord("K")]:
        out_name = f"selected_{kept:06d}_{img_path.stem}.jpg"
        out_path = OUTPUT_DIR / out_name
        shutil.copy2(img_path, out_path)
        kept += 1
        print(f"KEPT: {out_name}")

    elif key in [ord("s"), ord("S"), ord(" ")]:
        skipped += 1
        print(f"SKIPPED: {img_path.name}")

    elif key in [ord("q"), ord("Q"), 27]:
        print("Stopped by user.")
        break

cv2.destroyAllWindows()

print("--------------------------------")
print(f"Done.")
print(f"Kept: {kept}")
print(f"Skipped: {skipped}")
print(f"Selected images saved in:")
print(OUTPUT_DIR)
