import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path(os.environ.get("UAV_BTR_IMAGE_DIR", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data" / "BTR" / "test" / "images"))
OUTPUT_VIDEO = Path(os.environ.get("UAV_DEMO_VIDEO_OUT", PROJECT_ROOT / "08_OUTPUTS" / "btr_demo.mp4"))

images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    images.extend(INPUT_DIR.glob(ext))

images = sorted(images)

if not images:
    raise FileNotFoundError(f"No images found in {INPUT_DIR}")

width, height = 960, 540
fps = 10
seconds_per_image = 2
frames_per_image = fps * seconds_per_image

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img = cv2.resize(img, (width, height))

    for _ in range(frames_per_image):
        out.write(img)

out.release()

print("BTR demo video created:")
print(OUTPUT_VIDEO)
print("Images used:", len(images))
