import cv2
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import BTR_DATASET_DIR, OUTPUTS_DIR, configured_path


INPUT_DIR = configured_path(
    "UAV_BTR_DEMO_IMAGES_DIR", BTR_DATASET_DIR / "test" / "images"
)
OUTPUT_VIDEO = configured_path(
    "UAV_BTR_DEMO_VIDEO_PATH", OUTPUTS_DIR / "windows_ai" / "btr_demo.mp4"
)
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

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
