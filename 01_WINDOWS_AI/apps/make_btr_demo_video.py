import cv2
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\UAVlab\Desktop\uav_ai_company\BTR.v1i.yolov8\test\images")
OUTPUT_VIDEO = Path(r"C:\Users\UAVlab\Desktop\uav_ai_company\btr_demo.mp4")

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