import cv2
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import MULTICLASS_DATASET_DIR


BASE = MULTICLASS_DATASET_DIR
VIDEO_DIR = BASE / "00_raw_videos"
OUT_DIR = BASE / "01_extracted_frames"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_STEP = 10  # save 1 frame every 10 frames

videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi")) + list(VIDEO_DIR.glob("*.mov"))

print(f"Found {len(videos)} videos")

for video_path in videos:
    print(f"Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    video_name = video_path.stem
    frame_id = 0
    saved_id = 0

    video_out = OUT_DIR / video_name
    video_out.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_STEP == 0:
            out_path = video_out / f"{video_name}_frame_{saved_id:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_id += 1

        frame_id += 1

    cap.release()
    print(f"Saved {saved_id} frames from {video_path.name}")

print("Frame extraction done.")
