import cv2
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.model_integrity import load_trusted_yolo
from uav_security.source_urls import resolve_video_source

YOUTUBE_URL = "https://www.youtube.com/@cityofmurfreesboro-traffic7836/live"

print("Extracting live stream URL...")
stream_url = resolve_video_source(os.environ.get("UAV_VIDEO_SOURCE", YOUTUBE_URL))

print("Opening stream...")
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("ERROR: Could not open stream.")
    exit()

model_path = os.environ.get("UAV_MODEL_PATH", "").strip()
if not model_path:
    raise RuntimeError("UAV_MODEL_PATH must identify a trusted local .pt checkpoint")
model = load_trusted_yolo(model_path)

print("Running YOLO live. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received. Stream may have ended.")
        break

    # Resize display frame to make it faster
    h, w = frame.shape[:2]
    new_w = 960
    new_h = int(h * new_w / w)
    frame = cv2.resize(frame, (new_w, new_h))

    results = model.predict(
        frame,
        device=0,
        imgsz=640,
        conf=0.25,
        classes=[0, 1, 2, 3, 5, 7],
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("Live YOLO Traffic Detection - Press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
