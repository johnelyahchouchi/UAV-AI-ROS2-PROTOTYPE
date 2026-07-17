import cv2
import subprocess
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/@cityofmurfreesboro-traffic7836/live"

print("Extracting live stream URL...")
stream_url = subprocess.check_output(
    ["yt-dlp", "-g", "-f", "best", YOUTUBE_URL],
    text=True
).strip().splitlines()[0]

print("Opening stream...")
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("ERROR: Could not open stream.")
    exit()

model = YOLO("yolov8n.pt")

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