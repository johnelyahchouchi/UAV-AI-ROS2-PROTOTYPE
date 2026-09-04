from pathlib import Path
import cv2
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from uav_security.model_integrity import load_trusted_yolo

DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_DIR / "04_DATASET_ENGINEERING" / "local_data"))
MODEL_PATH = Path(os.environ.get("UAV_MODEL_PATH", PROJECT_DIR / "03_MODELS" / "active" / "detector" / "military_kaggle_v1.pt"))
VIDEO_PATH = Path(os.environ.get("UAV_TEST_VIDEO", PROJECT_DIR / "06_TEST_MEDIA" / "videos" / "vehicles.mp4"))

RAW_DIR = DATASET_DIR / "00_raw_by_class"
OUTPUT_REVIEW_DIR = RAW_DIR / "99_uncertain_review"

CONF_THRESHOLD = 0.25
IMG_SIZE = 960
FRAME_STRIDE = 5
MIN_CROP_SIZE = 35

ALLOWED_KEYWORDS = [
    "military",
    "tank",
    "truck",
    "vehicle",
    "artillery",
    "armored",
    "armoured",
    "btr",
    "bmp",
    "apc",
]


def normalize_name(name):
    return str(name).lower().strip().replace(" ", "_").replace("-", "_")


def is_military_class(class_name):
    c = normalize_name(class_name)
    return any(k in c for k in ALLOWED_KEYWORDS)


def safe_crop(frame, x1, y1, x2, y2, pad_ratio=0.12):
    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return frame[y1:y2, x1:x2]


def main():
    OUTPUT_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model:")
    print(MODEL_PATH)

    model = load_trusted_yolo(MODEL_PATH)

    print("Opening video:")
    print(VIDEO_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    frame_index = 0
    saved_count = 0

    print("Extracting military target crops...")
    print("Crops will be saved to:")
    print(OUTPUT_REVIEW_DIR)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        if frame_index % FRAME_STRIDE != 0:
            continue

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = normalize_name(model.names[cls_id])

                if not is_military_class(class_name):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                    continue

                crop = safe_crop(frame, x1, y1, x2, y2)

                if crop.size == 0:
                    continue

                saved_count += 1

                filename = (
                    f"mission_crop_"
                    f"frame_{frame_index:06d}_"
                    f"{class_name}_"
                    f"conf_{conf:.2f}_"
                    f"{saved_count:05d}.jpg"
                )

                out_path = OUTPUT_REVIEW_DIR / filename
                cv2.imwrite(str(out_path), crop)

        if frame_index % 100 == 0:
            print(f"Frame {frame_index} | Saved crops: {saved_count}")

    cap.release()

    print("=" * 60)
    print("DONE")
    print(f"Total saved crops: {saved_count}")
    print(f"Saved inside: {OUTPUT_REVIEW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
