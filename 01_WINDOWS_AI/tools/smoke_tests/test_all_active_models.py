import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), f"class_{class_id}"))

    try:
        return str(names[int(class_id)])
    except Exception:
        return f"class_{class_id}"


def classification_top_k(result, top_k=3):
    if result.probs is None:
        return []

    probabilities = result.probs.data.detach().cpu().numpy()
    indices = probabilities.argsort()[::-1][:top_k]

    return [
        {
            "class_id": int(index),
            "class_name": get_class_name(result.names, int(index)),
            "confidence": round(float(probabilities[index]), 6),
        }
        for index in indices
    ]


def read_test_frame(video_path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.25))

    success, frame = capture.read()

    if not success or frame is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()

    capture.release()

    if not success or frame is None:
        raise RuntimeError(f"Cannot read a frame from: {video_path}")

    return frame


def select_largest_crop(frame, boxes):
    if boxes is None or len(boxes) == 0:
        return frame.copy(), None

    height, width = frame.shape[:2]

    largest_area = -1
    selected_box = None

    for box in boxes.xyxy.detach().cpu().numpy():
        x1, y1, x2, y2 = [int(value) for value in box]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))

        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area > largest_area:
            largest_area = area
            selected_box = (x1, y1, x2, y2)

    if selected_box is None:
        return frame.copy(), None

    x1, y1, x2, y2 = selected_box
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame.copy(), None

    return crop, selected_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    video_path = (
        Path(args.video)
        if args.video
        else project_root / "06_TEST_MEDIA" / "videos" / "vehicles.mp4"
    )

    output_dir = (
        project_root
        / "08_OUTPUTS"
        / "model_tests"
        / "offline_active_models_smoke_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "military_detector": (
            project_root
            / "03_MODELS"
            / "active"
            / "detector"
            / "military_kaggle_v1.pt"
        ),
        "tank_classifier": (
            project_root
            / "03_MODELS"
            / "active"
            / "tank_classifier"
            / "tank_type_classifier_v3_only_tanks.pt"
        ),
        "armored_classifier": (
            project_root
            / "03_MODELS"
            / "active"
            / "armored_classifier"
            / "armored_vehicle_classifier_v1.pt"
        ),
        "artillery_classifier": (
            project_root
            / "03_MODELS"
            / "active"
            / "artillery_classifier"
            / "artillery_launcher_classifier_v1.pt"
        ),
    }

    missing = [
        str(path)
        for path in model_paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    report = {
        "status": "RUNNING",
        "video": str(video_path),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "models": {},
    }

    frame = read_test_frame(video_path)

    print("=" * 70)
    print("OFFLINE ACTIVE-MODEL SMOKE TEST")
    print("=" * 70)
    print(f"Video:  {video_path}")
    print(f"Device: {device}")

    detector_start = time.perf_counter()

    detector = YOLO(str(model_paths["military_detector"]))

    detector_result = detector.predict(
        source=frame,
        conf=0.15,
        iou=0.45,
        imgsz=960,
        device=device,
        verbose=False,
    )[0]

    detector_time = time.perf_counter() - detector_start
    detector_count = (
        len(detector_result.boxes)
        if detector_result.boxes is not None
        else 0
    )

    detector_detections = []

    if detector_result.boxes is not None:
        for box in detector_result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            detector_detections.append(
                {
                    "class_id": class_id,
                    "class_name": get_class_name(
                        detector_result.names,
                        class_id,
                    ),
                    "confidence": round(confidence, 6),
                    "xyxy": [
                        round(float(value), 2)
                        for value in box.xyxy[0].detach().cpu().tolist()
                    ],
                }
            )

    annotated_path = output_dir / "detector_annotated_frame.jpg"
    cv2.imwrite(str(annotated_path), detector_result.plot())

    crop, crop_box = select_largest_crop(
        frame,
        detector_result.boxes,
    )

    crop_path = output_dir / "classifier_test_crop.jpg"
    cv2.imwrite(str(crop_path), crop)

    report["models"]["military_detector"] = {
        "status": "PASS",
        "path": str(model_paths["military_detector"]),
        "task": str(detector.task),
        "inference_seconds": round(detector_time, 4),
        "detection_count": detector_count,
        "detections": detector_detections,
        "annotated_frame": str(annotated_path),
        "selected_crop_box": crop_box,
    }

    print(
        f"[PASS] Military detector: "
        f"{detector_count} detections in {detector_time:.3f}s"
    )

    classifier_names = [
        "tank_classifier",
        "armored_classifier",
        "artillery_classifier",
    ]

    failures = []

    for classifier_name in classifier_names:
        model_path = model_paths[classifier_name]

        try:
            start = time.perf_counter()

            classifier = YOLO(str(model_path))

            result = classifier.predict(
                source=crop,
                device=device,
                verbose=False,
            )[0]

            elapsed = time.perf_counter() - start
            predictions = classification_top_k(result, top_k=3)

            report["models"][classifier_name] = {
                "status": "PASS",
                "path": str(model_path),
                "task": str(classifier.task),
                "inference_seconds": round(elapsed, 4),
                "top_predictions": predictions,
            }

            top_prediction = (
                predictions[0]["class_name"]
                if predictions
                else "no prediction"
            )

            print(
                f"[PASS] {classifier_name}: "
                f"{top_prediction} in {elapsed:.3f}s"
            )

        except Exception as error:
            failures.append(classifier_name)

            report["models"][classifier_name] = {
                "status": "FAIL",
                "path": str(model_path),
                "error": str(error),
            }

            print(f"[FAIL] {classifier_name}: {error}")

    report["status"] = "FAIL" if failures else "PASS"
    report["failed_models"] = failures
    report["classifier_crop"] = str(crop_path)

    report_path = output_dir / "active_models_smoke_test_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("=" * 70)
    print(f"Final status: {report['status']}")
    print(f"Report: {report_path}")
    print(f"Annotated frame: {annotated_path}")
    print(f"Classifier crop: {crop_path}")
    print("=" * 70)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
