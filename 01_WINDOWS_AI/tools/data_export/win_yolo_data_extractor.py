import argparse
import csv
import math
import sys
import time
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import ACTIVE_DETECTOR_MODEL, OUTPUTS_DIR


MILITARY_KEYWORDS = [
    "military_tank",
    "military_vehicle",
    "military_truck",
    "military_artillery",
    "tank",
    "btr",
    "bmp",
    "apc",
    "armored",
    "artillery",
]


class CleanIDMapper:
    def __init__(self):
        self.raw_to_clean = {}
        self.next_clean_id = 1

    def get_clean_id(self, raw_id):
        if raw_id is None:
            return None

        raw_key = str(raw_id)

        if raw_key not in self.raw_to_clean:
            self.raw_to_clean[raw_key] = self.next_clean_id
            self.next_clean_id += 1

        return self.raw_to_clean[raw_key]


def get_class_name(model, cls_id):
    names = getattr(model, "names", {})
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), f"class_{cls_id}"))
        return str(names[int(cls_id)])
    except Exception:
        return f"class_{cls_id}"


def is_military_target(class_name):
    name = str(class_name).lower()
    return any(k in name for k in MILITARY_KEYWORDS)


def direction_from_delta(dx, dy):
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def threat_level(class_name, confidence):
    c = class_name.lower()

    if "tank" in c or "artillery" in c:
        return "HIGH"

    if "military_vehicle" in c or "military_truck" in c or "btr" in c or "bmp" in c or "apc" in c:
        return "MEDIUM"

    if confidence >= 0.70:
        return "MEDIUM"

    return "LOW"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=str(ACTIVE_DETECTOR_MODEL))
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--military_only", type=int, default=1)
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "model_data_exports"))

    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Set UAV_MODEL_PATH or pass --model."
        )

    source = args.source
    if "://" not in source and not source.isdigit():
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Video source not found: {source_path}")
        source = str(source_path.resolve())

    frame_csv = output_dir / "frame_by_frame_detections.csv"
    summary_csv = output_dir / "target_summary.csv"
    mission_csv = output_dir / "mission_report.csv"

    print("[INFO] Loading model:", model_path)
    model = YOLO(str(model_path))

    print("[INFO] Model classes:", getattr(model, "names", {}))
    print("[INFO] Source:", source)
    print("[INFO] Output folder:", output_dir)

    clean_mapper = CleanIDMapper()

    target_memory = {}

    start_time = time.time()
    frame_index = 0
    total_detections = 0
    accepted_detections = 0

    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame_index",
            "time_seconds",
            "target_id",
            "clean_track_id",
            "raw_track_id",
            "class_name",
            "confidence",
            "threat_level",
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "center_x",
            "center_y",
            "norm_center_x",
            "norm_center_y",
            "pixel_speed",
            "direction_deg",
            "status",
            "model",
            "tracker",
            "source_video",
        ])

        results_stream = model.track(
            source=source,
            device=0,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            persist=True,
            stream=True,
            verbose=False,
            classes=None,
        )

        for result in results_stream:
            frame_index += 1
            frame_time = time.time() - start_time

            if result.orig_img is not None:
                h, w = result.orig_img.shape[:2]
            else:
                h, w = 1, 1

            if result.boxes is None:
                continue

            for box in result.boxes:
                total_detections += 1

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = get_class_name(model, cls_id)

                if args.military_only == 1 and not is_military_target(class_name):
                    continue

                raw_track_id = None
                if box.id is not None:
                    raw_track_id = int(box.id[0].item())

                clean_track_id = clean_mapper.get_clean_id(raw_track_id)

                if clean_track_id is None:
                    continue

                target_id = f"Target_{clean_track_id}"

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                norm_center_x = center_x / float(w)
                norm_center_y = center_y / float(h)

                if target_id not in target_memory:
                    target_memory[target_id] = {
                        "target_id": target_id,
                        "clean_track_id": clean_track_id,
                        "raw_track_ids": set(),
                        "class_counts": {},
                        "first_frame": frame_index,
                        "last_frame": frame_index,
                        "first_seen_sec": frame_time,
                        "last_seen_sec": frame_time,
                        "total_seen_frames": 0,
                        "max_confidence": conf,
                        "avg_confidence_sum": 0.0,
                        "last_center": (center_x, center_y),
                        "last_speed": 0.0,
                        "max_speed": 0.0,
                        "last_direction_deg": 0.0,
                        "bbox_area_sum": 0.0,
                        "max_bbox_area": bbox_area,
                        "threat_level": threat_level(class_name, conf),
                    }

                mem = target_memory[target_id]

                old_center = mem["last_center"]
                dx = center_x - old_center[0]
                dy = center_y - old_center[1]

                pixel_speed = math.sqrt(dx * dx + dy * dy)
                direction_deg = direction_from_delta(dx, dy)

                mem["raw_track_ids"].add(raw_track_id)
                mem["class_counts"][class_name] = mem["class_counts"].get(class_name, 0) + 1
                mem["last_frame"] = frame_index
                mem["last_seen_sec"] = frame_time
                mem["total_seen_frames"] += 1
                mem["max_confidence"] = max(mem["max_confidence"], conf)
                mem["avg_confidence_sum"] += conf
                mem["last_center"] = (center_x, center_y)
                mem["last_speed"] = pixel_speed
                mem["max_speed"] = max(mem["max_speed"], pixel_speed)
                mem["last_direction_deg"] = direction_deg
                mem["bbox_area_sum"] += bbox_area
                mem["max_bbox_area"] = max(mem["max_bbox_area"], bbox_area)

                current_threat = threat_level(class_name, conf)
                if current_threat == "HIGH":
                    mem["threat_level"] = "HIGH"
                elif current_threat == "MEDIUM" and mem["threat_level"] != "HIGH":
                    mem["threat_level"] = "MEDIUM"

                accepted_detections += 1

                writer.writerow([
                    frame_index,
                    round(frame_time, 3),
                    target_id,
                    clean_track_id,
                    raw_track_id,
                    class_name,
                    round(conf, 4),
                    current_threat,
                    x1,
                    y1,
                    x2,
                    y2,
                    bbox_width,
                    bbox_height,
                    bbox_area,
                    center_x,
                    center_y,
                    round(norm_center_x, 6),
                    round(norm_center_y, 6),
                    round(pixel_speed, 3),
                    round(direction_deg, 3),
                    "tracked",
                    str(model_path),
                    args.tracker,
                    source,
                ])

            if frame_index % 50 == 0:
                print(f"[INFO] Processed frame {frame_index} | accepted detections: {accepted_detections}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "target_id",
            "clean_track_id",
            "raw_track_ids",
            "dominant_class",
            "threat_level",
            "first_frame",
            "last_frame",
            "first_seen_sec",
            "last_seen_sec",
            "duration_sec",
            "total_seen_frames",
            "max_confidence",
            "average_confidence",
            "last_center_x",
            "last_center_y",
            "last_pixel_speed",
            "max_pixel_speed",
            "last_direction_deg",
            "average_bbox_area",
            "max_bbox_area",
        ])

        for target_id, mem in target_memory.items():
            dominant_class = max(mem["class_counts"], key=mem["class_counts"].get)
            avg_conf = mem["avg_confidence_sum"] / max(1, mem["total_seen_frames"])
            avg_bbox_area = mem["bbox_area_sum"] / max(1, mem["total_seen_frames"])
            duration = mem["last_seen_sec"] - mem["first_seen_sec"]

            writer.writerow([
                target_id,
                mem["clean_track_id"],
                ";".join(str(x) for x in sorted(mem["raw_track_ids"]) if x is not None),
                dominant_class,
                mem["threat_level"],
                mem["first_frame"],
                mem["last_frame"],
                round(mem["first_seen_sec"], 3),
                round(mem["last_seen_sec"], 3),
                round(duration, 3),
                mem["total_seen_frames"],
                round(mem["max_confidence"], 4),
                round(avg_conf, 4),
                mem["last_center"][0],
                mem["last_center"][1],
                round(mem["last_speed"], 3),
                round(mem["max_speed"], 3),
                round(mem["last_direction_deg"], 3),
                round(avg_bbox_area, 2),
                mem["max_bbox_area"],
            ])

    mission_duration = time.time() - start_time
    total_targets = len(target_memory)
    high_threat_targets = sum(1 for m in target_memory.values() if m["threat_level"] == "HIGH")
    medium_threat_targets = sum(1 for m in target_memory.values() if m["threat_level"] == "MEDIUM")
    low_threat_targets = sum(1 for m in target_memory.values() if m["threat_level"] == "LOW")

    with open(mission_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["metric", "value"])
        writer.writerow(["source_video", source])
        writer.writerow(["model", str(model_path)])
        writer.writerow(["tracker", args.tracker])
        writer.writerow(["confidence_threshold", args.conf])
        writer.writerow(["image_size", args.imgsz])
        writer.writerow(["processed_frames", frame_index])
        writer.writerow(["mission_duration_seconds", round(mission_duration, 3)])
        writer.writerow(["total_raw_detections", total_detections])
        writer.writerow(["accepted_military_detections", accepted_detections])
        writer.writerow(["unique_targets_tracked", total_targets])
        writer.writerow(["high_threat_targets", high_threat_targets])
        writer.writerow(["medium_threat_targets", medium_threat_targets])
        writer.writerow(["low_threat_targets", low_threat_targets])

    print("")
    print("[DONE] Data extraction complete.")
    print("[OUTPUT]", frame_csv)
    print("[OUTPUT]", summary_csv)
    print("[OUTPUT]", mission_csv)


if __name__ == "__main__":
    main()
