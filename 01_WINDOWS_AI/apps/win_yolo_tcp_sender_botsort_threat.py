import argparse
import json
import socket
import struct
import subprocess
import time

import cv2
from ultralytics import YOLO


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
    "armoured",
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


def normalize_name(name):
    return str(name).lower().strip().replace(" ", "_").replace("-", "_")


def resolve_source(source: str) -> str:
    if "youtube.com" in source or "youtu.be" in source:
        print("Extracting YouTube stream URL with yt-dlp...")
        url = subprocess.check_output(
            ["yt-dlp", "-g", "-f", "best", source],
            text=True
        ).strip().splitlines()[0]
        return url
    return source


def get_yolo_class_name(model, cls_id):
    names = getattr(model, "names", {})
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), f"class_{cls_id}"))
        return str(names[int(cls_id)])
    except Exception:
        return f"class_{cls_id}"


def is_military_target(class_name):
    name = normalize_name(class_name)
    return any(k in name for k in MILITARY_KEYWORDS)


def get_platform_category(class_name):
    """
    Converts model class names into operational platform categories.
    This is important for C3 integration because the C3 system should not only know
    the raw neural-network class, but also the tactical category of the object.
    """
    c = normalize_name(class_name)

    if "artillery" in c:
        return "artillery_system"

    if "tank" in c:
        return "main_battle_tank"

    if "btr" in c or "apc" in c:
        return "armored_personnel_carrier"

    if "bmp" in c:
        return "infantry_fighting_vehicle"

    if "armored" in c or "armoured" in c:
        return "armored_vehicle"

    if "military_vehicle" in c:
        return "military_vehicle"

    if "military_truck" in c or "truck" in c:
        return "military_logistics_vehicle"

    if "military" in c:
        return "generic_military_vehicle"

    return "unknown_military_object"


def get_base_threat_score(class_name):
    """
    Base threat reflects the tactical danger of the platform type before confidence
    and movement/persistence adjustments.
    """
    c = normalize_name(class_name)

    if "artillery" in c:
        return 96

    if "tank" in c:
        return 95

    if "bmp" in c:
        return 86

    if "btr" in c or "apc" in c:
        return 82

    if "armored" in c or "armoured" in c:
        return 80

    if "military_vehicle" in c:
        return 78

    if "military_truck" in c or "truck" in c:
        return 62

    if "military" in c:
        return 65

    return 45


def compute_threat_score(class_name, confidence, bbox_area, frame_area):
    """
    Threat score is not only the class name.
    It also considers confidence and apparent target size.

    Larger bounding box area can suggest proximity or visual importance.
    This is image-space estimation only, not real distance.
    """
    base = get_base_threat_score(class_name)

    confidence_factor = max(0.30, min(1.00, confidence))

    if frame_area > 0:
        area_ratio = bbox_area / float(frame_area)
    else:
        area_ratio = 0.0

    proximity_bonus = 0

    if area_ratio > 0.10:
        proximity_bonus = 8
    elif area_ratio > 0.05:
        proximity_bonus = 5
    elif area_ratio > 0.02:
        proximity_bonus = 3

    score = (base * confidence_factor) + proximity_bonus

    score = max(0, min(100, score))

    return round(score, 2), base, round(area_ratio, 6)


def get_threat_level(threat_score):
    if threat_score >= 90:
        return "CRITICAL"

    if threat_score >= 75:
        return "HIGH"

    if threat_score >= 45:
        return "MEDIUM"

    return "LOW"


def get_alert_priority(threat_level):
    if threat_level == "CRITICAL":
        return 1

    if threat_level == "HIGH":
        return 2

    if threat_level == "MEDIUM":
        return 3

    return 4


def send_packet(sock, frame, detections, seq):
    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return False

    jpeg_bytes = jpeg.tobytes()

    header = {
        "seq": seq,
        "timestamp": time.time(),
        "source_width": frame.shape[1],
        "source_height": frame.shape[0],
        "jpeg_size": len(jpeg_bytes),
        "detections": detections,
    }

    header_bytes = json.dumps(header).encode("utf-8")

    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(jpeg_bytes)

    return True


def connect_to_bridge(ip, port):
    while True:
        try:
            print(f"Connecting to ROS2 bridge at {ip}:{port} ...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            print("Connected.")
            return sock
        except Exception as e:
            print(f"[WAIT] Could not connect: {e}")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Ubuntu VM IP")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--source", default="vehicles.mp4")
    parser.add_argument("--model", default="military_kaggle_v1.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--send_width", type=int, default=960)
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--military_only", type=int, default=1)
    parser.add_argument("--tracker", default="botsort.yaml")
    args = parser.parse_args()

    source = resolve_source(args.source)

    print("Opening source:", source)
    print("Loading YOLO model:", args.model)

    model = YOLO(args.model)

    print("Model classes:", getattr(model, "names", {}))
    print("Tracker:", args.tracker)

    clean_mapper = CleanIDMapper()

    sock = connect_to_bridge(args.target, args.port)

    seq = 0
    last_log = time.time()

    print("Running Windows GPU YOLO TCP Sender with BoT-SORT + Clean IDs + Threat Engine.")
    print("This version sends threat_score, threat_level, platform_category, and alert_priority.")
    print("Press Q in the video window to quit.")

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
        seq += 1

        if args.stride > 1 and seq % args.stride != 0:
            continue

        frame = result.orig_img.copy()

        h, w = frame.shape[:2]

        if args.send_width > 0 and w != args.send_width:
            new_w = args.send_width
            new_h = int(h * new_w / w)
            scale_x = new_w / float(w)
            scale_y = new_h / float(h)
            frame = cv2.resize(frame, (new_w, new_h))
        else:
            scale_x = 1.0
            scale_y = 1.0

        source_h, source_w = frame.shape[:2]
        frame_area = source_h * source_w

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = get_yolo_class_name(model, cls_id)

                if args.military_only == 1 and not is_military_target(class_name):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w <= 2 or box_h <= 2:
                    continue

                bbox_area = box_w * box_h

                raw_track_id = None
                if box.id is not None:
                    raw_track_id = int(box.id[0].item())

                clean_track_id = clean_mapper.get_clean_id(raw_track_id)

                if clean_track_id is None:
                    target_id = "Target_unknown"
                    clean_track_id = -1
                    target_status = "DETECTED"
                else:
                    target_id = f"Target_{clean_track_id}"
                    target_status = "TRACKED"

                platform_category = get_platform_category(class_name)
                threat_score, base_threat, image_area_ratio = compute_threat_score(
                    class_name=class_name,
                    confidence=conf,
                    bbox_area=bbox_area,
                    frame_area=frame_area,
                )
                threat_level = get_threat_level(threat_score)
                alert_priority = get_alert_priority(threat_level)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "uav_id": "uav_1",
                    "source": "windows_gpu_yolo_tcp_botsort_threat",
                    "model": args.model,

                    "class": class_name,
                    "class_name": class_name,
                    "final_class": class_name,
                    "platform_type": class_name,
                    "platform_category": platform_category,

                    "label": f"{target_id} {class_name}",
                    "name": target_id,
                    "class_id": cls_id,
                    "cls": cls_id,
                    "confidence": round(conf, 4),

                    "track_id": clean_track_id,
                    "raw_id": raw_track_id,
                    "raw_track_id": raw_track_id,
                    "clean_track_id": clean_track_id,
                    "target_id": target_id,
                    "source_id": target_id,
                    "target_status": target_status,
                    "status": target_status.lower(),
                    "is_target": True,

                    "threat_score": threat_score,
                    "base_threat": base_threat,
                    "threat_level": threat_level,
                    "alert_priority": alert_priority,
                    "image_area_ratio": image_area_ratio,

                    "bbox": [x1, y1, x2, y2],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, box_w, box_h],

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x": x1,
                    "y": y1,
                    "w": box_w,
                    "h": box_h,

                    "center_x": cx,
                    "center_y": cy,
                    "cx": cx,
                    "cy": cy,
                    "norm_center_x": round(cx / source_w, 6),
                    "norm_center_y": round(cy / source_h, 6),

                    "bbox_width": box_w,
                    "bbox_height": box_h,
                    "bbox_area": bbox_area,

                    "source_width": source_w,
                    "source_height": source_h,
                    "timestamp": time.time(),

                    "priority": "high" if threat_level in ["CRITICAL", "HIGH"] else "medium",
                })

        try:
            send_packet(sock, frame, detections, seq)
        except Exception as e:
            print(f"[WARN] TCP send failed: {e}")
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_to_bridge(args.target, args.port)
            continue

        if args.show:
            annotated = result.plot()

            if args.send_width > 0 and annotated.shape[1] != args.send_width:
                ah, aw = annotated.shape[:2]
                new_w = args.send_width
                new_h = int(ah * new_w / aw)
                annotated = cv2.resize(annotated, (new_w, new_h))

            cv2.imshow("Windows YOLO BoT-SORT Threat Sender - Press Q", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        now = time.time()
        if now - last_log > 2.0:
            critical = sum(1 for d in detections if d.get("threat_level") == "CRITICAL")
            high = sum(1 for d in detections if d.get("threat_level") == "HIGH")
            medium = sum(1 for d in detections if d.get("threat_level") == "MEDIUM")

            print(
                f"Sent frame {seq} | detections={len(detections)} "
                f"| CRITICAL={critical} HIGH={high} MEDIUM={medium} "
                f"| clean_targets={clean_mapper.raw_to_clean} "
                f"| target={args.target}:{args.port}"
            )
            last_log = now

    try:
        sock.close()
    except Exception:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()