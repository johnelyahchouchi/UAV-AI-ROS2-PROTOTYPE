

#!/usr/bin/env python3

import csv
import json
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.csv_safe import sanitize_csv_row


TARGET_COLORS = [
    (255, 255, 0),
    (255, 0, 255),
    (0, 165, 255),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 255),
]

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
DARK = (20, 20, 20)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)

MILITARY_ONLY = True
MIN_TARGET_CONFIDENCE = 0.20
TARGET_TIMEOUT_SECONDS = 1.2
MAX_TRAIL_POINTS = 45

EXCLUDED_CLASSES = {
    "car",
    "truck",
    "person",
    "people",
    "pedestrian",
    "civilian",
    "civilian_vehicle",
    "bus",
    "van",
    "motor",
    "motorcycle",
    "bicycle",
}

ALLOWED_EXACT_CLASSES = {
    "military_tank",
    "tank",
    "military_vehicle",
    "armored_vehicle",
    "military_truck",
    "military_artillery",
    "artillery",
    "btr",
    "bmp",
    "apc",
    "possible_tank_or_armored_vehicle",
}

ALLOWED_KEYWORDS = [
    "military",
    "tank",
    "armored",
    "armoured",
    "artillery",
    "btr",
    "bmp",
    "apc",
]


def normalize_class_name(class_name):
    return str(class_name).lower().strip().replace(" ", "_").replace("-", "_")


def pretty_name(name):
    return normalize_class_name(name).replace("_", " ").title()


def is_allowed_target_class(class_name):
    c = normalize_class_name(class_name)

    if c in EXCLUDED_CLASSES:
        return False

    if "civilian" in c:
        return False

    if c in ALLOWED_EXACT_CLASSES:
        return True

    return any(keyword in c for keyword in ALLOWED_KEYWORDS)


def target_color(target_id):
    try:
        n = int(str(target_id).split("_")[-1])
    except Exception:
        n = 1

    return TARGET_COLORS[(n - 1) % len(TARGET_COLORS)]


def threat_color(level):
    level = str(level).upper()

    if level == "CRITICAL":
        return RED

    if level == "HIGH":
        return ORANGE

    if level == "MEDIUM":
        return YELLOW

    if level == "LOW":
        return GREEN

    return GRAY


def fallback_platform_category(class_name):
    c = normalize_class_name(class_name)

    if "artillery" in c:
        return "artillery_system"

    if "tank" in c:
        return "main_battle_tank"

    if "bmp" in c:
        return "infantry_fighting_vehicle"

    if "btr" in c or "apc" in c:
        return "armored_personnel_carrier"

    if "armored" in c or "armoured" in c:
        return "armored_vehicle"

    if "military_truck" in c or "truck" in c:
        return "military_logistics_vehicle"

    if "military_vehicle" in c:
        return "military_vehicle"

    if "military" in c:
        return "generic_military_vehicle"

    return "unknown_military_object"


def fallback_base_threat(class_name):
    c = normalize_class_name(class_name)

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


def fallback_threat_score(class_name, confidence):
    base = fallback_base_threat(class_name)
    confidence_factor = max(0.30, min(1.00, float(confidence)))
    score = base * confidence_factor
    return round(max(0, min(100, score)), 2)


def fallback_threat_level(score):
    if score >= 90:
        return "CRITICAL"

    if score >= 75:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


def draw_text_box(frame, text, x, y, color, scale=0.52, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = int(max(5, x))
    y = int(max(25, y))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x2 = min(frame.shape[1] - 5, x + tw + 16)
    y1 = max(5, y - th - baseline - 10)
    y2 = min(frame.shape[0] - 5, y + baseline + 8)

    cv2.rectangle(frame, (x, y1), (x2, y2), BLACK, -1)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x + 8, y), font, scale, WHITE, thickness)


def draw_badge(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x1 = int(x)
    y1 = int(y)
    x2 = min(frame.shape[1] - 5, x1 + tw + 18)
    y2 = min(frame.shape[0] - 5, y1 + th + 12)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + 8, y1 + th + 4), font, scale, BLACK, thickness)


def draw_fading_trail(frame, trail, color):
    trail = list(trail)

    if len(trail) < 2:
        return

    n = len(trail)

    for i in range(1, n):
        p1 = trail[i - 1]
        p2 = trail[i]

        alpha = i / float(n)
        thickness = max(1, int(1 + alpha * 3))
        faded_color = tuple(int(c * alpha + 40 * (1 - alpha)) for c in color)

        cv2.line(frame, p1, p2, faded_color, thickness)

    cv2.circle(frame, trail[-1], 5, color, -1)

    if len(trail) >= 6:
        p_old = trail[-6]
        p_new = trail[-1]

        dx = p_new[0] - p_old[0]
        dy = p_new[1] - p_old[1]

        if abs(dx) + abs(dy) > 8:
            start = (int(p_new[0] - dx * 0.35), int(p_new[1] - dy * 0.35))
            end = p_new
            cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.35)


class UAVCleanTargetDashboardV5(Node):
    def __init__(self):
        super().__init__("uav_clean_target_dashboard_v5")

        self.bridge = CvBridge()

        self.frame = None
        self.frame_stamp = None
        self.frame_index = 0

        self.latest_detections = []
        self.last_detection_time = 0.0

        self.target_memory = {}

        self.last_total_detections = 0
        self.last_accepted_detections = 0
        self.last_ignored_detections = 0
        self.last_ignored_classes = Counter()
        self.last_threat_counts = Counter()
        self.last_platform_counts = Counter()

        self.start_time = time.time()
        self.last_render_time = time.time()
        self.fps = 0.0

        self.output_dir = Path.home() / "uav_demo_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "uav_target_extraction_v5_threat.csv"
        self.summary_path = self.output_dir / "uav_target_summary_v5_threat.csv"

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp",
            "frame_index",
            "target_id",
            "raw_id",
            "track_id",
            "class_name",
            "platform_category",
            "confidence",
            "threat_score",
            "threat_level",
            "alert_priority",
            "base_threat",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "norm_center_x",
            "norm_center_y",
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "pixel_speed",
            "direction_deg",
            "status",
            "source"
        ])

        self.create_subscription(
            Image,
            "/uav_1/camera/image_raw",
            self.image_callback,
            10
        )

        self.create_subscription(
            String,
            "/uav_1/coco_detections",
            self.detections_callback,
            10
        )

        self.timer = self.create_timer(0.03, self.render)

        cv2.namedWindow("UAV ISR Dashboard V5 Threat Engine", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("UAV ISR Dashboard V5 Threat Engine", 1280, 720)

        self.get_logger().info("UAV ISR Dashboard V5 started.")
        self.get_logger().info("V5 = platform category + threat score + threat level visualization.")
        self.get_logger().info("VM tracking remains OFF. Windows BoT-SORT clean IDs are trusted.")
        self.get_logger().info(f"CSV extraction: {self.csv_path}")

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame_stamp = time.time()
            self.frame_index += 1
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def detections_callback(self, msg):
        try:
            data = json.loads(msg.data)

            if isinstance(data, dict):
                detections = data.get("detections", [data])
            elif isinstance(data, list):
                detections = data
            else:
                detections = []

            self.latest_detections = detections
            self.last_detection_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Detection JSON parse failed: {e}")
            self.latest_detections = []

    def get_bbox(self, det):
        if "bbox" in det:
            bbox = det["bbox"]

            if isinstance(bbox, list) and len(bbox) >= 4:
                return [int(float(v)) for v in bbox[:4]]

            if isinstance(bbox, dict):
                x1 = bbox.get("x1", bbox.get("xmin", None))
                y1 = bbox.get("y1", bbox.get("ymin", None))
                x2 = bbox.get("x2", bbox.get("xmax", None))
                y2 = bbox.get("y2", bbox.get("ymax", None))

                if None not in [x1, y1, x2, y2]:
                    return [
                        int(float(x1)),
                        int(float(y1)),
                        int(float(x2)),
                        int(float(y2)),
                    ]

        if all(k in det for k in ["x1", "y1", "x2", "y2"]):
            return [int(float(det[k])) for k in ["x1", "y1", "x2", "y2"]]

        return None

    def get_class_name(self, det):
        return str(
            det.get(
                "class_name",
                det.get(
                    "class",
                    det.get(
                        "final_class",
                        det.get("name", det.get("label", "unknown"))
                    )
                )
            )
        )

    def get_confidence(self, det):
        return float(det.get("confidence", det.get("conf", det.get("score", 0.0))))

    def get_raw_id(self, det):
        raw = det.get("raw_track_id", det.get("raw_id", det.get("tracker_id", "")))
        return str(raw)

    def get_track_id(self, det):
        track_id = det.get("clean_track_id", det.get("track_id", det.get("raw_id", None)))

        if track_id is None or str(track_id) in ["", "None", "unknown", "-1"]:
            return None

        try:
            return int(track_id)
        except Exception:
            return str(track_id)

    def get_target_id(self, det):
        target_id = det.get("target_id", det.get("name", ""))

        if target_id and str(target_id) not in ["", "None", "unknown", "Target_unknown"]:
            return str(target_id)

        track_id = self.get_track_id(det)

        if track_id is not None:
            return f"Target_{track_id}"

        return None

    def get_platform_category(self, det, class_name):
        return str(det.get("platform_category", fallback_platform_category(class_name)))

    def get_threat_score(self, det, class_name, confidence):
        if "threat_score" in det:
            try:
                return float(det["threat_score"])
            except Exception:
                pass

        return fallback_threat_score(class_name, confidence)

    def get_threat_level(self, det, threat_score):
        if "threat_level" in det:
            return str(det["threat_level"]).upper()

        return fallback_threat_level(threat_score)

    def get_alert_priority(self, det, threat_level):
        if "alert_priority" in det:
            try:
                return int(det["alert_priority"])
            except Exception:
                pass

        if threat_level == "CRITICAL":
            return 1

        if threat_level == "HIGH":
            return 2

        if threat_level == "MEDIUM":
            return 3

        return 4

    def expire_old_targets(self):
        now = time.time()

        old_targets = [
            target_id
            for target_id, mem in self.target_memory.items()
            if now - mem["last_seen_time"] > TARGET_TIMEOUT_SECONDS
        ]

        for target_id in old_targets:
            self.target_memory.pop(target_id, None)

    def update_target_memory(
        self,
        target_id,
        center,
        class_name,
        platform_category,
        confidence,
        threat_score,
        threat_level,
        alert_priority,
        bbox,
        raw_id,
        track_id,
        source
    ):
        now = time.time()

        if target_id not in self.target_memory:
            self.target_memory[target_id] = {
                "first_seen": datetime.now().isoformat(timespec="seconds"),
                "first_seen_time": now,
                "last_seen": datetime.now().isoformat(timespec="seconds"),
                "last_seen_time": now,
                "last_frame": self.frame_index,
                "last_center": center,
                "last_bbox": bbox,
                "last_confidence": confidence,
                "max_confidence": confidence,
                "class_name": class_name,
                "platform_category": platform_category,
                "threat_score": threat_score,
                "max_threat_score": threat_score,
                "threat_level": threat_level,
                "alert_priority": alert_priority,
                "trail": deque(maxlen=MAX_TRAIL_POINTS),
                "total_seen": 0,
                "last_speed": 0.0,
                "last_direction_deg": 0.0,
                "raw_id": raw_id,
                "track_id": track_id,
                "source": source,
            }

        mem = self.target_memory[target_id]

        old_center = mem.get("last_center", center)
        dx = center[0] - old_center[0]
        dy = center[1] - old_center[1]
        pixel_speed = math.sqrt(dx * dx + dy * dy)

        if pixel_speed > 0:
            direction_deg = math.degrees(math.atan2(dy, dx))
        else:
            direction_deg = mem.get("last_direction_deg", 0.0)

        mem["last_seen"] = datetime.now().isoformat(timespec="seconds")
        mem["last_seen_time"] = now
        mem["last_frame"] = self.frame_index
        mem["last_center"] = center
        mem["last_bbox"] = bbox
        mem["last_confidence"] = confidence
        mem["max_confidence"] = max(mem.get("max_confidence", confidence), confidence)
        mem["class_name"] = class_name
        mem["platform_category"] = platform_category
        mem["threat_score"] = threat_score
        mem["max_threat_score"] = max(mem.get("max_threat_score", threat_score), threat_score)
        mem["threat_level"] = threat_level
        mem["alert_priority"] = alert_priority
        mem["total_seen"] += 1
        mem["last_speed"] = pixel_speed
        mem["last_direction_deg"] = direction_deg
        mem["raw_id"] = raw_id
        mem["track_id"] = track_id
        mem["source"] = source

        mem["trail"].append((int(center[0]), int(center[1])))

        return pixel_speed, direction_deg

    def write_summary(self):
        try:
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "target_id",
                    "track_id",
                    "raw_id",
                    "class_name",
                    "platform_category",
                    "first_seen",
                    "last_seen",
                    "total_seen_frames",
                    "last_confidence",
                    "max_confidence",
                    "threat_level",
                    "threat_score",
                    "max_threat_score",
                    "alert_priority",
                    "last_center_x",
                    "last_center_y",
                    "last_pixel_speed",
                    "last_direction_deg",
                    "status",
                    "source"
                ])

                now = time.time()

                for target_id, mem in self.target_memory.items():
                    age_sec = now - mem["last_seen_time"]
                    status = "active" if age_sec <= TARGET_TIMEOUT_SECONDS else "lost"

                    writer.writerow(sanitize_csv_row([
                        target_id,
                        mem["track_id"],
                        mem["raw_id"],
                        mem["class_name"],
                        mem["platform_category"],
                        mem["first_seen"],
                        mem["last_seen"],
                        mem["total_seen"],
                        round(mem["last_confidence"], 4),
                        round(mem["max_confidence"], 4),
                        mem["threat_level"],
                        round(mem["threat_score"], 2),
                        round(mem["max_threat_score"], 2),
                        mem["alert_priority"],
                        round(mem["last_center"][0], 2),
                        round(mem["last_center"][1], 2),
                        round(mem["last_speed"], 2),
                        round(mem["last_direction_deg"], 2),
                        status,
                        mem["source"],
                    ]))
        except Exception:
            pass

    def draw_dashboard_panel(self, canvas, active_targets):
        h, w = canvas.shape[:2]

        panel_w = 390
        cv2.rectangle(canvas, (w - panel_w, 0), (w, h), DARK, -1)
        cv2.line(canvas, (w - panel_w, 0), (w - panel_w, h), WHITE, 1)

        x = w - panel_w + 18
        y = 35

        cv2.putText(canvas, "UAV ISR THREAT DASHBOARD", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, WHITE, 2)

        y += 30
        cv2.putText(canvas, "MODE: AI PERCEPTION + THREAT", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, CYAN, 1)

        y += 24
        cv2.putText(canvas, "VM Tracking: OFF", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)

        y += 22
        cv2.putText(canvas, "Threat engine: WINDOWS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)

        mission_time = int(time.time() - self.start_time)

        y += 32
        cv2.putText(canvas, f"Mission Time: {mission_time}s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1)

        y += 24
        cv2.putText(canvas, f"Frame: {self.frame_index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1)

        y += 24
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1)

        y += 24
        cv2.putText(canvas, f"Active Targets: {len(active_targets)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1)

        y += 24
        cv2.putText(canvas, f"Accepted: {self.last_accepted_detections} / Total: {self.last_total_detections}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

        y += 30
        cv2.putText(canvas, "THREAT COUNTS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)

        y += 24
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            color = threat_color(level)
            count = self.last_threat_counts.get(level, 0)
            cv2.putText(canvas, f"{level}: {count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
            y += 20

        y += 12
        cv2.putText(canvas, "ACTIVE TARGETS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)

        y += 25

        now = time.time()

        sorted_targets = sorted(
            self.target_memory.items(),
            key=lambda item: item[1]["alert_priority"]
        )

        for target_id, mem in sorted_targets[:7]:
            age_sec = now - mem["last_seen_time"]

            if age_sec > TARGET_TIMEOUT_SECONDS:
                continue

            color = target_color(target_id)
            tcolor = threat_color(mem["threat_level"])

            cv2.circle(canvas, (x + 8, y - 5), 6, color, -1)

            line1 = f"{target_id.replace('_', ' ')} | {pretty_name(mem['class_name'])}"
            line2 = f"{mem['threat_level']} | Score {mem['threat_score']:.1f} | Conf {mem['last_confidence'] * 100:.0f}%"
            line3 = f"{pretty_name(mem['platform_category'])}"
            line4 = f"Track {mem['track_id']} | Raw {mem['raw_id']} | Age {age_sec:.1f}s"

            cv2.putText(canvas, line1, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
            y += 17
            cv2.putText(canvas, line2, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tcolor, 1)
            y += 17
            cv2.putText(canvas, line3, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRAY, 1)
            y += 17
            cv2.putText(canvas, line4, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, GRAY, 1)
            y += 23

            if y > h - 105:
                break

        y = h - 85
        cv2.putText(canvas, "SYSTEM STATUS:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

        y += 22
        cv2.putText(canvas, "ROS2: ONLINE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1)

        y += 18
        cv2.putText(canvas, "TCP BRIDGE: RECEIVING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1)

        cv2.putText(canvas, "Press Q to close window", (x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, YELLOW, 1)

    def render(self):
        now = time.time()
        dt = now - self.last_render_time
        self.last_render_time = now

        if dt > 0:
            self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)

        self.expire_old_targets()

        if self.frame is None:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "Waiting for /uav_1/camera/image_raw...",
                (60, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                WHITE,
                2
            )
            cv2.imshow("UAV ISR Dashboard V5 Threat Engine", canvas)
            cv2.waitKey(1)
            return

        frame = self.frame.copy()
        raw_h, raw_w = frame.shape[:2]

        display_w = 890
        display_h = int(raw_h * (display_w / raw_w))
        display_h = min(display_h, 720)

        frame = cv2.resize(frame, (display_w, display_h))

        sx = display_w / float(raw_w)
        sy = display_h / float(raw_h)

        active_targets = set()

        total_detections = 0
        accepted_detections = 0
        ignored_detections = 0
        ignored_classes = Counter()
        threat_counts = Counter()
        platform_counts = Counter()

        for det in self.latest_detections:
            bbox = self.get_bbox(det)

            if bbox is None:
                continue

            class_name_original = self.get_class_name(det)
            class_name = normalize_class_name(class_name_original)
            confidence = self.get_confidence(det)

            total_detections += 1

            if confidence < MIN_TARGET_CONFIDENCE:
                ignored_detections += 1
                ignored_classes[f"{class_name}_low_conf"] += 1
                continue

            if MILITARY_ONLY and not is_allowed_target_class(class_name):
                ignored_detections += 1
                ignored_classes[class_name] += 1
                continue

            target_id = self.get_target_id(det)

            if target_id is None:
                ignored_detections += 1
                ignored_classes["no_track_id"] += 1
                continue

            platform_category = self.get_platform_category(det, class_name)
            threat_score = self.get_threat_score(det, class_name, confidence)
            threat_level = self.get_threat_level(det, threat_score)
            alert_priority = self.get_alert_priority(det, threat_level)
            base_threat = det.get("base_threat", fallback_base_threat(class_name))

            accepted_detections += 1
            threat_counts[threat_level] += 1
            platform_counts[platform_category] += 1

            x1, y1, x2, y2 = bbox

            x1 = int(x1 * sx)
            x2 = int(x2 * sx)
            y1 = int(y1 * sy)
            y2 = int(y2 * sy)

            x1 = max(0, min(display_w - 1, x1))
            x2 = max(0, min(display_w - 1, x2))
            y1 = max(0, min(display_h - 1, y1))
            y2 = max(0, min(display_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            raw_id = self.get_raw_id(det)
            track_id = self.get_track_id(det)
            source = str(det.get("source", ""))

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center = (cx, cy)

            bbox_scaled = [x1, y1, x2, y2]

            pixel_speed, direction_deg = self.update_target_memory(
                target_id=target_id,
                center=center,
                class_name=class_name,
                platform_category=platform_category,
                confidence=confidence,
                threat_score=threat_score,
                threat_level=threat_level,
                alert_priority=alert_priority,
                bbox=bbox_scaled,
                raw_id=raw_id,
                track_id=track_id,
                source=source,
            )

            active_targets.add(target_id)

            color = target_color(target_id)
            tcolor = threat_color(threat_level)

            cv2.rectangle(frame, (x1, y1), (x2, y2), tcolor, 3)

            label = (
                f"{target_id.replace('_', ' ')} | "
                f"{pretty_name(class_name)} | "
                f"{threat_level} {threat_score:.1f} | "
                f"{confidence * 100:.0f}%"
            )

            draw_text_box(frame, label, x1, y1 - 12, tcolor, scale=0.50, thickness=2)

            draw_badge(frame, threat_level, x1, y2 + 8, tcolor)
            draw_badge(frame, f"Score {threat_score:.1f}", x1 + 95, y2 + 8, color)

            trail = self.target_memory[target_id]["trail"]
            draw_fading_trail(frame, trail, color)

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_area = bbox_w * bbox_h

            norm_cx = cx / float(display_w)
            norm_cy = cy / float(display_h)

            timestamp = datetime.now().isoformat(timespec="milliseconds")

            self.csv_writer.writerow(sanitize_csv_row([
                timestamp,
                self.frame_index,
                target_id,
                raw_id,
                track_id,
                class_name,
                platform_category,
                round(confidence, 4),
                round(threat_score, 2),
                threat_level,
                alert_priority,
                base_threat,
                x1,
                y1,
                x2,
                y2,
                round(cx, 2),
                round(cy, 2),
                round(norm_cx, 5),
                round(norm_cy, 5),
                bbox_w,
                bbox_h,
                bbox_area,
                round(pixel_speed, 3),
                round(direction_deg, 3),
                "active",
                source,
            ]))

        self.last_total_detections = total_detections
        self.last_accepted_detections = accepted_detections
        self.last_ignored_detections = ignored_detections
        self.last_ignored_classes = ignored_classes
        self.last_threat_counts = threat_counts
        self.last_platform_counts = platform_counts

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:display_h, :display_w] = frame

        self.draw_dashboard_panel(canvas, active_targets)

        cv2.imshow("UAV ISR Dashboard V5 Threat Engine", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            rclpy.shutdown()

        if self.frame_index % 30 == 0:
            self.csv_file.flush()
            self.write_summary()

    def close_files(self):
        try:
            self.write_summary()
        except Exception:
            pass

        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)

    node = UAVCleanTargetDashboardV5()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_files()
        cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == "__main__":
    main()
