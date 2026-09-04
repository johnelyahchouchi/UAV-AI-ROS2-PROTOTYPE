

#!/usr/bin/env python3

import csv
import json
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.csv_safe import sanitize_csv_row


WINDOW_NAME = "UAV Tank Type Timeline Dashboard V1"

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
DARK = (20, 20, 20)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

TARGET_TIMEOUT = 2.0
MAX_EVENTS = 18


def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def pretty_platform(name):
    n = normalize_name(name)

    names = {
        "tank_t72": "T-72 Main Battle Tank",
        "tank_t80": "T-80 Main Battle Tank",
        "tank_t90": "T-90 Main Battle Tank",
        "tank_m1_abrams": "M1 Abrams Main Battle Tank",
        "tank_leopard2": "Leopard 2 Main Battle Tank",
        "tank_merkava": "Merkava Main Battle Tank",
        "tank_challenger2": "Challenger 2 Main Battle Tank",
        "tank_leclerc": "Leclerc Main Battle Tank",
        "tank_unknown": "Tank Unknown Type",
        "ifv_bmp": "BMP Infantry Fighting Vehicle",
        "military_truck": "Military Truck",
        "armored_truck": "Armored Truck",
        "artillery": "Artillery System",
        "unknown_platform": "Unknown Platform",
    }

    return names.get(n, n.replace("_", " ").title())


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


def get_target_id(det):
    target_id = det.get("target_id", det.get("name", ""))

    if target_id and str(target_id) not in ["", "None", "unknown", "Target_unknown"]:
        return str(target_id)

    track_id = det.get("clean_track_id", det.get("track_id", det.get("raw_id", None)))

    if track_id is None or str(track_id) in ["", "None", "unknown", "-1"]:
        return "Target_unknown"

    return f"Target_{track_id}"


def get_platform_type(det):
    for key in ["platform_type", "final_class", "display_class", "class_name", "class"]:
        value = det.get(key, None)

        if value is not None and str(value).strip() != "":
            return str(value)

    return "unknown_platform"


def get_platform_pretty(det):
    value = det.get("platform_pretty", None)

    if value is not None and str(value).strip() != "":
        return str(value)

    return pretty_platform(get_platform_type(det))


def get_confidence(det):
    for key in ["platform_confidence", "confidence", "yolo_confidence"]:
        if key in det:
            try:
                return float(det[key])
            except Exception:
                pass

    return 0.0


def get_threat_level(det):
    return str(det.get("threat_level", "UNKNOWN")).upper()


def get_threat_score(det):
    try:
        return float(det.get("threat_score", 0.0))
    except Exception:
        return 0.0


def is_tank_or_military_platform(platform_type, platform_pretty):
    text = f"{platform_type} {platform_pretty}".lower()

    keywords = [
        "tank",
        "t-72",
        "t-80",
        "t-90",
        "abrams",
        "leopard",
        "merkava",
        "challenger",
        "leclerc",
        "bmp",
        "artillery",
        "truck",
    ]

    return any(k in text for k in keywords)


def draw_text(img, text, x, y, color=WHITE, scale=0.5, thickness=1):
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class UAVTankTypeTimelineDashboard(Node):
    def __init__(self):
        super().__init__("uav_tank_type_timeline_dashboard_v1")

        self.start_time = time.time()
        self.last_msg_time = 0.0

        self.active_targets = OrderedDict()
        self.timeline_events = deque(maxlen=MAX_EVENTS)

        self.last_seen_signature = {}

        self.output_dir = Path.home() / "uav_demo_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "tank_type_timeline_events.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow(sanitize_csv_row([
            "timestamp",
            "mission_time_sec",
            "target_id",
            "platform_type",
            "platform_pretty",
            "platform_confidence",
            "threat_score",
            "threat_level",
            "event_type",
        ]))

        self.create_subscription(
            String,
            "/uav_1/coco_detections",
            self.detections_callback,
            10
        )

        self.timer = self.create_timer(0.10, self.render)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 720)

        self.get_logger().info("Tank Type Timeline Dashboard V1 started.")
        self.get_logger().info(f"Saving timeline CSV to: {self.csv_path}")

    def add_event(self, target_id, platform_type, platform_pretty, confidence, threat_score, threat_level, event_type):
        now = time.time()
        mission_time = now - self.start_time

        event = {
            "mission_time": mission_time,
            "clock": datetime.now().strftime("%H:%M:%S"),
            "target_id": target_id,
            "platform_type": platform_type,
            "platform_pretty": platform_pretty,
            "confidence": confidence,
            "threat_score": threat_score,
            "threat_level": threat_level,
            "event_type": event_type,
        }

        self.timeline_events.appendleft(event)

        self.csv_writer.writerow(sanitize_csv_row([
            datetime.now().isoformat(timespec="milliseconds"),
            round(mission_time, 3),
            target_id,
            platform_type,
            platform_pretty,
            round(confidence, 4),
            round(threat_score, 2),
            threat_level,
            event_type,
        ]))

        self.csv_file.flush()

    def detections_callback(self, msg):
        try:
            data = json.loads(msg.data)

            if isinstance(data, dict):
                detections = data.get("detections", [data])
            elif isinstance(data, list):
                detections = data
            else:
                detections = []

        except Exception as e:
            self.get_logger().error(f"Could not parse detections: {e}")
            return

        self.last_msg_time = time.time()
        now = time.time()

        for det in detections:
            target_id = get_target_id(det)
            platform_type = normalize_name(get_platform_type(det))
            platform_pretty = get_platform_pretty(det)
            confidence = get_confidence(det)
            threat_level = get_threat_level(det)
            threat_score = get_threat_score(det)

            if not is_tank_or_military_platform(platform_type, platform_pretty):
                continue

            previous = self.active_targets.get(target_id, None)

            self.active_targets[target_id] = {
                "first_seen": previous["first_seen"] if previous else now,
                "last_seen": now,
                "target_id": target_id,
                "platform_type": platform_type,
                "platform_pretty": platform_pretty,
                "confidence": confidence,
                "threat_level": threat_level,
                "threat_score": threat_score,
            }

            signature = f"{target_id}|{platform_type}|{threat_level}"

            if target_id not in self.last_seen_signature:
                self.last_seen_signature[target_id] = signature
                self.add_event(
                    target_id,
                    platform_type,
                    platform_pretty,
                    confidence,
                    threat_score,
                    threat_level,
                    "FIRST_SEEN",
                )

            elif self.last_seen_signature[target_id] != signature:
                self.last_seen_signature[target_id] = signature
                self.add_event(
                    target_id,
                    platform_type,
                    platform_pretty,
                    confidence,
                    threat_score,
                    threat_level,
                    "UPDATED",
                )

    def clean_old_targets(self):
        now = time.time()

        old_ids = [
            target_id for target_id, data in self.active_targets.items()
            if now - data["last_seen"] > TARGET_TIMEOUT
        ]

        for target_id in old_ids:
            self.active_targets.pop(target_id, None)

    def render(self):
        self.clean_old_targets()

        canvas = np.zeros((720, 1200, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 12)

        now = time.time()
        mission_time = now - self.start_time

        active_count = len(self.active_targets)

        critical_count = sum(
            1 for t in self.active_targets.values()
            if t["threat_level"] == "CRITICAL"
        )

        high_count = sum(
            1 for t in self.active_targets.values()
            if t["threat_level"] == "HIGH"
        )

        draw_text(canvas, "UAV TANK TYPE TIMELINE DASHBOARD", 25, 38, CYAN, 0.85, 2)
        draw_text(canvas, f"Mission Time: {mission_time:0.1f}s", 25, 70, WHITE, 0.5, 1)
        draw_text(canvas, f"Active Platforms: {active_count}", 230, 70, WHITE, 0.5, 1)
        draw_text(canvas, f"Critical: {critical_count}", 430, 70, RED, 0.5, 1)
        draw_text(canvas, f"High: {high_count}", 560, 70, ORANGE, 0.5, 1)
        draw_text(canvas, "CSV: ~/uav_demo_outputs/tank_type_timeline_events.csv", 720, 70, GRAY, 0.42, 1)

        cv2.rectangle(canvas, (25, 100), (1175, 330), DARK, -1)
        cv2.rectangle(canvas, (25, 100), (1175, 330), GRAY, 1)

        draw_text(canvas, "CURRENTLY VISIBLE TARGETS", 40, 130, WHITE, 0.58, 2)

        header_y = 165
        draw_text(canvas, "Target", 45, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Platform Type", 180, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Conf", 610, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Threat", 710, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Score", 850, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "First Seen", 950, header_y, CYAN, 0.46, 1)

        y = 195

        active_sorted = sorted(
            self.active_targets.values(),
            key=lambda t: t["threat_score"],
            reverse=True
        )

        for target in active_sorted[:6]:
            color = threat_color(target["threat_level"])
            first_seen_relative = target["first_seen"] - self.start_time

            draw_text(canvas, target["target_id"], 45, y, WHITE, 0.46, 1)
            draw_text(canvas, target["platform_pretty"], 180, y, color, 0.46, 2)
            draw_text(canvas, f"{target['confidence']:.2f}", 610, y, WHITE, 0.46, 1)
            draw_text(canvas, target["threat_level"], 710, y, color, 0.46, 2)
            draw_text(canvas, f"{target['threat_score']:.1f}", 850, y, color, 0.46, 2)
            draw_text(canvas, f"{first_seen_relative:.1f}s", 950, y, WHITE, 0.46, 1)

            y += 28

        cv2.rectangle(canvas, (25, 355), (1175, 690), DARK, -1)
        cv2.rectangle(canvas, (25, 355), (1175, 690), GRAY, 1)

        draw_text(canvas, "MISSION TIMELINE - WHAT TYPE WAS SEEN AT WHAT TIME", 40, 385, WHITE, 0.58, 2)

        header_y = 420
        draw_text(canvas, "Time", 45, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Event", 145, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Target", 275, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Detected Type", 410, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Conf", 830, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Threat", 920, header_y, CYAN, 0.46, 1)
        draw_text(canvas, "Score", 1060, header_y, CYAN, 0.46, 1)

        y = 450

        for event in list(self.timeline_events):
            color = threat_color(event["threat_level"])

            draw_text(canvas, f"{event['mission_time']:.1f}s", 45, y, WHITE, 0.44, 1)
            draw_text(canvas, event["event_type"], 145, y, GRAY, 0.44, 1)
            draw_text(canvas, event["target_id"], 275, y, WHITE, 0.44, 1)
            draw_text(canvas, event["platform_pretty"], 410, y, color, 0.44, 2)
            draw_text(canvas, f"{event['confidence']:.2f}", 830, y, WHITE, 0.44, 1)
            draw_text(canvas, event["threat_level"], 920, y, color, 0.44, 2)
            draw_text(canvas, f"{event['threat_score']:.1f}", 1060, y, color, 0.44, 2)

            y += 23

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            rclpy.shutdown()

    def close_files(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)

    node = UAVTankTypeTimelineDashboard()

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
