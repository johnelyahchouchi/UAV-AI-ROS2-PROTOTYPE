
#!/usr/bin/env python3

import csv
import json
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import OUTPUTS_DIR


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (20, 20, 20)
GRAY = (150, 150, 150)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

WINDOW_NAME = "UAV Threat Analytics Dashboard V2"

HISTORY_LEN = 180
TARGET_TIMEOUT = 1.5
LONG_MEMORY_TIMEOUT = 10.0


def normalize_name(name):
    return str(name).lower().strip().replace(" ", "_").replace("-", "_")


def pretty_name(name):
    return normalize_name(name).replace("_", " ").title()


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


def target_color(target_id):
    colors = [
        (255, 255, 0),
        (255, 0, 255),
        (0, 165, 255),
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 255),
    ]

    try:
        n = int(str(target_id).split("_")[-1])
    except Exception:
        n = 1

    return colors[(n - 1) % len(colors)]


def get_target_id(det):
    target_id = det.get("target_id", det.get("name", ""))

    if target_id and str(target_id) not in ["", "None", "unknown", "Target_unknown"]:
        return str(target_id)

    track_id = det.get("clean_track_id", det.get("track_id", det.get("raw_id", None)))

    if track_id is None or str(track_id) in ["", "None", "unknown", "-1"]:
        return None

    return f"Target_{track_id}"


def get_class_name(det):
    return str(
        det.get(
            "class_name",
            det.get(
                "class",
                det.get(
                    "final_class",
                    det.get("label", "unknown")
                )
            )
        )
    )


def get_confidence(det):
    try:
        return float(det.get("confidence", det.get("conf", 0.0)))
    except Exception:
        return 0.0


def get_center(det):
    if "center_x" in det and "center_y" in det:
        return float(det["center_x"]), float(det["center_y"])

    if "cx" in det and "cy" in det:
        return float(det["cx"]), float(det["cy"])

    bbox = det.get("bbox", None)

    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = map(float, bbox[:4])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return None, None


def get_norm_center(det):
    if "norm_center_x" in det and "norm_center_y" in det:
        try:
            return float(det["norm_center_x"]), float(det["norm_center_y"])
        except Exception:
            pass

    cx, cy = get_center(det)

    if cx is None or cy is None:
        return 0.0, 0.0

    source_w = float(det.get("source_width", 1))
    source_h = float(det.get("source_height", 1))

    if source_w <= 0 or source_h <= 0:
        return 0.0, 0.0

    return cx / source_w, cy / source_h


def fallback_threat_score(class_name, confidence):
    c = normalize_name(class_name)

    if "artillery" in c:
        base = 96
    elif "tank" in c:
        base = 95
    elif "bmp" in c:
        base = 86
    elif "btr" in c or "apc" in c:
        base = 82
    elif "armored" in c or "armoured" in c:
        base = 80
    elif "military_vehicle" in c:
        base = 78
    elif "military_truck" in c or "truck" in c:
        base = 62
    elif "military" in c:
        base = 65
    else:
        base = 45

    score = base * max(0.30, min(1.0, confidence))
    return round(max(0, min(100, score)), 2)


def fallback_threat_level(score):
    if score >= 90:
        return "CRITICAL"

    if score >= 75:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


def get_threat_score(det, class_name, confidence):
    if "threat_score" in det:
        try:
            return float(det["threat_score"])
        except Exception:
            pass

    return fallback_threat_score(class_name, confidence)


def get_threat_level(det, threat_score):
    if "threat_level" in det:
        return str(det["threat_level"]).upper()

    return fallback_threat_level(threat_score)


def get_alert_priority(det, threat_level):
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


def get_platform_category(det, class_name):
    if "platform_category" in det:
        return str(det["platform_category"])

    c = normalize_name(class_name)

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


def draw_panel_title(img, title, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), DARK, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)
    cv2.putText(img, title, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


def draw_line_chart(img, title, values, x, y, w, h, color, max_value=None, suffix=""):
    draw_panel_title(img, title, x, y, w, h)

    plot_x = x + 45
    plot_y = y + 45
    plot_w = w - 65
    plot_h = h - 65

    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (35, 35, 35), -1)
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), GRAY, 1)

    if not values:
        cv2.putText(img, "Waiting for data", (plot_x + 20, plot_y + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        return

    vals = list(values)[-HISTORY_LEN:]

    if max_value is None:
        max_value = max(vals) if max(vals) > 0 else 1.0

    points = []

    for i, v in enumerate(vals):
        px = int(plot_x + (i / max(1, len(vals) - 1)) * plot_w)
        py = int(plot_y + plot_h - (v / max(1e-6, max_value)) * plot_h)
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(img, points[i - 1], points[i], color, 2)

    latest = vals[-1]
    cv2.circle(img, points[-1], 4, color, -1)

    cv2.putText(
        img,
        f"{latest:.2f}{suffix}",
        (plot_x + plot_w - 105, plot_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


def draw_multi_line_chart(img, title, series_dict, x, y, w, h):
    draw_panel_title(img, title, x, y, w, h)

    plot_x = x + 45
    plot_y = y + 45
    plot_w = w - 65
    plot_h = h - 65

    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (35, 35, 35), -1)
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), GRAY, 1)

    colors = {
        "CRITICAL": RED,
        "HIGH": ORANGE,
        "MEDIUM": YELLOW,
        "LOW": GREEN,
    }

    max_value = 1

    for values in series_dict.values():
        if values:
            max_value = max(max_value, max(values))

    legend_y = y + 25

    for label, values in series_dict.items():
        color = colors.get(label, WHITE)

        cv2.putText(img, label, (x + w - 235, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        legend_y += 17

        vals = list(values)[-HISTORY_LEN:]

        if len(vals) < 2:
            continue

        points = []

        for i, v in enumerate(vals):
            px = int(plot_x + (i / max(1, len(vals) - 1)) * plot_w)
            py = int(plot_y + plot_h - (v / max_value) * plot_h)
            points.append((px, py))

        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], color, 2)


def draw_tactical_map(img, target_memory, x, y, w, h):
    draw_panel_title(img, "Upper-view Live Tactical Map", x, y, w, h)

    map_x = x + 35
    map_y = y + 45
    map_w = w - 70
    map_h = h - 75

    cv2.rectangle(img, (map_x, map_y), (map_x + map_w, map_y + map_h), (30, 30, 30), -1)
    cv2.rectangle(img, (map_x, map_y), (map_x + map_w, map_y + map_h), CYAN, 1)

    cv2.line(img, (map_x + map_w // 2, map_y), (map_x + map_w // 2, map_y + map_h), (60, 60, 60), 1)
    cv2.line(img, (map_x, map_y + map_h // 2), (map_x + map_w, map_y + map_h // 2), (60, 60, 60), 1)

    cv2.putText(img, "Image-space tactical view", (map_x + 10, map_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    now = time.time()

    for target_id, mem in target_memory.items():
        if now - mem["last_seen_time"] > TARGET_TIMEOUT:
            continue

        nx, ny = mem["norm_center"]
        px = int(map_x + nx * map_w)
        py = int(map_y + ny * map_h)

        color = threat_color(mem["threat_level"])

        trail = list(mem["norm_trail"])

        for i in range(1, len(trail)):
            p1 = trail[i - 1]
            p2 = trail[i]

            x1 = int(map_x + p1[0] * map_w)
            y1 = int(map_y + p1[1] * map_h)
            x2 = int(map_x + p2[0] * map_w)
            y2 = int(map_y + p2[1] * map_h)

            cv2.line(img, (x1, y1), (x2, y2), color, 2)

        cv2.circle(img, (px, py), 8, color, -1)
        cv2.putText(img, target_id.replace("_", " "), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
        cv2.putText(img, mem["threat_level"], (px + 8, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)


def draw_target_lifetime_bars(img, target_memory, x, y, w, h):
    draw_panel_title(img, "Target Lifetime and Priority", x, y, w, h)

    now = time.time()

    active = [
        (tid, mem)
        for tid, mem in target_memory.items()
        if now - mem["last_seen_time"] <= TARGET_TIMEOUT
    ]

    active = sorted(active, key=lambda item: item[1]["alert_priority"])[:8]

    if not active:
        cv2.putText(img, "No active targets yet", (x + 20, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        return

    max_duration = max(1.0, max(now - mem["first_seen_time"] for _, mem in active))

    start_y = y + 55

    for idx, (target_id, mem) in enumerate(active):
        ty = start_y + idx * 30

        duration = now - mem["first_seen_time"]
        bar_w = int((duration / max_duration) * (w - 185))
        color = threat_color(mem["threat_level"])

        cv2.putText(img, target_id.replace("_", " "), (x + 15, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1)
        cv2.rectangle(img, (x + 105, ty), (x + 105 + bar_w, ty + 16), color, -1)
        cv2.rectangle(img, (x + 105, ty), (x + w - 35, ty + 16), GRAY, 1)

        txt = f"{duration:.1f}s | {mem['threat_level']} {mem['threat_score']:.0f}"
        cv2.putText(img, txt, (x + w - 145, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)


class UAVThreatAnalyticsDashboardV2(Node):
    def __init__(self):
        super().__init__("uav_analytics_dashboard_v2")

        self.start_time = time.time()

        self.latest_detections = []
        self.last_msg_time = 0.0

        self.target_memory = {}

        self.confidence_history = deque(maxlen=HISTORY_LEN)
        self.active_targets_history = deque(maxlen=HISTORY_LEN)
        self.detection_rate_history = deque(maxlen=HISTORY_LEN)
        self.avg_threat_score_history = deque(maxlen=HISTORY_LEN)

        self.critical_history = deque(maxlen=HISTORY_LEN)
        self.high_history = deque(maxlen=HISTORY_LEN)
        self.medium_history = deque(maxlen=HISTORY_LEN)
        self.low_history = deque(maxlen=HISTORY_LEN)

        self.total_detections = 0
        self.last_second_count = 0
        self.last_rate_time = time.time()

        self.output_dir = OUTPUTS_DIR / "ros2_dashboards"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.positions_csv_path = self.output_dir / "live_target_positions_threat_v2.csv"

        self.positions_file = open(self.positions_csv_path, "w", newline="", encoding="utf-8")
        self.positions_writer = csv.writer(self.positions_file)
        self.positions_writer.writerow([
            "timestamp",
            "mission_time_sec",
            "target_id",
            "class_name",
            "platform_category",
            "confidence",
            "threat_score",
            "threat_level",
            "alert_priority",
            "center_x",
            "center_y",
            "norm_center_x",
            "norm_center_y",
            "track_id",
            "raw_id",
            "source",
        ])

        self.create_subscription(
            String,
            "/uav_1/coco_detections",
            self.detections_callback,
            10
        )

        self.timer = self.create_timer(0.10, self.render)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1400, 900)

        self.get_logger().info("UAV Threat Analytics Dashboard V2 started.")
        self.get_logger().info("V2 reads threat_score, threat_level, platform_category from Windows sender.")
        self.get_logger().info(f"Saving threat-aware positions to: {self.positions_csv_path}")

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
            self.last_msg_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Failed to parse detections: {e}")
            self.latest_detections = []

    def update_target_memory(self):
        now = time.time()
        mission_time = now - self.start_time

        active_ids = set()
        confidence_values = []
        threat_scores = []
        threat_counts = Counter()

        for det in self.latest_detections:
            target_id = get_target_id(det)

            if target_id is None:
                continue

            class_name = normalize_name(get_class_name(det))
            confidence = get_confidence(det)
            threat_score = get_threat_score(det, class_name, confidence)
            threat_level = get_threat_level(det, threat_score)
            alert_priority = get_alert_priority(det, threat_level)
            platform_category = get_platform_category(det, class_name)

            cx, cy = get_center(det)
            nx, ny = get_norm_center(det)

            if cx is None or cy is None:
                continue

            track_id = det.get("clean_track_id", det.get("track_id", ""))
            raw_id = det.get("raw_track_id", det.get("raw_id", ""))
            source = det.get("source", "")

            if target_id not in self.target_memory:
                self.target_memory[target_id] = {
                    "first_seen_time": now,
                    "last_seen_time": now,
                    "class_name": class_name,
                    "platform_category": platform_category,
                    "confidence": confidence,
                    "threat_score": threat_score,
                    "max_threat_score": threat_score,
                    "threat_level": threat_level,
                    "alert_priority": alert_priority,
                    "center": (cx, cy),
                    "norm_center": (nx, ny),
                    "norm_trail": deque(maxlen=60),
                    "total_seen": 0,
                }

            mem = self.target_memory[target_id]
            mem["last_seen_time"] = now
            mem["class_name"] = class_name
            mem["platform_category"] = platform_category
            mem["confidence"] = confidence
            mem["threat_score"] = threat_score
            mem["max_threat_score"] = max(mem.get("max_threat_score", threat_score), threat_score)
            mem["threat_level"] = threat_level
            mem["alert_priority"] = alert_priority
            mem["center"] = (cx, cy)
            mem["norm_center"] = (nx, ny)
            mem["norm_trail"].append((nx, ny))
            mem["total_seen"] += 1

            active_ids.add(target_id)
            confidence_values.append(confidence)
            threat_scores.append(threat_score)
            threat_counts[threat_level] += 1

            self.positions_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                round(mission_time, 3),
                target_id,
                class_name,
                platform_category,
                round(confidence, 4),
                round(threat_score, 2),
                threat_level,
                alert_priority,
                round(cx, 2),
                round(cy, 2),
                round(nx, 6),
                round(ny, 6),
                track_id,
                raw_id,
                source,
            ])

        self.total_detections += len(self.latest_detections)
        self.last_second_count += len(self.latest_detections)

        avg_conf = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
        avg_threat = (sum(threat_scores) / len(threat_scores)) if threat_scores else 0.0

        self.confidence_history.append(avg_conf * 100.0)
        self.avg_threat_score_history.append(avg_threat)
        self.active_targets_history.append(len(active_ids))

        self.critical_history.append(threat_counts["CRITICAL"])
        self.high_history.append(threat_counts["HIGH"])
        self.medium_history.append(threat_counts["MEDIUM"])
        self.low_history.append(threat_counts["LOW"])

        if now - self.last_rate_time >= 1.0:
            rate = self.last_second_count / max(0.001, now - self.last_rate_time)
            self.detection_rate_history.append(rate)
            self.last_second_count = 0
            self.last_rate_time = now
        else:
            if len(self.detection_rate_history) == 0:
                self.detection_rate_history.append(0.0)

    def expire_old_targets(self):
        now = time.time()

        old_targets = [
            tid for tid, mem in self.target_memory.items()
            if now - mem["last_seen_time"] > LONG_MEMORY_TIMEOUT
        ]

        for tid in old_targets:
            self.target_memory.pop(tid, None)

    def render(self):
        self.update_target_memory()
        self.expire_old_targets()

        canvas = np.zeros((900, 1400, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 12)

        cv2.putText(canvas, "UAV THREAT ANALYTICS DASHBOARD", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

        mission_time = int(time.time() - self.start_time)
        now = time.time()

        active_now = sum(
            1 for mem in self.target_memory.values()
            if now - mem["last_seen_time"] <= TARGET_TIMEOUT
        )

        critical_now = sum(
            1 for mem in self.target_memory.values()
            if now - mem["last_seen_time"] <= TARGET_TIMEOUT and mem["threat_level"] == "CRITICAL"
        )

        high_now = sum(
            1 for mem in self.target_memory.values()
            if now - mem["last_seen_time"] <= TARGET_TIMEOUT and mem["threat_level"] == "HIGH"
        )

        cv2.putText(canvas, f"Mission Time: {mission_time}s", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)
        cv2.putText(canvas, f"Active Targets: {active_now}", (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)
        cv2.putText(canvas, f"Critical: {critical_now}", (430, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, RED, 1)
        cv2.putText(canvas, f"High: {high_now}", (560, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ORANGE, 1)
        cv2.putText(canvas, f"Total Detections: {self.total_detections}", (680, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)
        cv2.putText(canvas, "ROS2: ONLINE | TCP: RECEIVING", (970, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1)

        draw_line_chart(
            canvas,
            "1. Average Confidence Over Time",
            self.confidence_history,
            25,
            90,
            430,
            220,
            GREEN,
            max_value=100,
            suffix="%"
        )

        draw_line_chart(
            canvas,
            "2. Average Threat Score Over Time",
            self.avg_threat_score_history,
            485,
            90,
            430,
            220,
            RED,
            max_value=100,
            suffix=""
        )

        draw_multi_line_chart(
            canvas,
            "3. Threat Level Timeline",
            {
                "CRITICAL": self.critical_history,
                "HIGH": self.high_history,
                "MEDIUM": self.medium_history,
                "LOW": self.low_history,
            },
            945,
            90,
            430,
            220
        )

        draw_line_chart(
            canvas,
            "4. Detection Rate Over Time",
            self.detection_rate_history,
            25,
            340,
            430,
            220,
            ORANGE,
            max_value=max(5, max(self.detection_rate_history) if self.detection_rate_history else 5),
            suffix="/s"
        )

        draw_target_lifetime_bars(
            canvas,
            self.target_memory,
            485,
            340,
            430,
            220
        )

        draw_tactical_map(
            canvas,
            self.target_memory,
            945,
            340,
            430,
            500
        )

        cv2.putText(
            canvas,
            f"Threat-aware position data saved to {self.positions_csv_path}",
            (25, 870),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            YELLOW,
            1
        )

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            rclpy.shutdown()

        self.positions_file.flush()

    def close_files(self):
        try:
            self.positions_file.flush()
            self.positions_file.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)

    node = UAVThreatAnalyticsDashboardV2()

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
