#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import OUTPUTS_DIR


class UAVSaveSnapshotOnce(Node):
    def __init__(self):
        super().__init__("uav_save_snapshot_once")

        self.bridge = CvBridge()
        self.frame = None
        self.detections = []

        self.output_dir = OUTPUTS_DIR / "ros2_dashboards"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(Image, "/uav_1/camera/image_raw", self.image_cb, 10)
        self.create_subscription(String, "/uav_1/coco_detections", self.det_cb, 10)

        self.timer = self.create_timer(0.5, self.try_save)

        self.get_logger().info("Waiting for frame + detections to save snapshot...")

    def image_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def det_cb(self, msg):
        try:
            self.detections = json.loads(msg.data)
        except Exception:
            self.detections = []

    def try_save(self):
        if self.frame is None:
            return

        frame = self.frame.copy()

        for det in self.detections:
            bbox = det.get("bbox") or det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = det.get("final_class", det.get("class", "object"))
            conf = float(det.get("confidence", 0.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"uav_ros2_snapshot_{ts}.jpg"

        cv2.imwrite(str(path), frame)

        self.get_logger().info(f"SNAPSHOT SAVED: {path}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = UAVSaveSnapshotOnce()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
