#!/usr/bin/env python3

import json
import socket
import struct
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

def force_btr_labels(detections):
    """
    Temporary BTR demo fix:
    For the custom BTR model, class_id 0 means BTR, not COCO person.
    This rewrites outgoing detection labels before publishing to ROS2.
    """
    if not isinstance(detections, list):
        return detections

    for det in detections:
        if not isinstance(det, dict):
            continue

        cls_id = det.get("class_id", det.get("cls", det.get("id", None)))
        label = str(det.get("class", det.get("class_name", det.get("final_class", "")))).lower()

        if cls_id == 0 or cls_id == "0" or label == "person":
            det["class"] = "BTR"
            det["class_name"] = "BTR"
            det["final_class"] = "BTR"
            det["label"] = "BTR"

    return detections



def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class UAVWindowsTCPFrameBridge(Node):
    def __init__(self):
        super().__init__("uav_windows_tcp_frame_bridge")

        self.uav_id = "uav_1"
        self.port = 5010
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, f"/{self.uav_id}/camera/image_raw", 10)
        self.det_pub = self.create_publisher(String, f"/{self.uav_id}/coco_detections", 10)

        self.latest_frame = None
        self.latest_detections = []
        self.latest_seq = -1
        self.published_seq = -1
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.server_loop, daemon=True)
        self.thread.start()

        self.timer = self.create_timer(0.01, self.publish_latest)

        self.get_logger().info("TCP bridge ready")
        self.get_logger().info(f"Listening on port {self.port}")
        self.get_logger().info(f"Publishing /{self.uav_id}/camera/image_raw")
        self.get_logger().info(f"Publishing /{self.uav_id}/coco_detections")

    def server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(1)

        while True:
            self.get_logger().info("Waiting for Windows YOLO sender...")
            conn, addr = srv.accept()
            self.get_logger().info(f"Connected: {addr}")

            try:
                while True:
                    header_len_bytes = recv_exact(conn, 4)
                    if header_len_bytes is None:
                        break

                    header_len = struct.unpack("!I", header_len_bytes)[0]

                    header_bytes = recv_exact(conn, header_len)
                    if header_bytes is None:
                        break

                    header = json.loads(header_bytes.decode("utf-8"))

                    jpeg_size = int(header["jpeg_size"])
                    jpeg_bytes = recv_exact(conn, jpeg_size)
                    if jpeg_bytes is None:
                        break

                    frame_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                    with self.lock:
                        self.latest_frame = frame
                        self.latest_detections = header.get("detections", [])
                        self.latest_seq = int(header.get("seq", 0))

            except Exception as e:
                self.get_logger().error(f"Bridge error: {e}")

            finally:
                conn.close()
                self.get_logger().warn("Windows sender disconnected")

    def publish_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return
            if self.latest_seq == self.published_seq:
                return

            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)
            seq = self.latest_seq

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "uav_1_camera"
        self.image_pub.publish(img_msg)

        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.det_pub.publish(det_msg)

        self.published_seq = seq


def main(args=None):
    rclpy.init(args=args)
    node = UAVWindowsTCPFrameBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()