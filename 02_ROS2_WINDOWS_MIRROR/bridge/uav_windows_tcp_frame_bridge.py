#!/usr/bin/env python3

import json
import ipaddress
import os
from pathlib import Path
import socket
import ssl
import sys
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_security.config import DEFAULT_BIND_ADDRESS, DEFAULT_PORT, SecurityLimits
from uav_security.detection import sanitize_frame_header
from uav_security.image_validation import decode_and_validate_jpeg
from uav_security.input_validation import validate_integer, validate_ip
from uav_security.transport import (
    ConnectionClosed,
    ProtocolError,
    SessionReplayCache,
    SessionSequenceValidator,
    create_server_tls_context,
    parse_allowed_cidrs,
    peer_is_allowed,
    receive_packet,
    server_tls_files,
    validate_negotiated_tls,
)

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

class UAVWindowsTCPFrameBridge(Node):
    def __init__(self):
        super().__init__("uav_windows_tcp_frame_bridge")

        self.uav_id = "uav_1"
        self.bind_address = validate_ip(os.environ.get("UAV_BRIDGE_BIND_ADDRESS", DEFAULT_BIND_ADDRESS))
        self.port = validate_integer(
            os.environ.get("UAV_BRIDGE_PORT", str(DEFAULT_PORT)), "Bridge port", 1, 65_535
        )
        self.allowed_networks = parse_allowed_cidrs(os.environ.get("UAV_BRIDGE_ALLOWED_CIDRS"))
        self.limits = SecurityLimits.from_environment()
        self.tls_context = create_server_tls_context(server_tls_files())
        self.replay_cache = SessionReplayCache()
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, f"/{self.uav_id}/camera/image_raw", 10)
        self.det_pub = self.create_publisher(String, f"/{self.uav_id}/coco_detections", 10)

        self.latest_frame = None
        self.latest_detections = []
        self.latest_message_id = None
        self.published_message_id = None
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.server_loop, daemon=True)
        self.thread.start()

        self.timer = self.create_timer(0.01, self.publish_latest)

        self.get_logger().info("Authenticated TLS 1.3 TCP bridge ready")
        self.get_logger().info(f"Listening on {self.bind_address}:{self.port}")
        self.get_logger().info(f"Publishing /{self.uav_id}/camera/image_raw")
        self.get_logger().info(f"Publishing /{self.uav_id}/coco_detections")

    def server_loop(self):
        family = socket.AF_INET6 if ipaddress.ip_address(self.bind_address).version == 6 else socket.AF_INET
        srv = socket.socket(family, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_address, self.port))
        srv.listen(self.limits.listen_backlog)
        srv.settimeout(self.limits.listener_timeout)

        while rclpy.ok():
            try:
                raw_conn, addr = srv.accept()
            except socket.timeout:
                continue
            peer_ip = addr[0]
            if not peer_is_allowed(peer_ip, self.allowed_networks):
                self.get_logger().warning(f"Rejected TCP peer outside allowlist: {peer_ip}")
                raw_conn.close()
                continue
            raw_conn.settimeout(self.limits.socket_read_timeout)

            try:
                with self.tls_context.wrap_socket(raw_conn, server_side=True) as conn:
                    validate_negotiated_tls(conn)
                    self.get_logger().info(f"Authenticated sender connected: {peer_ip}")
                    self.handle_connection(conn)
            except ConnectionClosed:
                self.get_logger().warning("Authenticated sender disconnected")
            except (ssl.SSLError, ProtocolError, ValueError) as error:
                self.get_logger().warning(f"Rejected sender data from {peer_ip}: {error}")
            except OSError as error:
                self.get_logger().warning(f"Sender connection ended: {error}")
            finally:
                raw_conn.close()

        srv.close()

    def handle_connection(self, conn):
        sequence_validator = SessionSequenceValidator(self.replay_cache)
        while rclpy.ok():
            packet = receive_packet(conn, limits=self.limits)
            sequence = sequence_validator.check(packet.header)
            frame = decode_and_validate_jpeg(packet.jpeg, limits=self.limits)
            height, width = frame.shape[:2]
            header = sanitize_frame_header(packet.header, width, height, limits=self.limits)
            detections = force_btr_labels(header["detections"])
            sequence_validator.commit(sequence)
            message_id = (header["session_id"], sequence)
            with self.lock:
                self.latest_frame = frame
                self.latest_detections = detections
                self.latest_message_id = message_id

    def publish_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return
            if self.latest_message_id == self.published_message_id:
                return

            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)
            message_id = self.latest_message_id

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "uav_1_camera"
        self.image_pub.publish(img_msg)

        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.det_pub.publish(det_msg)

        self.published_message_id = message_id


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
