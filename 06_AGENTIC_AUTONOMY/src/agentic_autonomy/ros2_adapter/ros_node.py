"""Optional ROS 2 shell.

Importing this module has no ROS dependency and creates no runtime resources.
ROS packages are imported only from :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

from agentic_autonomy.replanner import replan_history
from agentic_autonomy.scenario_loader import load_policy
from agentic_autonomy.serialization import canonical_json

from .adapter import MissionStateAdapter
from .adapter_configuration import load_adapter_policy
from .errors import AdapterError
from .event_domain import EventTimestamp
from .normalized_events import parse_event
from .ros_message_mapper import (
    map_battery_state,
    map_canonical_json_message,
    map_legacy_detection_message,
    map_pose_stamped,
)
from .serialization import write_canonical_json
from .validation import validate_phase2_history


SUBSYSTEM_ROOT = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    """Launch the ROS node after importing ROS dependencies lazily."""
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import BatteryState
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are required only to launch ros_node; "
            "the adapter core remains usable without ROS 2."
        ) from exc

    parser = argparse.ArgumentParser(description="Advisory ROS 2 mission-state adapter")
    parser.add_argument("--adapter-policy", required=True)
    parser.add_argument("--planner-policy", required=True)
    args, ros_arguments = parser.parse_known_args(argv)
    adapter_policy = load_adapter_policy(args.adapter_policy)
    planner_policy = load_policy(args.planner_policy)

    class MissionStateAdapterNode(Node):
        def __init__(self):
            super().__init__("agentic_mission_state_adapter")
            self._adapter = MissionStateAdapter(adapter_policy, planner_policy)
            self._queue: Queue = Queue()
            self._sequence_lock = Lock()
            self._next_ingestion_sequence = 0
            ros = adapter_policy["ros"]
            self._snapshot_pub = self.create_publisher(String, ros["snapshot_topic"], 10)
            self._diagnostics_pub = self.create_publisher(
                String, ros["diagnostics_topic"], 10
            )
            self._advisory_pub = self.create_publisher(String, ros["advisory_topic"], 10)
            self.create_subscription(String, ros["event_topic"], self._canonical_callback, 10)
            for binding in ros["legacy_detection_topics"]:
                self.create_subscription(
                    String,
                    binding["topic"],
                    lambda message, item=binding: self._legacy_callback(message, item),
                    10,
                )
            for binding in ros["battery_topics"]:
                self.create_subscription(
                    BatteryState,
                    binding["topic"],
                    lambda message, item=binding: self._battery_callback(message, item),
                    10,
                )
            for binding in ros["pose_topics"]:
                self.create_subscription(
                    PoseStamped,
                    binding["topic"],
                    lambda message, item=binding: self._pose_callback(message, item),
                    10,
                )
            self.create_service(Trigger, ros["snapshot_service"], self._snapshot_service)
            self.create_timer(0.02, self._drain_queue)
            self.create_timer(
                adapter_policy["snapshot_trigger"]["periodic_interval_seconds"],
                self._snapshot_tick,
            )
            self.get_logger().info(
                "Mission-state adapter started in advisory-only mode; no flight-command interfaces exist."
            )

        def _next_sequence(self) -> int:
            with self._sequence_lock:
                self._next_ingestion_sequence += 1
                return self._next_ingestion_sequence

        def _ros_time(self) -> EventTimestamp:
            total = int(self.get_clock().now().nanoseconds)
            return EventTimestamp("ros", total // 1_000_000_000, total % 1_000_000_000)

        def _mission_id(self) -> str:
            mission_id = self._adapter.store.state.mission_id
            if mission_id is None:
                raise AdapterError(
                    "MISSION_CONFIGURED must arrive before mapped telemetry or detections"
                )
            return mission_id

        def _canonical_callback(self, message):
            try:
                if len(message.data.encode("utf-8")) > adapter_policy["limits"][
                    "maximum_message_bytes"
                ]:
                    raise AdapterError("canonical event message exceeds maximum_message_bytes")
                self._queue.put(
                    map_canonical_json_message(message, self._next_sequence())
                )
            except Exception as exc:
                self._log_error("CANONICAL_EVENT_REJECTED", exc)

        def _legacy_callback(self, message, binding):
            try:
                if len(message.data.encode("utf-8")) > adapter_policy["limits"][
                    "maximum_message_bytes"
                ]:
                    raise AdapterError("legacy detection message exceeds maximum_message_bytes")
                events = map_legacy_detection_message(
                    message,
                    mission_id=self._mission_id(),
                    next_sequence=self._next_sequence,
                    source_uav_id=binding["uav_id"],
                    source_session_id=binding["source_session_id"],
                    receipt_time=self._ros_time(),
                    topic=binding["topic"],
                )
                for event in events:
                    self._queue.put(event)
            except Exception as exc:
                self._log_error("LEGACY_DETECTION_REJECTED", exc)

        def _battery_callback(self, message, binding):
            try:
                self._queue.put(
                    map_battery_state(
                        message,
                        mission_id=self._mission_id(),
                        ingestion_sequence=self._next_sequence(),
                        uav_id=binding["uav_id"],
                        source_id=binding["source_id"],
                        receipt_time=self._ros_time(),
                        topic=binding["topic"],
                    )
                )
            except Exception as exc:
                self._log_error("BATTERY_MESSAGE_REJECTED", exc)

        def _pose_callback(self, message, binding):
            try:
                self._queue.put(
                    map_pose_stamped(
                        message,
                        mission_id=self._mission_id(),
                        ingestion_sequence=self._next_sequence(),
                        uav_id=binding["uav_id"],
                        source_id=binding["source_id"],
                        receipt_time=self._ros_time(),
                        topic=binding["topic"],
                    )
                )
            except Exception as exc:
                self._log_error("POSE_MESSAGE_REJECTED", exc)

        def _snapshot_tick(self):
            try:
                self._queue.put(self._control_event("SNAPSHOT_TICK", "periodic ROS timer"))
            except Exception as exc:
                self._log_error("SNAPSHOT_TICK_REJECTED", exc)

        def _snapshot_service(self, request, response):
            del request
            try:
                self._queue.put(
                    self._control_event("SNAPSHOT_REQUESTED", "explicit ROS service request")
                )
                response.success = True
                response.message = "Snapshot request queued."
            except Exception as exc:
                response.success = False
                response.message = str(exc)
            return response

        def _control_event(self, event_type, reason):
            sequence = self._next_sequence()
            timestamp = self._ros_time()
            return parse_event(
                {
                    "schema_version": "1.0",
                    "mission_id": self._mission_id(),
                    "event_id": f"ros-shell-{event_type.lower()}-{sequence:09d}",
                    "sequence": sequence,
                    "event_type": event_type,
                    "observed_at": {
                        "clock_id": timestamp.clock_id,
                        "sec": timestamp.sec,
                        "nanosec": timestamp.nanosec,
                    },
                    "source": {
                        "source_id": "agentic_mission_state_adapter",
                        "source_node": "agentic_mission_state_adapter",
                        "topic": None,
                        "message_type": "internal",
                        "source_uav_id": None,
                        "source_session_id": None,
                        "source_timestamp": timestamp.label,
                        "upstream_sequence": None,
                    },
                    "payload": {"reason": reason},
                }
            )

        def _drain_queue(self):
            while True:
                try:
                    event = self._queue.get_nowait()
                except Empty:
                    return
                try:
                    snapshot = self._adapter.process_event(event)
                    if snapshot is not None:
                        self._publish_outputs()
                except Exception as exc:
                    self._log_error("EVENT_APPLICATION_REJECTED", exc)

        def _publish_outputs(self):
            history = self._adapter.history_document()
            diagnostics = self._adapter.diagnostics_document()
            snapshot_message = String()
            snapshot_message.data = canonical_json(history)
            self._snapshot_pub.publish(snapshot_message)
            diagnostics_message = String()
            diagnostics_message.data = canonical_json(diagnostics)
            self._diagnostics_pub.publish(diagnostics_message)
            configured_output = os.getenv(
                "UAV_AGENTIC_OUTPUT_DIR",
                adapter_policy["ros"]["output_directory"],
            )
            output_dir = Path(os.path.expandvars(configured_output)).expanduser()
            if not output_dir.is_absolute():
                output_dir = SUBSYSTEM_ROOT / output_dir
            write_canonical_json(history, output_dir / "mission_state_sequence.json")
            write_canonical_json(diagnostics, output_dir / "diagnostics.json")
            if adapter_policy["ros"]["invoke_replanner"]:
                result = replan_history(
                    validate_phase2_history(history), planner_policy
                )
                advisory = String()
                advisory.data = canonical_json(result)
                self._advisory_pub.publish(advisory)
                write_canonical_json(result, output_dir / "replanning_advisory.json")

        def _log_error(self, code, exc):
            record = {
                "severity": "ERROR",
                "code": code,
                "message": str(exc),
            }
            self.get_logger().error(json.dumps(record, sort_keys=True))

    rclpy.init(args=ros_arguments)
    node = MissionStateAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
