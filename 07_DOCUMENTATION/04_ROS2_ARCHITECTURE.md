# 04 - ROS 2 Architecture

`02_ROS2_WINDOWS_MIRROR/` contains the current bridge and three complementary dashboard
scripts. It is a source/deployment mirror, not a complete local ROS 2 installation.

## Bridge

The Windows sender is a mutual-TLS client. The bridge is a mutual-TLS server that
authenticates the sender, validates bounded protocol-v2 headers/JPEG data, rejects replay
and malformed detections, decodes the frame, and only then publishes ROS 2 messages.
There is no plaintext fallback.

The existing published topics remain:

- `/uav_1/camera/image_raw`
- `/uav_1/coco_detections`

## Dashboards

- `uav_operational_dashboard.py`: primary image, detection, track, and threat view.
- `uav_analytics_dashboard.py`: aggregate history, charts, map, and target lifetimes.
- `uav_timeline_dashboard.py`: first-seen and classification-change events.

These are purpose-specific consumers, not successive dashboard versions. Runtime CSVs
belong in external/local output directories. Renamed node identities require matching
SROS2 enclave policy updates before Ubuntu deployment.

## Validation boundary

Python syntax and dashboard source/topic tests can run without ROS 2. Actual node startup,
DDS/SROS2 policy enforcement, Ubuntu package layout, QoS, live topic flow, and hardware
integration must be validated in the real deployed ROS 2 workspace.
