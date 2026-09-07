# ROS 2 Dashboard Mirror

These three scripts are complementary ROS 2 views, not successive copies. Source the
ROS 2 environment and the deployment workspace before running them. This Windows tree
is a source mirror; it does not prove that ROS 2 or an SROS2 keystore is installed.

| Script | Role | Topics |
|---|---|---|
| `uav_operational_dashboard.py` | Primary presentation view: camera image, detections, tracks, threat badges, and CSV summaries. | `/uav_1/camera/image_raw`, `/uav_1/coco_detections` |
| `uav_analytics_dashboard.py` | Research view: aggregate threat history, charts, map, and target lifetimes. | `/uav_1/coco_detections` |
| `uav_timeline_dashboard.py` | Specialist event view: first-seen and changed target classifications over mission time. | `/uav_1/coco_detections` |

The primary demo uses `uav_operational_dashboard.py`. The other two remain because
they expose distinct historical analysis that the operational display does not.
Runtime CSV files are written to `~/uav_demo_outputs/` and must not be committed.

Example after sourcing the ROS environment:

```bash
python3 02_ROS2_WINDOWS_MIRROR/dashboards/uav_operational_dashboard.py
```

The topic names and JSON message expectations remain unchanged by the filename cleanup.
Node names are now unversioned; regenerate the corresponding SROS2 enclaves as described
in `07_DOCUMENTATION/11_SROS2_DEPLOYMENT.md` before deploying these renamed nodes.
