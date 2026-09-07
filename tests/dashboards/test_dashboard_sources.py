from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "02_ROS2_WINDOWS_MIRROR" / "dashboards"


def test_only_canonical_dashboard_entrypoints_remain() -> None:
    scripts = sorted(path.name for path in DASHBOARD_ROOT.glob("*.py"))
    assert scripts == [
        "uav_analytics_dashboard.py",
        "uav_operational_dashboard.py",
        "uav_timeline_dashboard.py",
    ]


def test_dashboard_sources_parse_without_ros_runtime() -> None:
    for path in DASHBOARD_ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_primary_dashboard_preserves_camera_and_detection_topics() -> None:
    content = (DASHBOARD_ROOT / "uav_operational_dashboard.py").read_text(
        encoding="utf-8"
    )
    assert '"/uav_1/camera/image_raw"' in content
    assert '"/uav_1/coco_detections"' in content
    assert 'super().__init__("uav_operational_dashboard")' in content


def test_specialist_dashboards_preserve_detection_topic() -> None:
    for name in ("uav_analytics_dashboard.py", "uav_timeline_dashboard.py"):
        content = (DASHBOARD_ROOT / name).read_text(encoding="utf-8")
        assert '"/uav_1/coco_detections"' in content
