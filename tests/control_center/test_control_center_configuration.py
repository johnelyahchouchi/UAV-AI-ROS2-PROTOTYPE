from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = PROJECT_ROOT / "01_WINDOWS_AI"
for search_path in (PROJECT_ROOT, WINDOWS_AI_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from control_center.configuration import (  # noqa: E402
    AUTO_VIDEO,
    MANUAL_REGION,
    VIDEO_FILE,
    ControlCenterError,
    DashboardSettings,
    LiveTesterSettings,
    MissionSettings,
    SenderLaunchSettings,
    build_dashboard_environment,
    build_live_tester_command,
    build_mission_command,
    build_sender_command,
    build_sender_environment,
    load_local_settings,
    save_local_settings,
)
from live_tester.configuration import build_parser  # noqa: E402


def test_live_command_exposes_video_inference_and_v1_v2_options(tmp_path: Path) -> None:
    base = tmp_path / "base model.pt"
    v2 = tmp_path / "dropout v2.pt"
    video = tmp_path / "demo video.mp4"
    registry = tmp_path / "trusted.csv"
    for path in (base, v2, video, registry):
        path.touch()
    settings = LiveTesterSettings(
        model_path=str(base),
        mcdo_model_path=str(v2),
        source_mode=VIDEO_FILE,
        video_path=str(video),
        loop_video=True,
        confidence="0.31",
        iou="0.52",
        image_size="768",
        device="cuda:0",
        max_fps="24",
        classes="military_tank,cargo_truck",
        tank_only=False,
        output_directory=str(tmp_path / "output"),
        registry_path=str(registry),
        uncertainty_enabled=True,
        uncertainty_samples="12",
        uncertainty_seed="7",
        uncertainty_match_iou="0.4",
        mcdo_enabled=True,
        mcdo_required=True,
        mcdo_passes="25",
        mcdo_match_iou="0.42",
    )

    command = build_live_tester_command(settings, sys.executable)

    assert command[:3] == [sys.executable, "-u", str(PROJECT_ROOT / "01_WINDOWS_AI" / "apps" / "live_screen_model_tester.py")]
    assert command[command.index("--model") + 1] == str(base.resolve())
    assert command[command.index("--video") + 1] == str(video.resolve())
    assert command[command.index("--mcdo-v2-model") + 1] == str(v2.resolve())
    assert "--loop-video" in command
    assert "--require-mcdo-v2" in command
    assert "--uncertainty-samples" in command
    parsed = build_parser().parse_args(command[3:])
    assert parsed.mcdo_v2_passes == 25
    assert parsed.require_mcdo_v2 is True
    assert parsed.device == "cuda:0"


def test_live_command_supports_manual_region_without_video_flags(tmp_path: Path) -> None:
    model = tmp_path / "base.pt"
    model.touch()
    settings = LiveTesterSettings(
        model_path=str(model),
        source_mode=MANUAL_REGION,
        monitor="2",
        left="-100",
        top="20",
        width="900",
        height="600",
        loop_video=True,
        uncertainty_enabled=False,
    )

    command = build_live_tester_command(settings, sys.executable)

    assert "--left" in command
    assert command[command.index("--left") + 1] == "-100"
    assert "--loop-video" not in command
    assert "--disable-uncertainty" in command
    build_parser().parse_args(command[3:])


def test_live_command_rejects_v2_when_uncertainty_is_disabled(tmp_path: Path) -> None:
    model = tmp_path / "base.pt"
    v2 = tmp_path / "v2.pt"
    model.touch()
    v2.touch()
    settings = LiveTesterSettings(
        model_path=str(model),
        mcdo_model_path=str(v2),
        source_mode=AUTO_VIDEO,
        uncertainty_enabled=False,
        mcdo_enabled=True,
    )

    with pytest.raises(ControlCenterError, match="uncertainty is disabled"):
        build_live_tester_command(settings, sys.executable)


def test_sender_command_and_tls_environment_are_separate(tmp_path: Path) -> None:
    model = tmp_path / "sender.pt"
    certificate = tmp_path / "sender.crt"
    private_key = tmp_path / "sender.key"
    ca = tmp_path / "bridge-ca.crt"
    for path in (model, certificate, private_key, ca):
        path.touch()
    settings = SenderLaunchSettings(
        model_path=str(model),
        source="rtsp://camera.example/live",
        target="127.0.0.1",
        port="5010",
        tracker="botsort.yaml",
        tls_certificate=str(certificate),
        tls_private_key=str(private_key),
        tls_ca_certificate=str(ca),
        tls_server_name="bridge.example",
    )

    command = build_sender_command(settings, sys.executable)
    environment = build_sender_environment(settings)

    assert command[command.index("--target") + 1] == "127.0.0.1"
    assert command[command.index("--source") + 1] == "rtsp://camera.example/live"
    assert str(private_key.resolve()) not in command
    assert environment == {
        "UAV_SENDER_TLS_CERT": str(certificate.resolve()),
        "UAV_SENDER_TLS_KEY": str(private_key.resolve()),
        "UAV_BRIDGE_TLS_CA": str(ca.resolve()),
        "UAV_BRIDGE_TLS_SERVER_NAME": "bridge.example",
    }


def test_dashboard_port_is_validated_and_kept_on_loopback_configuration() -> None:
    assert build_dashboard_environment(DashboardSettings(port="8765")) == {
        "UAV_DASHBOARD_PORT": "8765"
    }
    with pytest.raises(ControlCenterError, match="Dashboard port"):
        build_dashboard_environment(DashboardSettings(port="not-a-port"))


def test_mission_command_sets_only_the_adapter_import_environment(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.json"
    policy = tmp_path / "policy.json"
    scenario.write_text("{}", encoding="utf-8")
    policy.write_text("{}", encoding="utf-8")
    output = tmp_path / "nested" / "plan.json"
    settings = MissionSettings(
        scenario_path=str(scenario),
        policy_path=str(policy),
        output_path=str(output),
        verbose=True,
        validate_only=False,
        require_complete=True,
    )

    command, environment = build_mission_command(settings, sys.executable)

    assert command[:4] == [sys.executable, "-u", "-m", "agentic_autonomy"]
    assert command[command.index("--output") + 1] == str(output.resolve())
    assert "--verbose" in command
    assert "--require-complete" in command
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPATH"].endswith("06_AGENTIC_AUTONOMY\\src")


def test_local_settings_round_trip_but_do_not_persist_stream_credentials(tmp_path: Path) -> None:
    destination = tmp_path / "settings.json"
    live = LiveTesterSettings(model_path="D:/models/base.pt")
    sender = SenderLaunchSettings(source="rtsp://user:password@camera.example/live")
    mission = MissionSettings()

    saved = save_local_settings(
        live=live,
        dashboard=DashboardSettings(port="8765"),
        sender=sender,
        mission=mission,
        path=destination,
    )
    payload = load_local_settings(saved)

    assert payload["live"]["model_path"] == "D:/models/base.pt"
    assert payload["dashboard"]["port"] == "8765"
    assert payload["sender"]["source"] == ""
    assert "password" not in json.dumps(payload)


def test_full_gui_batch_launcher_is_relative_and_portable() -> None:
    launcher = PROJECT_ROOT / "01_WINDOWS_AI" / "launchers" / "Start_UAV_Prototype_GUI.bat"
    content = launcher.read_text(encoding="utf-8")

    assert "%~dp0..\\apps\\uav_prototype_control_center.py" in content
    assert ".venv\\Scripts\\python.exe" in content
    assert "..\\UAV_YOLO_ENV\\Scripts\\python.exe" in content
    assert "UAV_YOLO_PYTHON" in content
    assert "C:\\Users\\" not in content
    assert "OneDrive" not in content


def test_gui_exposes_every_supported_configuration_field() -> None:
    source = (
        PROJECT_ROOT / "01_WINDOWS_AI" / "control_center" / "app.py"
    ).read_text(encoding="utf-8")

    for model, variable_name in (
        (LiveTesterSettings, "live_vars"),
        (DashboardSettings, "dashboard_vars"),
        (SenderLaunchSettings, "sender_vars"),
        (MissionSettings, "mission_vars"),
    ):
        for field in fields(model):
            explicit_binding = f'self.{variable_name}["{field.name}"]'
            loop_binding = f'"{field.name}"'
            assert explicit_binding in source or loop_binding in source
