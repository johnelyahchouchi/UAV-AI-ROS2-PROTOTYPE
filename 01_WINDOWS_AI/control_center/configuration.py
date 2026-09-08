"""Validated settings and subprocess commands for the desktop control center."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from uav_security.input_validation import InputValidationError, validate_sender_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = PROJECT_ROOT / "01_WINDOWS_AI"
LIVE_TESTER_SCRIPT = WINDOWS_AI_ROOT / "apps" / "live_screen_model_tester.py"
SENDER_SCRIPT = WINDOWS_AI_ROOT / "apps" / "win_yolo_tcp_sender_botsort_threat.py"
DASHBOARD_LAUNCHER = WINDOWS_AI_ROOT / "model_test_dashboard" / "launch_dashboard.ps1"
MISSION_ROOT = PROJECT_ROOT / "06_AGENTIC_AUTONOMY"
MISSION_SOURCE = MISSION_ROOT / "src"
LOCAL_SETTINGS_PATH = PROJECT_ROOT / "08_OUTPUTS" / "control_center" / "settings.json"

VIDEO_PICKER = "Select MP4"
VIDEO_FILE = "Video file"
AUTO_VIDEO = "Auto-detect MP4"
FULL_MONITOR = "Full monitor"
SELECT_REGION = "Select screen region"
MANUAL_REGION = "Manual screen region"
SOURCE_MODES = (
    VIDEO_PICKER,
    VIDEO_FILE,
    AUTO_VIDEO,
    FULL_MONITOR,
    SELECT_REGION,
    MANUAL_REGION,
)


class ControlCenterError(ValueError):
    """Raised when GUI settings cannot produce a safe application command."""


@dataclass(frozen=True)
class LiveTesterSettings:
    """All configurable live tester inputs exposed by the control center."""

    model_path: str = os.environ.get("UAV_MODEL_PATH", "")
    mcdo_model_path: str = os.environ.get("UAV_MCDO_V2_MODEL_PATH", "")
    source_mode: str = VIDEO_PICKER
    video_path: str = ""
    loop_video: bool = True
    monitor: str = "1"
    left: str = "0"
    top: str = "0"
    width: str = "1280"
    height: str = "720"
    confidence: str = "0.25"
    iou: str = "0.45"
    image_size: str = "640"
    device: str = "auto"
    max_fps: str = "0"
    classes: str = "all"
    tank_only: bool = False
    output_directory: str = ""
    registry_path: str = os.environ.get("UAV_TRUSTED_MODEL_REGISTRY", "")
    uncertainty_enabled: bool = True
    uncertainty_samples: str = "10"
    uncertainty_seed: str = "42"
    uncertainty_match_iou: str = "0.50"
    mcdo_enabled: bool = False
    mcdo_required: bool = True
    mcdo_passes: str = "20"
    mcdo_match_iou: str = "0.50"


@dataclass(frozen=True)
class DashboardSettings:
    """Loopback-only recorded analysis dashboard settings."""

    port: str = os.environ.get("UAV_DASHBOARD_PORT", "7860")


@dataclass(frozen=True)
class SenderLaunchSettings:
    """Windows sender inputs retained as an independently launchable workflow."""

    model_path: str = os.environ.get("UAV_MODEL_PATH", "")
    source: str = ""
    target: str = "127.0.0.1"
    port: str = "5010"
    confidence: str = "0.25"
    iou: str = "0.45"
    image_size: str = "960"
    stride: str = "1"
    send_width: str = "960"
    tracker: str = "botsort.yaml"
    show: bool = True
    military_only: bool = True
    tls_certificate: str = os.environ.get("UAV_SENDER_TLS_CERT", "")
    tls_private_key: str = os.environ.get("UAV_SENDER_TLS_KEY", "")
    tls_ca_certificate: str = os.environ.get("UAV_BRIDGE_TLS_CA", "")
    tls_server_name: str = os.environ.get("UAV_BRIDGE_TLS_SERVER_NAME", "")


@dataclass(frozen=True)
class MissionSettings:
    """Offline deterministic Mission Copilot inputs."""

    scenario_path: str = str(MISSION_ROOT / "scenarios" / "basic_reconnaissance.json")
    policy_path: str = str(MISSION_ROOT / "config" / "default_policy.json")
    output_path: str = str(
        PROJECT_ROOT / "08_OUTPUTS" / "mission_copilot" / "mission_plan.json"
    )
    verbose: bool = True
    validate_only: bool = False
    require_complete: bool = False


def _existing_file(value: str, label: str, suffix: str | None = None) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ControlCenterError(f"{label} is required")
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ControlCenterError(f"{label} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ControlCenterError(f"{label} is not a file: {resolved}")
    if suffix and resolved.suffix.lower() != suffix.lower():
        raise ControlCenterError(f"{label} must be a {suffix} file")
    return resolved


def _integer(value: str, label: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as error:
        raise ControlCenterError(f"{label} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise ControlCenterError(f"{label} must be between {minimum} and {maximum}")
    return parsed


def _number(value: str, label: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(str(value).strip())
    except ValueError as error:
        raise ControlCenterError(f"{label} must be a number") from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ControlCenterError(f"{label} must be between {minimum} and {maximum}")
    return parsed


def _python_command(python_executable: str | Path, script: Path) -> list[str]:
    python_path = _existing_file(str(python_executable), "Python executable")
    if not script.is_file():
        raise ControlCenterError(f"Application entry point is missing: {script}")
    return [str(python_path), "-u", str(script)]


def build_live_tester_command(
    settings: LiveTesterSettings,
    python_executable: str | Path,
) -> list[str]:
    """Build the complete live tester command after validating every GUI field."""

    command = _python_command(python_executable, LIVE_TESTER_SCRIPT)
    model = _existing_file(settings.model_path, "Base detector", ".pt")
    command.extend(("--model", str(model)))

    if settings.source_mode not in SOURCE_MODES:
        raise ControlCenterError(f"Unsupported source mode: {settings.source_mode}")
    video_mode = settings.source_mode in {VIDEO_PICKER, VIDEO_FILE, AUTO_VIDEO}
    if settings.source_mode == VIDEO_PICKER:
        command.append("--select-video")
    elif settings.source_mode == VIDEO_FILE:
        video = _existing_file(settings.video_path, "Video", ".mp4")
        command.extend(("--video", str(video)))
    elif settings.source_mode == AUTO_VIDEO:
        command.append("--auto-video")
    else:
        monitor = _integer(settings.monitor, "Monitor", 1, 64)
        command.extend(("--monitor", str(monitor)))
        if settings.source_mode == SELECT_REGION:
            command.append("--select-region")
        elif settings.source_mode == MANUAL_REGION:
            left = _integer(settings.left, "Left", -100_000, 100_000)
            top = _integer(settings.top, "Top", -100_000, 100_000)
            width = _integer(settings.width, "Width", 1, 100_000)
            height = _integer(settings.height, "Height", 1, 100_000)
            command.extend(
                (
                    "--left",
                    str(left),
                    "--top",
                    str(top),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                )
            )
    if settings.loop_video and video_mode:
        command.append("--loop-video")

    confidence = _number(settings.confidence, "Confidence", 0.0, 1.0)
    iou = _number(settings.iou, "IoU", 0.0, 1.0)
    image_size = _integer(settings.image_size, "Image size", 1, 8192)
    max_fps = _number(settings.max_fps, "Maximum FPS", 0.0, 240.0)
    device = str(settings.device).strip() or "auto"
    classes = str(settings.classes).strip() or "all"
    command.extend(
        (
            "--conf",
            str(confidence),
            "--iou",
            str(iou),
            "--imgsz",
            str(image_size),
            "--device",
            device,
            "--max-fps",
            str(max_fps),
            "--classes",
            classes,
        )
    )
    if settings.tank_only:
        command.append("--tank-only")

    if str(settings.output_directory).strip():
        output = Path(settings.output_directory).expanduser().resolve()
        command.extend(("--output-dir", str(output)))
    if str(settings.registry_path).strip():
        registry = _existing_file(settings.registry_path, "Model registry", ".csv")
        command.extend(("--registry", str(registry)))

    if not settings.uncertainty_enabled:
        if settings.mcdo_enabled:
            raise ControlCenterError("V2 cannot be enabled when uncertainty is disabled")
        command.append("--disable-uncertainty")
        return command

    samples = _integer(settings.uncertainty_samples, "V1 samples", 1, 50)
    seed = _integer(settings.uncertainty_seed, "V1 seed", -2_147_483_648, 2_147_483_647)
    match_iou = _number(settings.uncertainty_match_iou, "V1 match IoU", 0.000001, 1.0)
    command.extend(
        (
            "--uncertainty-samples",
            str(samples),
            "--uncertainty-seed",
            str(seed),
            "--uncertainty-match-iou",
            str(match_iou),
        )
    )
    if settings.mcdo_enabled:
        mcdo_model = _existing_file(
            settings.mcdo_model_path, "V2 MC Dropout checkpoint", ".pt"
        )
        passes = _integer(settings.mcdo_passes, "V2 passes", 1, 100)
        mcdo_match = _number(settings.mcdo_match_iou, "V2 match IoU", 0.000001, 1.0)
        command.extend(
            (
                "--mcdo-v2-model",
                str(mcdo_model),
                "--mcdo-v2-passes",
                str(passes),
                "--mcdo-v2-match-iou",
                str(mcdo_match),
            )
        )
        if settings.mcdo_required:
            command.append("--require-mcdo-v2")
    return command


def build_sender_command(
    settings: SenderLaunchSettings,
    python_executable: str | Path,
) -> list[str]:
    """Build the existing TCP sender command using its shared scalar validation."""

    command = _python_command(python_executable, SENDER_SCRIPT)
    model = _existing_file(settings.model_path, "Sender model", ".pt")
    source = str(settings.source).strip()
    if not source:
        raise ControlCenterError("Sender source is required")
    try:
        validated = validate_sender_settings(
            target=settings.target,
            port=settings.port,
            confidence=settings.confidence,
            iou=settings.iou,
            image_size=settings.image_size,
            stride=settings.stride,
            send_width=settings.send_width,
            show="1" if settings.show else "0",
            military_only="1" if settings.military_only else "0",
        )
    except InputValidationError as error:
        raise ControlCenterError(str(error)) from error
    tracker = str(settings.tracker).strip()
    if not tracker:
        raise ControlCenterError("Tracker configuration is required")
    command.extend(
        (
            "--target",
            validated.target,
            "--port",
            str(validated.port),
            "--source",
            source,
            "--model",
            str(model),
            "--conf",
            str(validated.confidence),
            "--iou",
            str(validated.iou),
            "--imgsz",
            str(validated.image_size),
            "--stride",
            str(validated.stride),
            "--send_width",
            str(validated.send_width),
            "--show",
            str(validated.show),
            "--military_only",
            str(validated.military_only),
            "--tracker",
            tracker,
        )
    )
    return command


def build_sender_environment(settings: SenderLaunchSettings) -> dict[str, str]:
    """Validate and return the sender's mutual-TLS environment."""

    certificate = _existing_file(settings.tls_certificate, "Sender TLS certificate")
    private_key = _existing_file(settings.tls_private_key, "Sender TLS private key")
    ca_certificate = _existing_file(
        settings.tls_ca_certificate, "Bridge CA certificate"
    )
    server_name = str(settings.tls_server_name or settings.target).strip()
    if not server_name:
        raise ControlCenterError("Bridge TLS server name is required")
    return {
        "UAV_SENDER_TLS_CERT": str(certificate),
        "UAV_SENDER_TLS_KEY": str(private_key),
        "UAV_BRIDGE_TLS_CA": str(ca_certificate),
        "UAV_BRIDGE_TLS_SERVER_NAME": server_name,
    }


def build_dashboard_environment(settings: DashboardSettings) -> dict[str, str]:
    """Validate the recorded dashboard's loopback port."""

    port = _integer(settings.port, "Dashboard port", 1024, 65_535)
    return {"UAV_DASHBOARD_PORT": str(port)}


def build_mission_command(
    settings: MissionSettings,
    python_executable: str | Path,
) -> tuple[list[str], dict[str, str]]:
    """Build the offline Mission Copilot command and its isolated import environment."""

    python_path = _existing_file(str(python_executable), "Python executable")
    scenario = _existing_file(settings.scenario_path, "Mission scenario", ".json")
    policy = _existing_file(settings.policy_path, "Mission policy", ".json")
    output_raw = str(settings.output_path).strip()
    if not output_raw:
        raise ControlCenterError("Mission output path is required")
    output = Path(output_raw).expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise ControlCenterError("Mission output must be a .json file")
    command = [
        str(python_path),
        "-u",
        "-m",
        "agentic_autonomy",
        "--scenario",
        str(scenario),
        "--policy",
        str(policy),
        "--output",
        str(output),
    ]
    if settings.verbose:
        command.append("--verbose")
    if settings.validate_only:
        command.append("--validate-only")
    if settings.require_complete:
        command.append("--require-complete")
    return command, {"PYTHONPATH": str(MISSION_SOURCE), "PYTHONDONTWRITEBYTECODE": "1"}


def load_local_settings(path: Path = LOCAL_SETTINGS_PATH) -> dict[str, Any]:
    """Load ignored workstation-only UI settings, returning defaults on bad input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_local_settings(
    *,
    live: LiveTesterSettings,
    dashboard: DashboardSettings = DashboardSettings(),
    sender: SenderLaunchSettings,
    mission: MissionSettings,
    path: Path = LOCAL_SETTINGS_PATH,
) -> Path:
    """Persist non-secret local paths and GUI settings under the ignored output tree."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sender_payload = asdict(sender)
    if "://" in sender_payload["source"]:
        sender_payload["source"] = ""
    payload: Mapping[str, Any] = {
        "version": 1,
        "live": asdict(live),
        "dashboard": asdict(dashboard),
        "sender": sender_payload,
        "mission": asdict(mission),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
