"""CLI, path, device, monitor, and source configuration for the live tester."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .domain import (
    CaptureRegion,
    DEFAULT_OUTPUT_DIRECTORY,
    PROJECT_ROOT,
    TANK_CLASS_NAME,
    TesterError,
    contains_rectangle,
    monitor_by_number,
    rectangles_intersect,
    validate_manual_region,
)


def discover_monitors(
    mss_factory: Callable[[], Any] | None = None,
) -> tuple[CaptureRegion, ...]:
    """Return physical monitors reported by MSS, excluding its aggregate."""

    if mss_factory is None:
        try:
            import mss
        except ImportError as error:
            raise TesterError(
                "Screen capture requires mss. Install requirements-windows.txt in "
                "the controlled UAV YOLO environment."
            ) from error
        mss_factory = mss.MSS
    try:
        with mss_factory() as grabber:
            raw_monitors = list(grabber.monitors)[1:]
    except Exception as error:
        raise TesterError(f"Could not enumerate monitors: {error}") from error
    monitors = tuple(
        CaptureRegion(
            left=int(item["left"]),
            top=int(item["top"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in raw_monitors
    )
    if not monitors:
        raise TesterError("No active monitors were reported by MSS")
    return monitors


def choose_preview_position(
    capture_region: CaptureRegion,
    monitors: Sequence[CaptureRegion],
    preview_width: int,
    preview_height: int,
    margin: int = 20,
) -> tuple[int, int, bool]:
    """Choose a preview origin, preferring outside the capture area."""

    if preview_width <= 0 or preview_height <= 0:
        raise TesterError("Preview dimensions must be positive")
    for monitor in monitors:
        if not rectangles_intersect(capture_region, monitor):
            return monitor.left + margin, monitor.top + margin, False
    containing = next(
        (monitor for monitor in monitors if contains_rectangle(monitor, capture_region)),
        monitors[0] if monitors else capture_region,
    )
    candidates = (
        (capture_region.right + margin, capture_region.top),
        (capture_region.left - preview_width - margin, capture_region.top),
        (capture_region.left, capture_region.bottom + margin),
        (capture_region.left, capture_region.top - preview_height - margin),
    )
    for left, top in candidates:
        preview = CaptureRegion(left, top, preview_width, preview_height)
        if contains_rectangle(containing, preview) and not rectangles_intersect(
            capture_region, preview
        ):
            return left, top, False
    return containing.left + margin, containing.top + margin, True


def resolve_model_path(
    cli_value: str | None, environment: Mapping[str, str] | None = None
) -> Path:
    """Resolve an explicit model path or the ``UAV_MODEL_PATH`` fallback."""

    source = os.environ if environment is None else environment
    raw_value = str(cli_value or source.get("UAV_MODEL_PATH", "")).strip()
    if not raw_value:
        raise TesterError("Provide --model or set UAV_MODEL_PATH to a trusted .pt file")
    candidate = Path(raw_value).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TesterError(f"Model checkpoint does not exist: {candidate}") from error
    if not resolved.is_file():
        raise TesterError(f"Model checkpoint is not a file: {resolved}")
    if resolved.suffix.lower() != ".pt":
        raise TesterError("Model checkpoint must be a trusted .pt file")
    return resolved


def resolve_mcdo_v2_model_path(
    cli_value: str | None, environment: Mapping[str, str] | None = None
) -> Path | None:
    """Resolve the optional external V2 checkpoint without inventing a fallback."""

    source = os.environ if environment is None else environment
    raw_value = str(cli_value or source.get("UAV_MCDO_V2_MODEL_PATH", "")).strip()
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TesterError(f"V2 MC Dropout checkpoint does not exist: {candidate}") from error
    if not resolved.is_file():
        raise TesterError(f"V2 MC Dropout checkpoint is not a file: {resolved}")
    if resolved.suffix.lower() != ".pt":
        raise TesterError("V2 MC Dropout checkpoint must be a trusted .pt file")
    return resolved


def windows_desktop_path() -> Path:
    """Resolve the current Windows Desktop, including folder redirection."""

    if os.name == "nt":
        try:
            import winreg

            key_path = (
                r"Software\Microsoft\Windows\CurrentVersion\Explorer"
                r"\User Shell Folders"
            )
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(value))).expanduser()
        except (OSError, ImportError):
            pass
    return Path.home() / "Desktop"


def discover_first_mp4(search_directories: Sequence[Path] | None = None) -> Path:
    """Find the newest MP4 in the first search directory containing one."""

    directories = search_directories or (
        PROJECT_ROOT,
        windows_desktop_path(),
        Path.home() / "Videos",
        Path.cwd(),
    )
    visited: set[Path] = set()
    for raw_directory in directories:
        try:
            directory = Path(raw_directory).expanduser().resolve(strict=True)
        except OSError:
            continue
        if directory in visited or not directory.is_dir():
            continue
        visited.add(directory)
        try:
            candidates = [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            ]
        except OSError:
            continue
        ranked: list[tuple[int, str, Path]] = []
        for path in candidates:
            try:
                ranked.append((-path.stat().st_mtime_ns, path.name.casefold(), path.resolve()))
            except OSError:
                continue
        if ranked:
            ranked.sort()
            return ranked[0][2]
    raise TesterError(
        "No MP4 was found in the repository root, Windows Desktop, user Videos "
        "directory, or current directory. Provide --video instead."
    )


def resolve_video_path(
    cli_value: Path | None,
    *,
    auto_detect: bool,
    environment: Mapping[str, str] | None = None,
    search_directories: Sequence[Path] | None = None,
) -> Path:
    """Resolve an explicit/configured MP4 or automatically discover one."""

    source = os.environ if environment is None else environment
    raw_value = str(cli_value or source.get("UAV_VIDEO_PATH", "")).strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise TesterError(f"Video file does not exist: {candidate}") from error
        if not resolved.is_file() or resolved.suffix.lower() != ".mp4":
            raise TesterError(f"Video source must be an existing MP4 file: {resolved}")
        return resolved
    if auto_detect:
        return discover_first_mp4(search_directories)
    raise TesterError("Provide --video, use --auto-video, or set UAV_VIDEO_PATH")


def select_video_with_dialog(
    *,
    tk_factory: Callable[[], Any] | None = None,
    askopenfilename: Callable[..., str] | None = None,
) -> Path:
    """Open a native picker and return the selected local MP4."""

    if tk_factory is None or askopenfilename is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise TesterError(
                "The video picker requires Tkinter in the selected Python environment"
            ) from error
        tk_factory = tk.Tk
        askopenfilename = filedialog.askopenfilename
    try:
        root = tk_factory()
    except Exception as error:
        raise TesterError(f"Could not create the video selection dialog: {error}") from error
    try:
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = askopenfilename(
            parent=root,
            title="Select an MP4 video for UAV model testing",
            filetypes=(("MP4 video", "*.mp4"),),
        )
    except Exception as error:
        raise TesterError(f"Video selection dialog failed: {error}") from error
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    if not selected:
        raise TesterError("Video selection was cancelled")
    return resolve_video_path(Path(selected), auto_detect=False, environment={})


def resolve_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
    cuda_device_count: int | None = None,
) -> str:
    """Resolve automatic, CPU, or explicit CUDA execution."""

    normalized = str(requested).strip().lower() or "auto"
    if cuda_available is None or cuda_device_count is None:
        try:
            import torch
        except ImportError as error:
            raise TesterError("PyTorch is unavailable in this Python environment") from error
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    if normalized == "auto":
        return "0" if cuda_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"gpu", "cuda"}:
        normalized = "0"
    elif normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.isdigit():
        index = int(normalized)
        if not cuda_available:
            raise TesterError(
                "CUDA was requested but is unavailable; use --device auto or --device cpu"
            )
        if index >= int(cuda_device_count):
            raise TesterError(
                f"CUDA device {index} is unavailable; detected {cuda_device_count} device(s)"
            )
        return str(index)
    raise TesterError("--device must be auto, cpu, gpu, cuda, cuda:N, or a GPU index")


def device_display_name(device: str) -> str:
    """Return a concise human-facing inference device label."""

    return "CPU" if device == "cpu" else f"CUDA:{device}"


def validate_numeric_options(args: argparse.Namespace) -> None:
    """Validate inference, pacing, and uncertainty arguments."""

    if not 0.0 <= args.conf <= 1.0:
        raise TesterError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise TesterError("--iou must be between 0 and 1")
    if args.imgsz <= 0 or args.imgsz > 8192:
        raise TesterError("--imgsz must be between 1 and 8192")
    if not math.isfinite(args.max_fps) or not 0 <= args.max_fps <= 240:
        raise TesterError("--max-fps must be between 0 and 240")
    if not 1 <= args.uncertainty_samples <= 50:
        raise TesterError("--uncertainty-samples must be between 1 and 50")
    if not 0.0 < args.uncertainty_match_iou <= 1.0:
        raise TesterError("--uncertainty-match-iou must be greater than 0 and at most 1")
    if not 1 <= args.mcdo_v2_passes <= 100:
        raise TesterError("--mcdo-v2-passes must be between 1 and 100")
    if not 0.0 < args.mcdo_v2_match_iou <= 1.0:
        raise TesterError("--mcdo-v2-match-iou must be greater than 0 and at most 1")

    video_mode = args.video is not None or args.auto_video or args.select_video
    manual_region = any(
        value is not None for value in (args.left, args.top, args.width, args.height)
    )
    if args.test_frame and video_mode:
        raise TesterError("--test-frame cannot be combined with a video source")
    if video_mode and (manual_region or args.select_region):
        raise TesterError("Video mode cannot be combined with screen-region options")
    if args.loop_video and not video_mode:
        raise TesterError("--loop-video requires a video source")


def capture_region_from_args(
    args: argparse.Namespace,
    monitors: Sequence[CaptureRegion],
    *,
    cv2_module: Any,
) -> CaptureRegion:
    """Resolve manual, interactive, or full-monitor capture configuration."""

    manual_values = (args.left, args.top, args.width, args.height)
    supplied = [value is not None for value in manual_values]
    if any(supplied) and not all(supplied):
        raise TesterError("Specify --left, --top, --width, and --height together")
    if all(supplied):
        if args.select_region:
            raise TesterError("Manual coordinates cannot be combined with --select-region")
        return validate_manual_region(
            CaptureRegion(*(int(value) for value in manual_values)), monitors
        )
    monitor = monitor_by_number(monitors, args.monitor)
    if args.select_region:
        from .sources import select_region_interactively

        return select_region_interactively(monitor, cv2_module=cv2_module)
    return monitor


def build_parser() -> argparse.ArgumentParser:
    """Build the live tester command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Read a local MP4 or capture a Windows desktop region, run a trusted "
            "YOLO model, and inspect V1 input robustness or validated V2 MC "
            "Dropout model uncertainty on demand."
        )
    )
    parser.add_argument("--model", help="Trusted local .pt model (or UAV_MODEL_PATH)")
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--video", type=Path, help="Read one local MP4")
    video_group.add_argument(
        "--auto-video", action="store_true", help="Find the first suitable local MP4"
    )
    video_group.add_argument(
        "--select-video", action="store_true", help="Open an MP4 file picker"
    )
    parser.add_argument("--loop-video", action="store_true", help="Restart at EOF")
    parser.add_argument("--monitor", type=int, default=1, help="One-based monitor")
    parser.add_argument("--list-monitors", action="store_true")
    parser.add_argument("--select-region", action="store_true")
    parser.add_argument("--left", type=int)
    parser.add_argument("--top", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-fps", type=float, default=0.0)
    parser.add_argument("--classes", default="all")
    parser.add_argument(
        "--tank-only", action="store_true", help=f"Display only {TANK_CLASS_NAME}"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--test-frame", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--uncertainty-samples", type=int, default=10)
    parser.add_argument("--uncertainty-seed", type=int, default=42)
    parser.add_argument("--uncertainty-match-iou", type=float, default=0.50)
    parser.add_argument(
        "--mcdo-v2-model",
        help="Trusted external V2 checkpoint (or UAV_MCDO_V2_MODEL_PATH)",
    )
    parser.add_argument("--mcdo-v2-passes", type=int, default=20)
    parser.add_argument("--mcdo-v2-match-iou", type=float, default=0.50)
    parser.add_argument(
        "--disable-uncertainty",
        action="store_true",
        help="Hide the U-key uncertainty inspector",
    )
    return parser


def print_monitors(monitors: Sequence[CaptureRegion]) -> None:
    """Print physical monitor indices and bounds."""

    for index, monitor in enumerate(monitors, start=1):
        print(
            f"Monitor {index}: left={monitor.left} top={monitor.top} "
            f"width={monitor.width} height={monitor.height}"
        )
