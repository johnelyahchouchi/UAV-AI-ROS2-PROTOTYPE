"""Command-line smoke runner for the real external model and uploaded video path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .configuration import (
    DeviceChoice,
    InferenceMode,
    ProcessingSettings,
    dashboard_root,
    default_model_path,
    validate_model_path,
    validate_video_path,
)
from .errors import DashboardError, ProcessingCancelled
from .model_manager import ModelManager
from .output_manager import OutputManager
from .processing_control import ProcessingController
from .video_processor import ProcessingRequest, VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=str(default_model_path()))
    parser.add_argument("--mode", choices=["detection", "botsort"], required=True)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--device",
        choices=["auto", "gpu0", "cpu"],
        default="gpu0",
    )
    parser.add_argument("--cancel-after-frames", type=int, default=0)
    parser.add_argument(
        "--no-copy-input",
        action="store_true",
        help="Use the supplied local video directly; the dashboard UI always stages uploads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = {
        "auto": DeviceChoice.AUTO,
        "gpu0": DeviceChoice.GPU_0,
        "cpu": DeviceChoice.CPU,
    }[args.device]
    mode = {
        "detection": InferenceMode.DETECTION,
        "botsort": InferenceMode.BOTSORT,
    }[args.mode]
    settings = ProcessingSettings.from_values(
        args.conf,
        args.iou,
        args.imgsz,
        device.value,
        mode.value,
    )
    controller = ProcessingController()
    processor = VideoProcessor(
        ModelManager(),
        OutputManager(dashboard_root() / "outputs"),
        controller,
    )
    progress_count = 0

    def progress(_: float, *, desc: str) -> None:
        nonlocal progress_count
        progress_count += 1
        if progress_count == 1 or progress_count % 100 == 0:
            print(f"[PROGRESS] {desc}", flush=True)
        if args.cancel_after_frames > 0 and progress_count >= args.cancel_after_frames:
            controller.request_cancel()

    try:
        result = processor.process(
            ProcessingRequest(
                video_path=validate_video_path(args.video),
                model_path=validate_model_path(args.model),
                settings=settings,
            ),
            progress=progress,
            copy_input=not args.no_copy_input,
        )
    except ProcessingCancelled as error:
        print(error.user_message())
        return 0
    except DashboardError as error:
        print(error.user_message())
        if error.detail:
            print(f"Detail: {error.detail}")
        return 1

    print("[PASS] Processing complete")
    print(json.dumps(result.summary, indent=2))
    print(f"Annotated MP4: {result.outputs.annotated_video}")
    print(f"Detection CSV: {result.outputs.csv_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
