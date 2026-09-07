"""Composition root and CLI entry point for the live model tester."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from uav_security.model_integrity import ModelIntegrityError

from .configuration import (
    build_parser,
    capture_region_from_args,
    device_display_name,
    discover_monitors,
    print_monitors,
    resolve_device,
    resolve_mcdo_v2_model_path,
    resolve_model_path,
    resolve_video_path,
    select_video_with_dialog,
    validate_numeric_options,
)
from .detector import YoloDetector
from .domain import CaptureRegion, TesterError, VideoMetadata, parse_class_filter
from .mc_dropout_adapter import MCDOV2LiveInspector
from .renderer import OverlayRenderer
from .runtime import (
    FrameProcessor,
    ScreenshotStore,
    run_live_preview,
    run_test_frame,
    run_video_preview,
)
from .sources import VideoFileSource, inspect_video
from .uncertainty_adapter import RobustnessInspector


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local tester CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_numeric_options(args)
        if args.list_monitors:
            print_monitors(discover_monitors())
            return 0

        model_path = resolve_model_path(args.model)
        device = resolve_device(args.device)
        allowed_classes = parse_class_filter(args.classes, tank_only=args.tank_only)
        try:
            import cv2
        except ImportError as error:
            raise TesterError(
                "OpenCV is unavailable. Install requirements-windows.txt in the "
                "controlled UAV YOLO environment."
            ) from error

        monitors: tuple[CaptureRegion, ...] = ()
        video_path: Path | None = None
        video_metadata: VideoMetadata | None = None
        if args.test_frame:
            frame = cv2.imread(str(args.test_frame.expanduser()))
            if frame is None:
                raise TesterError(f"OpenCV could not decode test frame: {args.test_frame}")
            frame_height, frame_width = frame.shape[:2]
            region = CaptureRegion(0, 0, frame_width, frame_height)
        elif args.video is not None or args.auto_video or args.select_video:
            video_path = (
                select_video_with_dialog()
                if args.select_video
                else resolve_video_path(args.video, auto_detect=args.auto_video)
            )
            video_metadata = inspect_video(video_path, cv2)
            region = CaptureRegion(0, 0, video_metadata.width, video_metadata.height)
        else:
            monitors = discover_monitors()
            print_monitors(monitors)
            region = capture_region_from_args(args, monitors, cv2_module=cv2)

        detector = YoloDetector(
            model_path,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=device,
            registry_path=args.registry,
        )
        display_device = device_display_name(device)
        uncertainty_available = not args.disable_uncertainty and not args.test_frame
        mcdo_inspector = None
        mcdo_unavailable_reason = (
            "configure UAV_MCDO_V2_MODEL_PATH with the validated external checkpoint"
        )
        if uncertainty_available:
            try:
                mcdo_model_path = resolve_mcdo_v2_model_path(args.mcdo_v2_model)
            except TesterError as error:
                mcdo_model_path = None
                mcdo_unavailable_reason = str(error)
            if mcdo_model_path is not None:
                try:
                    mcdo_inspector = MCDOV2LiveInspector.from_checkpoint(
                        mcdo_model_path,
                        device=device,
                        confidence=args.conf,
                        nms_iou=args.iou,
                        image_size=args.imgsz,
                        sample_count=args.mcdo_v2_passes,
                        match_iou=args.mcdo_v2_match_iou,
                        cv2_module=cv2,
                        registry_path=args.registry,
                    )
                except (ModelIntegrityError, TesterError, ValueError, RuntimeError) as error:
                    mcdo_unavailable_reason = str(error)
        processor = FrameProcessor(
            detector,
            OverlayRenderer(cv2),
            allowed_classes=allowed_classes,
            model_name=model_path.name,
            region=region,
            device=display_device,
            source_label=(f"video {video_path.name}" if video_path else None),
            uncertainty_available=uncertainty_available,
            mcdo_v2_available=mcdo_inspector is not None,
        )
        screenshot_store = ScreenshotStore(args.output_dir)
        inspector = None
        if uncertainty_available:
            inspector = RobustnessInspector(
                detector,
                cv2_module=cv2,
                sample_count=args.uncertainty_samples,
                seed=args.uncertainty_seed,
                match_iou=args.uncertainty_match_iou,
            )

        filter_summary = (
            "all" if allowed_classes is None else ", ".join(sorted(allowed_classes))
        )
        print(f"Model: {model_path}")
        print(f"Verified SHA-256: {detector.model_sha256}")
        print(f"Device: {display_device}")
        if args.test_frame:
            capture_mode = "test frame"
        elif video_path is not None:
            capture_mode = (
                "file-picker video"
                if args.select_video
                else "automatically selected video"
                if args.auto_video
                else "video file"
            )
        elif args.select_region:
            capture_mode = f"interactive selection on monitor {args.monitor}"
        elif args.left is not None:
            capture_mode = "manual region"
        else:
            capture_mode = f"full monitor {args.monitor}"
        print(f"Capture mode: {capture_mode}")
        if video_metadata is not None:
            print(f"Video: {video_metadata.path}")
            print(
                f"Video metadata: {video_metadata.width}x{video_metadata.height} "
                f"at {video_metadata.fps:.3f} FPS, {video_metadata.frame_count} frames"
            )
        print(
            f"Capture region: left={region.left} top={region.top} "
            f"width={region.width} height={region.height}"
        )
        print(
            f"Inference: conf={args.conf:.3f} iou={args.iou:.3f} "
            f"imgsz={args.imgsz} max_fps={args.max_fps:g}"
        )
        print(f"Display classes: {filter_summary}")
        if inspector is not None:
            print(
                "U inspection: V1 input-perturbation robustness, "
                f"1 clean + {args.uncertainty_samples} variants"
            )
            if mcdo_inspector is None:
                print(f"V2 unavailable - {mcdo_unavailable_reason}")
            else:
                report = mcdo_inspector.architecture_report
                print(
                    "V2 MC Dropout: available on demand, "
                    f"{args.mcdo_v2_passes} same-frame passes"
                )
                print(f"V2 model: {mcdo_inspector.runner.model_path}")
                print(f"V2 verified SHA-256: {mcdo_inspector.model_sha256}")
                print(
                    f"V2 architecture: {report.dropout_count} Dropout2d p="
                    f"{report.dropout_probability:.2f}; "
                    f"{report.batchnorm_count} BatchNorm eval"
                )

        if args.test_frame:
            saved = run_test_frame(
                args.test_frame, processor, screenshot_store, cv2_module=cv2
            )
            print(f"Test-frame result saved: {saved}")
            return 0
        if video_metadata is not None:
            run_video_preview(
                VideoFileSource(video_metadata, loop=args.loop_video, cv2_module=cv2),
                processor,
                screenshot_store,
                max_fps=args.max_fps,
                cv2_module=cv2,
                uncertainty_inspector=inspector,
                mcdo_v2_inspector=mcdo_inspector,
            )
            return 0
        run_live_preview(
            region,
            monitors,
            processor,
            screenshot_store,
            max_fps=args.max_fps,
            cv2_module=cv2,
            uncertainty_inspector=inspector,
            mcdo_v2_inspector=mcdo_inspector,
        )
        return 0
    except (TesterError, ModelIntegrityError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
