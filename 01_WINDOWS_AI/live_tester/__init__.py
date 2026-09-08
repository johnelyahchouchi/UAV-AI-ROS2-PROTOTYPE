"""Reusable live screen/video model tester package."""

from .app import main
from .configuration import (
    build_parser,
    capture_region_from_args,
    choose_preview_position,
    device_display_name,
    discover_first_mp4,
    discover_monitors,
    resolve_device,
    resolve_mcdo_v2_model_path,
    resolve_model_path,
    resolve_video_path,
    select_checkpoint_with_dialog,
    select_video_with_dialog,
    validate_numeric_options,
)
from .detector import YoloDetector
from .domain import (
    CaptureRegion,
    DetectionResult,
    FrameMetrics,
    FrameOutcome,
    TesterError,
    VideoMetadata,
    VideoSourceEnded,
    clip_bbox,
    filter_detections,
    monitor_by_number,
    normalize_class_name,
    parse_class_filter,
    validate_manual_region,
)
from .renderer import OverlayRenderer
from .mc_dropout_adapter import MCDOV2LiveInspector
from .mc_dropout_adapter import mcdo_frame_summary
from .presentation_ui import (
    FontResolver,
    PillowCanvas,
    Rect,
    calculate_presentation_layout,
    presentation_canvas_size,
    render_method_selector,
)
from .runtime import (
    FrameProcessor,
    PerformanceTracker,
    ScreenshotStore,
    run_live_preview,
    run_test_frame,
    run_video_preview,
)
from .sources import MSSScreenSource, VideoFileSource, inspect_video
from .uncertainty_adapter import RobustnessInspector

__all__ = [name for name in globals() if not name.startswith("_")]
