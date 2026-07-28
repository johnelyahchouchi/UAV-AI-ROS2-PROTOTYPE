"""Gradio interface for local uploaded-video YOLO model testing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from .configuration import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    SUPPORTED_IMAGE_SIZES,
    DeviceChoice,
    InferenceMode,
    ProcessingSettings,
    dashboard_root,
    default_model_path,
    model_location_warning,
    validate_model_path,
    validate_video_path,
)
from .detection_records import CSV_FIELDS
from .errors import DashboardError, ProcessingCancelled
from .model_manager import ModelManager
from .output_manager import OutputManager
from .processing_control import ProcessingController
from .video_processor import ProcessingRequest, VideoProcessor


MODEL_MANAGER = ModelManager()
OUTPUT_MANAGER = OutputManager(dashboard_root() / "outputs")
PROCESSING_CONTROLLER = ProcessingController()
VIDEO_PROCESSOR = VideoProcessor(
    MODEL_MANAGER,
    OUTPUT_MANAGER,
    PROCESSING_CONTROLLER,
)


def resolved_device_markdown(device_value: str) -> str:
    """Resolve a device selection before processing begins."""
    try:
        return MODEL_MANAGER.describe_device(DeviceChoice(device_value))
    except DashboardError as error:
        return f"**Resolved inference device:** unavailable\n\n{error.user_message()}"


def use_uploaded_model(uploaded_path: str | None, current_path: str) -> str:
    """Prefer a browsed .pt file while retaining the external default otherwise."""
    return str(uploaded_path) if uploaded_path else current_path


def selected_model_warning(model_path: str) -> str:
    """Show repository-weight guidance as soon as a path is selected."""
    if not model_path.strip():
        return ""
    return model_location_warning(Path(model_path).expanduser()) or ""


def request_cancellation() -> str:
    """Request cooperative cancellation without entering the processing queue."""
    if PROCESSING_CONTROLLER.request_cancel():
        return "Cancellation requested. Finishing the current frame and cleaning partial output…"
    return "No processing job is currently running."


def run_processing(
    video_path: str | None,
    model_path_text: str,
    model_upload: str | None,
    confidence: float,
    iou: float,
    image_size: int,
    device: str,
    mode: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[
    str,
    str | None,
    list[list[object]],
    list[list[object]],
    dict[str, object] | None,
    str | None,
    str | None,
    str,
]:
    """Validate UI inputs and return completed processing artifacts."""
    try:
        settings = ProcessingSettings.from_values(
            confidence,
            iou,
            image_size,
            device,
            mode,
        )
        selected_model = model_upload or model_path_text
        result = VIDEO_PROCESSOR.process(
            ProcessingRequest(
                video_path=validate_video_path(video_path),
                model_path=validate_model_path(selected_model),
                settings=settings,
            ),
            progress=progress,
        )
        warning = result.model_warning or ""
        status = (
            "✅ Processing completed successfully. "
            f"Saved outputs to `{result.outputs.directory}`."
        )
        return (
            status,
            str(result.outputs.annotated_video),
            result.detection_rows,
            result.class_count_rows,
            result.summary,
            str(result.outputs.annotated_video),
            str(result.outputs.csv_report),
            warning,
        )
    except ProcessingCancelled as error:
        return (
            f"🟡 {error.message}",
            None,
            [],
            [],
            None,
            None,
            None,
            "",
        )
    except DashboardError as error:
        detail = f"\n\nTechnical detail: {error.detail}" if error.detail else ""
        raise gr.Error(error.user_message() + detail) from error


def build_app() -> gr.Blocks:
    """Create the local-only dashboard without launching it."""
    with gr.Blocks(
        title="UAV Model Test Dashboard",
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown("# UAV Model Test Dashboard")
        gr.Markdown(
            "Local uploaded-video evaluation for Ultralytics detection models. "
            "No ROS 2, TCP, camera, or internet connection is used."
        )

        with gr.Row():
            with gr.Column(scale=3):
                video_input = gr.Video(
                    label="Input video",
                    sources=["upload"],
                    format=None,
                )
            with gr.Column(scale=2):
                model_path = gr.Textbox(
                    label="YOLO model path",
                    value=str(default_model_path()),
                )
                model_upload = gr.File(
                    label="Browse for another .pt model",
                    file_types=[".pt"],
                    file_count="single",
                    type="filepath",
                )
                model_warning = gr.Markdown()

        with gr.Row():
            confidence = gr.Slider(
                0.01,
                1.00,
                value=DEFAULT_CONFIDENCE,
                step=0.01,
                label="Confidence threshold",
            )
            iou = gr.Slider(
                0.01,
                1.00,
                value=DEFAULT_IOU,
                step=0.01,
                label="IoU threshold",
            )
            image_size = gr.Dropdown(
                choices=list(SUPPORTED_IMAGE_SIZES),
                value=DEFAULT_IMAGE_SIZE,
                label="Inference image size",
            )

        with gr.Row():
            device = gr.Radio(
                choices=[choice.value for choice in DeviceChoice],
                value=DeviceChoice.AUTO.value,
                label="Inference device",
            )
            mode = gr.Radio(
                choices=[choice.value for choice in InferenceMode],
                value=InferenceMode.DETECTION.value,
                label="Processing mode",
            )

        device_status = gr.Markdown(
            value=resolved_device_markdown(DeviceChoice.AUTO.value)
        )
        with gr.Row():
            start_button = gr.Button(
                "Start Processing",
                variant="primary",
            )
            cancel_button = gr.Button(
                "Cancel Processing",
                variant="stop",
            )
        status = gr.Markdown("Ready.")

        with gr.Tabs():
            with gr.Tab("Annotated video"):
                output_video = gr.Video(
                    label="Annotated MP4",
                    interactive=False,
                    format="mp4",
                )
            with gr.Tab("Detections"):
                detection_table = gr.Dataframe(
                    headers=list(CSV_FIELDS),
                    interactive=False,
                    wrap=True,
                    label="Detection records (display capped at 10,000 rows)",
                )
            with gr.Tab("Summary"):
                summary = gr.JSON(label="Run summary")
                class_counts = gr.Dataframe(
                    headers=["class_name", "count"],
                    interactive=False,
                    label="Counts per class",
                )
            with gr.Tab("Downloads"):
                video_download = gr.File(
                    label="Download annotated MP4",
                    interactive=False,
                )
                csv_download = gr.File(
                    label="Download complete CSV report",
                    interactive=False,
                )

        model_upload.change(
            fn=use_uploaded_model,
            inputs=[model_upload, model_path],
            outputs=model_path,
            queue=False,
        )
        model_path.change(
            fn=selected_model_warning,
            inputs=model_path,
            outputs=model_warning,
            queue=False,
        )
        device.change(
            fn=resolved_device_markdown,
            inputs=device,
            outputs=device_status,
            queue=False,
        )
        start_button.click(
            fn=run_processing,
            inputs=[
                video_input,
                model_path,
                model_upload,
                confidence,
                iou,
                image_size,
                device,
                mode,
            ],
            outputs=[
                status,
                output_video,
                detection_table,
                class_counts,
                summary,
                video_download,
                csv_download,
                model_warning,
            ],
            concurrency_limit=1,
            show_progress="full",
        )
        cancel_button.click(
            fn=request_cancellation,
            outputs=status,
            queue=False,
        )

    return demo


def main() -> None:
    """Launch the dashboard on the loopback interface only."""
    try:
        port = int(os.environ.get("UAV_DASHBOARD_PORT", "7860"))
    except ValueError as error:
        raise DashboardError(
            "INVALID_PORT",
            "UAV_DASHBOARD_PORT must be an integer.",
        ) from error
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=True,
        allowed_paths=[str(OUTPUT_MANAGER.output_root)],
    )


if __name__ == "__main__":
    main()
