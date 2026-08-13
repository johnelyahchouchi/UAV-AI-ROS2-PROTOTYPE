"""Gradio interface for local uncertainty experiment analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import uuid

import cv2
import gradio as gr

from .annotation import annotate_detection_records, write_annotated_image
from .comparison import (
    COMPARISON_HEADERS,
    comparison_rows,
    parse_sample_counts,
    write_comparison_csv,
)
from .configuration import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MATCH_IOU,
    DEFAULT_NMS_IOU,
    DEFAULT_OVERLAP_IOU,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_VIDEO_INTERVAL_SECONDS,
    DEFAULT_VIDEO_MAX_FRAMES,
    IMAGE_SIZE_CHOICES,
    METHOD_INPUT_PERTURBATION_V1,
    SAMPLE_COUNT_CHOICES,
    DeviceChoice,
    ExperimentSettings,
    InputKind,
    VideoSamplingMode,
    dashboard_root,
    default_model_path,
    repository_path_warning,
    validate_input_path,
    validate_model_path,
)
from .diagnostics import OVERLAP_HEADERS, instability_events, overlapping_cluster_rows
from .errors import DashboardError, ProcessingCancelled
from .experiment_runner import (
    DetectorCache,
    ExperimentRunner,
    ImageExperimentRequest,
    VideoExperimentRequest,
)
from .output_manager import OutputManager
from .processing_control import ProcessingController
from .result_loader import (
    FAMILY_HEADERS,
    SAMPLE_HEADERS,
    TARGET_HEADERS,
    family_rows,
    list_saved_runs,
    load_run,
    overview_markdown,
    sample_rows,
    schema_note,
    target_detail,
    target_rows,
)
from .result_models import LoadedExperiment
from .visualizations import (
    comparison_figure,
    sample_detection_figure,
    target_metrics_figure,
    video_timeline_figure,
)


OUTPUT_MANAGER = OutputManager(dashboard_root() / "outputs")
PROCESSING_CONTROLLER = ProcessingController()
DETECTOR_CACHE = DetectorCache()
EXPERIMENT_RUNNER = ExperimentRunner(
    DETECTOR_CACHE,
    OUTPUT_MANAGER,
    PROCESSING_CONTROLLER,
)

VIDEO_HEADERS = (
    "Selection",
    "Frame index",
    "Timestamp (s)",
    "Target clusters",
    "Dominant classes",
    "Mean persistence",
    "Mean confidence",
    "Mean class agreement",
    "Mean entropy",
    "Mean IoU",
    "Instability events",
)


def use_uploaded_model(uploaded: str | None, current: str) -> str:
    """Use a browsed .pt file when present."""
    return str(uploaded) if uploaded else current


def selected_input_warning(path_value: str | None) -> str:
    """Show repository-local runtime guidance without rejecting an input."""
    if not path_value:
        return ""
    return repository_path_warning(Path(path_value).expanduser()) or ""


def request_cancellation() -> str:
    """Request cancellation outside the queued GPU event."""
    if PROCESSING_CONTROLLER.request_cancel():
        return "Cancellation requested. The current inference call will finish before cleanup."
    return "No experiment is currently running."


def _settings(
    method: object,
    samples: object,
    seed: object,
    image_size: object,
    confidence: object,
    nms_iou: object,
    match_iou: object,
    device: object,
    overlap_iou: object,
) -> ExperimentSettings:
    return ExperimentSettings.from_values(
        method,
        samples,
        seed,
        image_size,
        confidence,
        nms_iou,
        match_iou,
        device,
        overlap_iou,
    )


def _video_request(
    video_path: str | None,
    model_path: Path,
    settings: ExperimentSettings,
    sampling_mode: object,
    interval_seconds: object,
    manual_timestamp_text: str,
    maximum_frames: object,
) -> VideoExperimentRequest:
    try:
        interval = float(interval_seconds)
        maximum = int(maximum_frames)
        mode = VideoSamplingMode(str(sampling_mode))
    except (TypeError, ValueError) as error:
        raise DashboardError("VIDEO_SETTINGS_INVALID", "Video sampling settings are invalid.") from error
    return VideoExperimentRequest(
        video_path=validate_input_path(video_path, InputKind.VIDEO),
        model_path=model_path,
        settings=settings,
        sampling_mode=mode,
        interval_seconds=interval,
        manual_timestamp_text=manual_timestamp_text,
        maximum_frames=maximum,
    )


def estimate_video(
    video_path: str | None,
    samples: object,
    sampling_mode: object,
    interval_seconds: object,
    manual_timestamp_text: str,
    maximum_frames: object,
) -> dict[str, object]:
    """Return a safe preflight estimate without loading the detector."""
    try:
        settings = ExperimentSettings.from_values(
            METHOD_INPUT_PERTURBATION_V1,
            samples,
            DEFAULT_SEED,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_CONFIDENCE,
            DEFAULT_NMS_IOU,
            DEFAULT_MATCH_IOU,
            DeviceChoice.AUTO.value,
        )
        request = _video_request(
            video_path,
            Path("unused.pt"),
            settings,
            sampling_mode,
            interval_seconds,
            manual_timestamp_text,
            maximum_frames,
        )
        return EXPERIMENT_RUNNER.video_estimate(request)
    except DashboardError as error:
        raise gr.Error(error.user_message()) from error


def _detail_directory(run: LoadedExperiment) -> Path:
    if run.video_summary and run.video_summary.get("frames"):
        return run.directory / str(run.video_summary["frames"][0]["directory"])
    return run.directory


def _video_table(run: LoadedExperiment) -> list[list[object]]:
    if not run.video_summary:
        return []
    return [
        [
            row["selection_index"],
            row["frame_index"],
            row["timestamp_seconds"],
            row["target_count"],
            ", ".join(row["dominant_classes"]),
            row["mean_persistence"],
            row["mean_confidence"],
            row["mean_class_agreement"],
            row["mean_entropy"],
            row["mean_iou"],
            " | ".join(row["instability_events"]),
        ]
        for row in run.video_summary.get("frames", [])
    ]


def _exports(root: Path, detail: Path) -> list[str]:
    candidates = (
        detail / "summary.json",
        detail / "targets.csv",
        detail / "sample_metadata.json",
        detail / "previews" / "sample_000.jpg",
        root / "dashboard_metadata.json",
        root / "video_summary.json",
        root / "video_frames.csv",
    )
    return [str(path) for path in candidates if path.is_file()]


def _render(run: LoadedExperiment, status: str) -> tuple[object, ...]:
    detail = _detail_directory(run)
    targets = list(run.summary.get("targets", []))
    overlaps = overlapping_cluster_rows(
        targets,
        float(run.metadata.get("configuration", {}).get("overlap_iou", DEFAULT_OVERLAP_IOU)),
    )
    events = instability_events(targets, overlaps)
    target_choices = [str(target["target_id"]) for target in targets]
    selected_target = target_choices[0] if target_choices else None
    sample_choices = [str(item["sample_index"]) for item in run.samples]
    selected_sample = sample_choices[0] if sample_choices else None
    preview = detail / "previews" / "sample_000.jpg"
    video_records = list((run.video_summary or {}).get("frames", []))
    video_choices = [str(item["selection_index"]) for item in video_records]
    state = {"root": str(run.directory), "detail": str(detail)}
    event_text = "\n".join(f"- {item}" for item in events) or "No instability event rule was triggered."
    return (
        status,
        state,
        overview_markdown(run) + f"\n\n{schema_note(run)}",
        target_metrics_figure(run.summary),
        target_rows(run.summary),
        gr.Dropdown(choices=target_choices, value=selected_target),
        target_detail(run.summary, selected_target or ""),
        sample_rows(run.samples),
        family_rows(run.samples),
        sample_detection_figure(run.samples),
        gr.Dropdown(choices=sample_choices, value=selected_sample),
        str(preview) if preview.is_file() else None,
        run.samples[0] if run.samples else {},
        overlaps,
        event_text,
        _video_table(run),
        video_timeline_figure(video_records) if video_records else None,
        gr.Dropdown(choices=video_choices, value=video_choices[0] if video_choices else None),
        run.summary,
        run.samples,
        run.metadata,
        target_rows(run.summary),
        _exports(run.directory, detail),
    )


def run_experiment(
    input_kind: object,
    image_path: str | None,
    video_path: str | None,
    model_path_text: str,
    model_upload: str | None,
    method: object,
    samples: object,
    seed: object,
    image_size: object,
    confidence: object,
    nms_iou: object,
    match_iou: object,
    device: object,
    overlap_iou: object,
    sampling_mode: object,
    interval_seconds: object,
    manual_timestamp_text: str,
    maximum_frames: object,
    progress: gr.Progress = gr.Progress(),
) -> tuple[object, ...]:
    """Validate UI fields, run one image/video job, and populate every result tab."""
    try:
        settings = _settings(
            method,
            samples,
            seed,
            image_size,
            confidence,
            nms_iou,
            match_iou,
            device,
            overlap_iou,
        )
        model_path = validate_model_path(model_upload or model_path_text)

        def report(fraction: float, description: str) -> None:
            progress(fraction, desc=description)

        if InputKind(str(input_kind)) is InputKind.IMAGE:
            request = ImageExperimentRequest(
                validate_input_path(image_path, InputKind.IMAGE),
                model_path,
                settings,
            )
            result = EXPERIMENT_RUNNER.run_image(request, progress=report)
        else:
            request = _video_request(
                video_path,
                model_path,
                settings,
                sampling_mode,
                interval_seconds,
                manual_timestamp_text,
                maximum_frames,
            )
            result = EXPERIMENT_RUNNER.run_video(request, progress=report)
        return _render(
            result.loaded,
            f"Experiment completed. Saved to `{result.published.directory}`.",
        )
    except ProcessingCancelled as error:
        raise gr.Error(error.user_message()) from error
    except DashboardError as error:
        detail = f" Technical detail: {error.detail}" if error.detail else ""
        raise gr.Error(error.user_message() + detail) from error


def refresh_saved_runs() -> gr.Dropdown:
    """Refresh saved-run choices without inference."""
    runs = list_saved_runs(OUTPUT_MANAGER.output_root)
    choices = [(run.display_name, str(run.directory)) for run in runs]
    return gr.Dropdown(choices=choices, multiselect=True, value=[])


def load_saved_run(path_values: list[str] | str | None) -> tuple[object, ...]:
    """Reopen one selected run in all analysis tabs."""
    if not path_values:
        raise gr.Error("Select a saved run first.")
    path = path_values[0] if isinstance(path_values, list) else path_values
    run = load_run(Path(path))
    return _render(run, f"Loaded saved run `{run.directory.name}` without inference.")


def update_sample_view(
    state: dict[str, str] | None,
    sample_value: str | None,
    target_id: str | None,
) -> tuple[str | None, dict[str, object], dict[str, object]]:
    """Render a selected sample and visibly emphasize the selected target."""
    if not state or sample_value is None:
        return None, {}, {}
    detail = Path(state["detail"])
    run = load_run(detail)
    sample_index = int(sample_value)
    sample = next((item for item in run.samples if int(item["sample_index"]) == sample_index), None)
    if sample is None:
        return None, {}, {}
    raw_path = detail / "samples" / f"sample_{sample_index:03d}.jpg"
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, sample, target_detail(run.summary, target_id or "")
    annotated = annotate_detection_records(
        image,
        list(sample.get("detections", [])),
        selected_target_id=target_id,
    )
    selected = target_id or "all"
    output = detail / "interactive" / f"sample_{sample_index:03d}_{selected}.jpg"
    write_annotated_image(output, annotated)
    return str(output), sample, target_detail(run.summary, target_id or "")


def select_video_frame(
    state: dict[str, str] | None,
    selection_value: str | None,
) -> tuple[object, ...]:
    """Open one sampled video frame in the standard target/sample analysis views."""
    if not state or selection_value is None:
        raise gr.Error("Select a sampled video frame.")
    root_run = load_run(Path(state["root"]))
    if not root_run.video_summary:
        raise gr.Error("The current run is not a video experiment.")
    record = next(
        item
        for item in root_run.video_summary["frames"]
        if int(item["selection_index"]) == int(selection_value)
    )
    detail = root_run.directory / str(record["directory"])
    frame_run = load_run(detail)
    targets = list(frame_run.summary.get("targets", []))
    choices = [str(target["target_id"]) for target in targets]
    sample_choices = [str(item["sample_index"]) for item in frame_run.samples]
    new_state = {"root": str(root_run.directory), "detail": str(detail)}
    return (
        new_state,
        target_rows(frame_run.summary),
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        target_detail(frame_run.summary, choices[0] if choices else ""),
        sample_rows(frame_run.samples),
        family_rows(frame_run.samples),
        sample_detection_figure(frame_run.samples),
        gr.Dropdown(choices=sample_choices, value=sample_choices[0] if sample_choices else None),
        str(detail / "previews" / "sample_000.jpg"),
        frame_run.samples[0] if frame_run.samples else {},
        frame_run.summary,
        frame_run.samples,
    )


def compare_saved_runs(path_values: list[str] | None) -> tuple[list[list[object]], object, str]:
    """Compare 2–4 saved runs and export the comparison table."""
    if not path_values:
        raise gr.Error("Select between two and four saved runs.")
    try:
        runs = [load_run(Path(value)) for value in path_values]
        rows = comparison_rows(runs)
        output_dir = OUTPUT_MANAGER.output_root / "comparisons"
        name = datetime.now(timezone.utc).strftime("comparison_%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8] + ".csv"
        path = output_dir / name
        write_comparison_csv(path, rows)
        return rows, comparison_figure(runs), str(path)
    except DashboardError as error:
        raise gr.Error(error.user_message()) from error


def run_sample_batch(
    image_path: str | None,
    model_path_text: str,
    model_upload: str | None,
    batch_counts: str,
    seed: object,
    image_size: object,
    confidence: object,
    nms_iou: object,
    match_iou: object,
    device: object,
    overlap_iou: object,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[list[object]], object, str, gr.Dropdown]:
    """Run sample-count configurations sequentially, then compare together."""
    try:
        counts = parse_sample_counts(batch_counts)
        settings = _settings(
            METHOD_INPUT_PERTURBATION_V1,
            counts[0],
            seed,
            image_size,
            confidence,
            nms_iou,
            match_iou,
            device,
            overlap_iou,
        )
        request = ImageExperimentRequest(
            validate_input_path(image_path, InputKind.IMAGE),
            validate_model_path(model_upload or model_path_text),
            settings,
        )

        def report(fraction: float, description: str) -> None:
            progress(fraction, desc=description)

        results = EXPERIMENT_RUNNER.run_image_batch(request, counts, progress=report)
        runs = [result.loaded for result in results]
        rows = comparison_rows(runs)
        output_dir = OUTPUT_MANAGER.output_root / "comparisons"
        path = output_dir / (datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8] + ".csv")
        write_comparison_csv(path, rows)
        saved = list_saved_runs(OUTPUT_MANAGER.output_root)
        choices = [(run.display_name, str(run.directory)) for run in saved]
        selected = [str(run.directory) for run in runs]
        return (
            f"Completed {len(runs)} sequential experiments. No concurrent model inference was used.",
            rows,
            comparison_figure(runs),
            str(path),
            gr.Dropdown(choices=choices, multiselect=True, value=selected),
        )
    except (DashboardError, ProcessingCancelled) as error:
        raise gr.Error(error.user_message()) from error


def clear_results() -> tuple[object, ...]:
    """Clear browser state without deleting saved runs."""
    return (
        "Ready.",
        None,
        "",
        None,
        [],
        gr.Dropdown(choices=[], value=None),
        {},
        [],
        [],
        None,
        gr.Dropdown(choices=[], value=None),
        None,
        {},
        [],
        "",
        [],
        None,
        gr.Dropdown(choices=[], value=None),
        {},
        [],
        {},
        [],
        [],
    )


def build_app() -> gr.Blocks:
    """Create the loopback-only dashboard without launching it."""
    with gr.Blocks(title="UAV Model Uncertainty Dashboard", delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# UAV Model Uncertainty Dashboard")
        gr.Markdown("Detection stability and Monte Carlo experiment analysis")
        run_state = gr.State()

        with gr.Tabs():
            with gr.Tab("Run Experiment"):
                with gr.Row():
                    input_kind = gr.Radio(
                        choices=[item.value for item in InputKind],
                        value=InputKind.IMAGE.value,
                        label="Input type",
                    )
                    method = gr.Dropdown(
                        choices=[METHOD_INPUT_PERTURBATION_V1],
                        value=METHOD_INPUT_PERTURBATION_V1,
                        label="Implemented method",
                    )
                with gr.Row():
                    image_input = gr.Image(label="Image", sources=["upload"], type="filepath")
                    video_input = gr.Video(label="Video", sources=["upload"], format=None)
                with gr.Row():
                    model_path = gr.Textbox(label="YOLO model path", value=str(default_model_path()))
                    model_upload = gr.File(label="Browse for .pt model", file_types=[".pt"], type="filepath")
                input_warning = gr.Markdown()
                with gr.Row():
                    device = gr.Radio(
                        choices=[item.value for item in DeviceChoice],
                        value=DeviceChoice.AUTO.value,
                        label="Device",
                    )
                    image_size = gr.Dropdown(
                        choices=list(IMAGE_SIZE_CHOICES),
                        value=DEFAULT_IMAGE_SIZE,
                        allow_custom_value=True,
                        label="Inference image size",
                    )
                    samples = gr.Dropdown(
                        choices=list(SAMPLE_COUNT_CHOICES),
                        value=DEFAULT_SAMPLE_COUNT,
                        allow_custom_value=True,
                        label="Perturbed samples (total predictions = N + 1)",
                    )
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
                with gr.Row():
                    confidence = gr.Slider(0.01, 1.0, DEFAULT_CONFIDENCE, step=0.01, label="Confidence threshold")
                    nms_iou = gr.Slider(0.01, 1.0, DEFAULT_NMS_IOU, step=0.01, label="Detector NMS IoU")
                    match_iou = gr.Slider(0.01, 1.0, DEFAULT_MATCH_IOU, step=0.01, label="Target matching IoU")
                    overlap_iou = gr.Slider(0.01, 1.0, DEFAULT_OVERLAP_IOU, step=0.01, label="Overlap diagnostic IoU")
                gr.Markdown("### Video frame selection")
                with gr.Row():
                    sampling_mode = gr.Radio(
                        choices=[item.value for item in VideoSamplingMode],
                        value=VideoSamplingMode.INTERVAL.value,
                        label="Sampling mode",
                    )
                    interval_seconds = gr.Number(value=DEFAULT_VIDEO_INTERVAL_SECONDS, label="Interval seconds")
                    manual_timestamps_text = gr.Textbox(label="Manual timestamps (seconds)", placeholder="5, 10, 15, 20, 30")
                    maximum_frames = gr.Number(value=DEFAULT_VIDEO_MAX_FRAMES, precision=0, label="Maximum frames")
                estimate_button = gr.Button("Calculate video workload")
                video_estimate = gr.JSON(label="Video workload estimate")
                with gr.Row():
                    start_button = gr.Button("Start Experiment", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="stop")
                    clear_button = gr.Button("Clear Results")
                status = gr.Markdown("Ready.")

            with gr.Tab("Overview"):
                overview = gr.Markdown()
                overview_plot = gr.Plot(label="Target stability indicators")

            with gr.Tab("Target Analysis"):
                target_table = gr.Dataframe(headers=list(TARGET_HEADERS), interactive=False, wrap=True)
                with gr.Row():
                    target_selector = gr.Dropdown(label="Select target")
                    sample_selector = gr.Dropdown(label="Select clean/perturbed sample")
                target_detail_json = gr.JSON(label="Selected target metrics")
                sample_image = gr.Image(label="Detection inspection", interactive=False)
                sample_detail_json = gr.JSON(label="Selected sample and perturbation parameters")
                overlap_table = gr.Dataframe(headers=list(OVERLAP_HEADERS), interactive=False, wrap=True, label="Possible overlapping/alternative detections")
                instability_markdown = gr.Markdown()

            with gr.Tab("Perturbation Analysis"):
                sample_table = gr.Dataframe(headers=list(SAMPLE_HEADERS), interactive=False, wrap=True)
                family_table = gr.Dataframe(headers=list(FAMILY_HEADERS), interactive=False, wrap=True)
                sample_plot = gr.Plot(label="Detections by sample")

            with gr.Tab("Video Analysis"):
                video_table = gr.Dataframe(headers=list(VIDEO_HEADERS), interactive=False, wrap=True)
                video_plot = gr.Plot(label="Sampled-frame timeline")
                with gr.Row():
                    video_frame_selector = gr.Dropdown(label="Open sampled frame")
                    open_frame_button = gr.Button("Open frame in Target Analysis")

            with gr.Tab("Compare Experiments"):
                with gr.Row():
                    saved_runs = gr.Dropdown(label="Saved runs (select 2–4)", multiselect=True)
                    refresh_runs = gr.Button("Refresh saved runs")
                    load_run_button = gr.Button("Load first selected run")
                    compare_button = gr.Button("Compare selected runs", variant="primary")
                comparison_table = gr.Dataframe(headers=list(COMPARISON_HEADERS), interactive=False, wrap=True)
                comparison_plot = gr.Plot(label="Per-target metric distributions")
                comparison_download = gr.File(label="Download comparison CSV", interactive=False)
                gr.Markdown("### Sequential sample-count batch for the selected image")
                batch_counts = gr.Textbox(value="5, 10, 20, 30", label="Perturbed sample counts")
                batch_button = gr.Button("Run sequential comparison")
                batch_status = gr.Markdown()

            with gr.Tab("Raw Results"):
                raw_summary = gr.JSON(label="Core summary JSON")
                raw_target_table = gr.Dataframe(headers=list(TARGET_HEADERS), interactive=False, wrap=True, label="Raw target table")
                raw_samples = gr.JSON(label="Enriched sample metadata")
                raw_metadata = gr.JSON(label="Dashboard experiment metadata")

            with gr.Tab("Exports"):
                export_files = gr.File(label="Download generated result files", file_count="multiple", interactive=False)
                gr.Markdown("Generated runs are stored below the dashboard outputs directory and are ignored by Git.")

        render_outputs = [
            status,
            run_state,
            overview,
            overview_plot,
            target_table,
            target_selector,
            target_detail_json,
            sample_table,
            family_table,
            sample_plot,
            sample_selector,
            sample_image,
            sample_detail_json,
            overlap_table,
            instability_markdown,
            video_table,
            video_plot,
            video_frame_selector,
            raw_summary,
            raw_samples,
            raw_metadata,
            raw_target_table,
            export_files,
        ]
        run_inputs = [
            input_kind,
            image_input,
            video_input,
            model_path,
            model_upload,
            method,
            samples,
            seed,
            image_size,
            confidence,
            nms_iou,
            match_iou,
            device,
            overlap_iou,
            sampling_mode,
            interval_seconds,
            manual_timestamps_text,
            maximum_frames,
        ]
        model_upload.change(use_uploaded_model, [model_upload, model_path], model_path, queue=False)
        model_path.change(selected_input_warning, model_path, input_warning, queue=False)
        estimate_button.click(
            estimate_video,
            [video_input, samples, sampling_mode, interval_seconds, manual_timestamps_text, maximum_frames],
            video_estimate,
            queue=False,
        )
        start_button.click(
            run_experiment,
            run_inputs,
            render_outputs,
            concurrency_limit=1,
            show_progress="full",
        )
        cancel_button.click(request_cancellation, outputs=status, queue=False)
        clear_button.click(clear_results, outputs=render_outputs, queue=False)
        sample_selector.change(
            update_sample_view,
            [run_state, sample_selector, target_selector],
            [sample_image, sample_detail_json, target_detail_json],
            queue=False,
        )
        target_selector.change(
            update_sample_view,
            [run_state, sample_selector, target_selector],
            [sample_image, sample_detail_json, target_detail_json],
            queue=False,
        )
        refresh_runs.click(refresh_saved_runs, outputs=saved_runs, queue=False)
        load_run_button.click(load_saved_run, saved_runs, render_outputs, queue=False)
        compare_button.click(
            compare_saved_runs,
            saved_runs,
            [comparison_table, comparison_plot, comparison_download],
            queue=False,
        )
        batch_button.click(
            run_sample_batch,
            [
                image_input,
                model_path,
                model_upload,
                batch_counts,
                seed,
                image_size,
                confidence,
                nms_iou,
                match_iou,
                device,
                overlap_iou,
            ],
            [batch_status, comparison_table, comparison_plot, comparison_download, saved_runs],
            concurrency_limit=1,
            show_progress="full",
        )
        open_frame_button.click(
            select_video_frame,
            [run_state, video_frame_selector],
            [
                run_state,
                target_table,
                target_selector,
                target_detail_json,
                sample_table,
                family_table,
                sample_plot,
                sample_selector,
                sample_image,
                sample_detail_json,
                raw_summary,
                raw_samples,
            ],
            queue=False,
        )
    return demo


def main() -> None:
    """Launch locally on 127.0.0.1 only."""
    try:
        port = int(os.environ.get("UAV_UNCERTAINTY_DASHBOARD_PORT", "7861"))
    except ValueError as error:
        raise DashboardError(
            "INVALID_PORT",
            "UAV_UNCERTAINTY_DASHBOARD_PORT must be an integer.",
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
