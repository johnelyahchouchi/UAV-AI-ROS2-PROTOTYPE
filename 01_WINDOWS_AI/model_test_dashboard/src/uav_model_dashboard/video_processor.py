"""Frame-by-frame Ultralytics inference and report generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import cv2

from .configuration import (
    DISPLAY_TABLE_LIMIT,
    InferenceMode,
    ProcessingSettings,
    model_location_warning,
)
from .detection_records import (
    CSV_FIELDS,
    DetectionAccumulator,
    records_from_result,
)
from .errors import DashboardError, ProcessingCancelled
from .model_manager import DeviceInfo, ModelManager
from .output_manager import OutputManager, PublishedOutputs
from .processing_control import ProcessingController
from .source_adapter import UploadedVideoSource, VideoMetadata


ProgressCallback = Callable[..., None]


@dataclass(frozen=True)
class ProcessingRequest:
    """Inputs for one dashboard run."""

    video_path: Path
    model_path: Path
    settings: ProcessingSettings


@dataclass(frozen=True)
class ProcessingResult:
    """Successful outputs and UI-ready data."""

    outputs: PublishedOutputs
    detection_rows: list[list[object]]
    class_count_rows: list[list[object]]
    summary: dict[str, object]
    model_warning: str | None


class VideoProcessor:
    """Orchestrate source, cached model, output, and cancellation services."""

    def __init__(
        self,
        model_manager: ModelManager,
        output_manager: OutputManager,
        controller: ProcessingController,
        *,
        source_factory: Callable[[Path], Any] = UploadedVideoSource,
        table_limit: int = DISPLAY_TABLE_LIMIT,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model_manager = model_manager
        self.output_manager = output_manager
        self.controller = controller
        self.source_factory = source_factory
        self.table_limit = table_limit
        self.clock = clock

    def process(
        self,
        request: ProcessingRequest,
        *,
        progress: ProgressCallback | None = None,
        copy_input: bool = True,
    ) -> ProcessingResult:
        """Process a complete uploaded video or cleanly raise a typed error."""
        token = self.controller.begin()
        paths = None
        source = None
        writer = None
        csv_file = None
        total_start = self.clock()

        try:
            token.raise_if_cancelled()
            paths = self.output_manager.prepare_job(
                request.video_path,
                token.job_id,
                copy_input=copy_input,
            )
            source = self.source_factory(paths.owned_input)
            metadata: VideoMetadata = source.metadata
            device = self.model_manager.resolve_device(request.settings.device)

            writer = self._open_writer(paths.intermediate_video, metadata)
            csv_file = paths.csv_report.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()

            accumulator = DetectionAccumulator()
            table_rows: list[list[object]] = []
            processed_frames = 0
            core_start = self.clock()

            with self.model_manager.acquire(request.model_path) as handle:
                for frame in source.frames():
                    token.raise_if_cancelled()
                    result = self._infer(
                        handle.model,
                        frame.image,
                        request.settings,
                        device,
                    )
                    token.raise_if_cancelled()

                    annotated = result.plot()
                    if (
                        annotated is None
                        or annotated.shape[0] != metadata.height
                        or annotated.shape[1] != metadata.width
                    ):
                        raise DashboardError(
                            "ANNOTATION_FRAME_INVALID",
                            f"Annotated frame {frame.number} has unexpected dimensions.",
                        )
                    writer.write(annotated)
                    records = records_from_result(
                        result,
                        frame.number,
                        frame.timestamp_seconds,
                    )
                    for record in records:
                        csv_writer.writerow(record.to_csv_row())
                        if len(table_rows) < self.table_limit:
                            table_rows.append(record.to_table_row())
                    accumulator.add(records)
                    processed_frames = frame.number

                    if progress is not None:
                        fraction = (
                            min(1.0, processed_frames / metadata.frame_count)
                            if metadata.frame_count > 0
                            else 0.0
                        )
                        progress(
                            fraction,
                            desc=(
                                f"Frame {processed_frames}"
                                + (
                                    f" of {metadata.frame_count}"
                                    if metadata.frame_count > 0
                                    else ""
                                )
                            ),
                        )

            core_elapsed = max(self.clock() - core_start, 1e-9)
            source.close()
            source = None
            writer.release()
            writer = None
            csv_file.flush()
            csv_file.close()
            csv_file = None

            if processed_frames <= 0:
                raise DashboardError(
                    "VIDEO_NO_PROCESSED_FRAMES",
                    "No video frame reached the inference stage.",
                )

            token.raise_if_cancelled()
            if progress is not None:
                progress(1.0, desc="Encoding browser-compatible H.264 MP4")
            self.output_manager.transcode_h264(
                paths.intermediate_video,
                paths.encoded_video,
                cancelled=lambda: token.is_cancelled,
            )
            token.raise_if_cancelled()
            self.output_manager.validate_final_video(paths.encoded_video, metadata)
            published = self.output_manager.publish(paths)
            paths = None

            total_elapsed = max(self.clock() - total_start, 1e-9)
            summary = self._summary(
                accumulator=accumulator,
                metadata=metadata,
                processed_frames=processed_frames,
                total_elapsed=total_elapsed,
                core_elapsed=core_elapsed,
                device=device,
                model_path=request.model_path,
                mode=request.settings.mode,
                table_truncated=accumulator.total > len(table_rows),
            )
            return ProcessingResult(
                outputs=published,
                detection_rows=table_rows,
                class_count_rows=accumulator.sorted_class_counts(),
                summary=summary,
                model_warning=model_location_warning(request.model_path),
            )
        except ProcessingCancelled:
            raise
        except DashboardError:
            raise
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" in message:
                raise DashboardError(
                    "GPU_OUT_OF_MEMORY",
                    "GPU memory was exhausted during inference.",
                    recovery="Lower the image size or select CPU, then try again.",
                    detail=str(error),
                ) from error
            if "cuda" in message:
                raise DashboardError(
                    "GPU_PROCESSING_FAILED",
                    "CUDA inference failed.",
                    recovery="Verify UAV_YOLO_ENV and the NVIDIA driver, or explicitly select CPU.",
                    detail=str(error),
                ) from error
            raise DashboardError(
                "PROCESSING_FAILED",
                "Video processing failed.",
                detail=str(error),
            ) from error
        except Exception as error:
            raise DashboardError(
                "PROCESSING_FAILED",
                "Unexpected video-processing failure.",
                detail=str(error),
            ) from error
        finally:
            if source is not None:
                source.close()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
            if paths is not None and paths.staging_dir.exists():
                self.output_manager.cleanup_staging(paths.staging_dir)
            self.controller.finish(token)

    @staticmethod
    def _open_writer(path: Path, metadata: VideoMetadata) -> Any:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            writer.release()
            raise DashboardError(
                "VIDEO_WRITER_FAILED",
                "OpenCV could not create the annotated intermediate video.",
                recovery="Check output permissions, disk space, and codec support.",
            )
        return writer

    @staticmethod
    def _infer(
        model: Any,
        frame: Any,
        settings: ProcessingSettings,
        device: DeviceInfo,
    ) -> Any:
        kwargs = {
            "source": frame,
            "conf": settings.confidence,
            "iou": settings.iou,
            "imgsz": settings.image_size,
            "device": device.argument,
            "verbose": False,
        }
        if settings.mode is InferenceMode.BOTSORT:
            results = model.track(
                **kwargs,
                tracker="botsort.yaml",
                persist=True,
            )
        else:
            results = model.predict(**kwargs)
        if not results:
            raise DashboardError(
                "INFERENCE_RESULT_MISSING",
                "Ultralytics returned no frame result.",
            )
        return results[0]

    @staticmethod
    def _summary(
        *,
        accumulator: DetectionAccumulator,
        metadata: VideoMetadata,
        processed_frames: int,
        total_elapsed: float,
        core_elapsed: float,
        device: DeviceInfo,
        model_path: Path,
        mode: InferenceMode,
        table_truncated: bool,
    ) -> dict[str, object]:
        return {
            "total_detections": accumulator.total,
            "counts_per_class": dict(
                sorted(accumulator.class_counts.items())
            ),
            "average_confidence": (
                round(accumulator.average_confidence, 6)
                if accumulator.average_confidence is not None
                else None
            ),
            "maximum_confidence": (
                round(accumulator.maximum_confidence, 6)
                if accumulator.maximum_confidence is not None
                else None
            ),
            "processed_frames": processed_frames,
            "processing_time_seconds": round(total_elapsed, 3),
            "average_fps": round(processed_frames / core_elapsed, 3),
            "inference_device": device.display_name,
            "mode": mode.value,
            "model": str(model_path.resolve(strict=False)),
            "resolution": f"{metadata.width}x{metadata.height}",
            "source_fps": round(metadata.fps, 6),
            "detection_table_truncated": table_truncated,
        }
