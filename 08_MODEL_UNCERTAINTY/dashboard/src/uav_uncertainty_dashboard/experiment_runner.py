"""Image, video, and sequential batch experiment orchestration."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
import threading
from typing import Callable, Iterator

import cv2

from uav_uncertainty.analysis_engine import AnalysisCancelled, ImageAnalysis
from uav_uncertainty.detector_adapter import UltralyticsDetector
from uav_uncertainty.mc_stability_runner import build_summary
from uav_uncertainty.perturbations import PerturbationConfig

from .configuration import ExperimentSettings, InputKind, VideoSamplingMode
from .diagnostics import instability_events, overlapping_cluster_rows
from .errors import DashboardError, ProcessingCancelled
from .methods import ExperimentMethod, method_for
from .output_manager import OutputManager
from .processing_control import CancellationToken, ProcessingController
from .result_loader import load_run
from .result_models import DASHBOARD_SCHEMA_VERSION, LoadedExperiment, PublishedRun, RunPaths
from .video_sampling import (
    interval_timestamps,
    manual_timestamps,
    probe_video,
    sample_video_frames,
    total_inference_calls,
)


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ImageExperimentRequest:
    """Validated inputs for one image experiment."""

    image_path: Path
    model_path: Path
    settings: ExperimentSettings


@dataclass(frozen=True)
class VideoExperimentRequest:
    """Validated inputs for one bounded video experiment."""

    video_path: Path
    model_path: Path
    settings: ExperimentSettings
    sampling_mode: VideoSamplingMode
    interval_seconds: float = 5.0
    manual_timestamp_text: str = ""
    maximum_frames: int = 20


@dataclass(frozen=True)
class ExperimentRunResult:
    """Completed artifacts and immediately reloadable data."""

    published: PublishedRun
    loaded: LoadedExperiment


class DetectorCache:
    """Cache one detector configuration and serialize model construction."""

    def __init__(self, factory: Callable[..., object] = UltralyticsDetector) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._key: tuple[object, ...] | None = None
        self._detector: object | None = None

    @staticmethod
    def _identity(path: Path, settings: ExperimentSettings) -> tuple[object, ...]:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        return (
            resolved,
            stat.st_size,
            stat.st_mtime_ns,
            settings.image_size,
            settings.confidence,
            settings.nms_iou,
            settings.device.argument,
        )

    def get(self, path: Path, settings: ExperimentSettings) -> object:
        """Return the unchanged cached detector or construct one."""
        key = self._identity(path, settings)
        with self._lock:
            if key == self._key and self._detector is not None:
                return self._detector
            detector = self._factory(
                path,
                image_size=settings.image_size,
                confidence=settings.confidence,
                nms_iou=settings.nms_iou,
                device=settings.device.argument,
            )
            self._key = key
            self._detector = detector
            return detector


class ExperimentRunner:
    """Run one GPU job at a time and publish only complete experiments."""

    def __init__(
        self,
        detector_cache: DetectorCache,
        output_manager: OutputManager,
        controller: ProcessingController,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.detector_cache = detector_cache
        self.output_manager = output_manager
        self.controller = controller
        self.clock = clock

    def run_image(
        self,
        request: ImageExperimentRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> ExperimentRunResult:
        """Run and publish one complete image experiment."""
        token = self.controller.begin()
        try:
            return self._run_image(request, token, token.job_id, progress=progress)
        finally:
            self.controller.finish(token)

    def run_image_batch(
        self,
        request: ImageExperimentRequest,
        sample_counts: list[int],
        *,
        progress: ProgressCallback | None = None,
    ) -> list[ExperimentRunResult]:
        """Run sample-count configurations sequentially with one cached detector."""
        if not sample_counts:
            raise DashboardError("BATCH_EMPTY", "Choose at least one sample count.")
        token = self.controller.begin()
        completed: list[ExperimentRunResult] = []
        try:
            for index, sample_count in enumerate(sample_counts, start=1):
                token.raise_if_cancelled()
                settings = replace(request.settings, sample_count=sample_count)
                child = replace(request, settings=settings)

                def child_progress(fraction: float, description: str) -> None:
                    overall = ((index - 1) + fraction) / len(sample_counts)
                    self._progress(
                        progress,
                        overall,
                        f"Experiment {index}/{len(sample_counts)} (N={sample_count}): {description}",
                    )

                completed.append(
                    self._run_image(
                        child,
                        token,
                        f"{index:02d}{token.job_id}",
                        progress=child_progress,
                    )
                )
            return completed
        finally:
            self.controller.finish(token)

    def _run_image(
        self,
        request: ImageExperimentRequest,
        token: CancellationToken,
        artifact_job_id: str,
        *,
        progress: ProgressCallback | None,
    ) -> ExperimentRunResult:
        paths: RunPaths | None = None
        try:
            token.raise_if_cancelled()
            paths = self.output_manager.prepare_run(
                request.image_path,
                method_for(request.settings.method).identity.identifier,
                artifact_job_id,
            )
            self._progress(progress, 0.02, "Loading image")
            image = cv2.imread(str(request.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise DashboardError("IMAGE_DECODE_FAILED", f"OpenCV could not decode: {request.image_path}")
            self._progress(progress, 0.05, "Loading model")
            detector = self.detector_cache.get(request.model_path, request.settings)
            token.raise_if_cancelled()
            analysis = self._analyze(
                image,
                detector,
                request.settings,
                token,
                progress,
                progress_start=0.08,
                progress_span=0.72,
            )
            self._progress(progress, 0.84, "Generating dashboard results")
            method = method_for(request.settings.method)
            summary = self._summary(
                request.model_path,
                request.image_path,
                request.settings,
                analysis,
            )
            metadata = self._metadata(
                InputKind.IMAGE,
                request.image_path,
                request.model_path,
                request.settings,
                method,
                artifacts={
                    "summary": "summary.json",
                    "targets_csv": "targets.csv",
                    "sample_metadata": "sample_metadata.json",
                    "baseline_preview": "previews/sample_000.jpg",
                    "diagnostics": "diagnostics.json",
                },
            )
            self._progress(progress, 0.90, "Saving outputs")
            self.output_manager.write_image_result(paths.staging_dir, summary, metadata, analysis)
            overlaps = overlapping_cluster_rows(summary["targets"], request.settings.overlap_iou)  # type: ignore[arg-type]
            diagnostics = {
                "overlap_iou_threshold": request.settings.overlap_iou,
                "overlapping_clusters": overlaps,
                "instability_events": instability_events(summary["targets"], overlaps),  # type: ignore[arg-type]
            }
            self.output_manager.write_json(paths.staging_dir / "diagnostics.json", diagnostics)
            token.raise_if_cancelled()
            published = self.output_manager.publish(paths)
            paths = None
            self._progress(progress, 1.0, "Experiment complete")
            return ExperimentRunResult(published, load_run(published.directory))
        except (AnalysisCancelled, ProcessingCancelled) as error:
            raise ProcessingCancelled() from error
        except DashboardError:
            raise
        except Exception as error:
            raise DashboardError("EXPERIMENT_FAILED", "Image uncertainty experiment failed.", detail=str(error)) from error
        finally:
            if paths is not None and paths.staging_dir.exists():
                self.output_manager.abort(paths)

    def run_video(
        self,
        request: VideoExperimentRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> ExperimentRunResult:
        """Run an independent V1 analysis for each selected video frame."""
        token = self.controller.begin()
        paths: RunPaths | None = None
        try:
            info = probe_video(request.video_path)
            if request.sampling_mode is VideoSamplingMode.INTERVAL:
                timestamps = interval_timestamps(
                    info.duration_seconds,
                    request.interval_seconds,
                    request.maximum_frames,
                )
            else:
                timestamps = manual_timestamps(
                    request.manual_timestamp_text,
                    info.duration_seconds,
                    request.maximum_frames,
                )
            token.raise_if_cancelled()
            method = method_for(request.settings.method)
            paths = self.output_manager.prepare_run(
                request.video_path,
                method.identity.identifier,
                token.job_id,
            )
            self._progress(progress, 0.02, "Decoding selected video frames")
            frames = sample_video_frames(request.video_path, timestamps)
            self._progress(progress, 0.05, "Loading model")
            detector = self.detector_cache.get(request.model_path, request.settings)
            frame_records: list[dict[str, object]] = []
            for index, frame in enumerate(frames, start=1):
                token.raise_if_cancelled()
                base = 0.08 + ((index - 1) / len(frames)) * 0.78
                span = 0.78 / len(frames)
                analysis = self._analyze(
                    frame.image,
                    detector,
                    request.settings,
                    token,
                    progress,
                    progress_start=base,
                    progress_span=span * 0.86,
                    prefix=f"Frame {index}/{len(frames)}",
                )
                summary = self._summary(
                    request.model_path,
                    request.video_path,
                    request.settings,
                    analysis,
                )
                summary["input"]["video_frame_index"] = frame.frame_index  # type: ignore[index]
                summary["input"]["video_timestamp_seconds"] = frame.timestamp_seconds  # type: ignore[index]
                relative_dir = Path("frames") / f"frame_{index:03d}_{int(frame.timestamp_seconds * 1000):010d}ms"
                frame_metadata = self._metadata(
                    InputKind.IMAGE,
                    request.video_path,
                    request.model_path,
                    request.settings,
                    method,
                    artifacts={},
                    extra={
                        "parent_input_kind": "Video",
                        "video_frame_index": frame.frame_index,
                        "video_timestamp_seconds": frame.timestamp_seconds,
                    },
                )
                self.output_manager.write_image_result(
                    paths.staging_dir / relative_dir,
                    summary,
                    frame_metadata,
                    analysis,
                )
                targets = list(summary["targets"])
                overlaps = overlapping_cluster_rows(targets, request.settings.overlap_iou)
                events = instability_events(targets, overlaps)
                frame_records.append(
                    {
                        "selection_index": index,
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "directory": relative_dir.as_posix(),
                        "target_count": len(targets),
                        "dominant_classes": sorted({str(target["dominant_class"]) for target in targets}),
                        "mean_persistence": self._mean(targets, "detection_persistence"),
                        "mean_confidence": self._mean(targets, "confidence_mean"),
                        "mean_class_agreement": self._mean(targets, "class_agreement"),
                        "mean_entropy": self._mean(targets, "class_entropy_bits"),
                        "mean_iou": self._mean(targets, "mean_iou_to_reference"),
                        "instability_events": events,
                    }
                )
            self._progress(progress, 0.90, "Saving video summary")
            video_summary = {
                "dashboard_schema_version": DASHBOARD_SCHEMA_VERSION,
                "method": method.identity.identifier,
                "method_name": method.identity.display_name,
                "method_version": method.identity.version,
                "input": str(request.video_path.resolve()),
                "selected_frame_count": len(frames),
                "perturbation_count_per_frame": request.settings.sample_count,
                "inference_calls_per_frame": request.settings.sample_count + 1,
                "estimated_total_inference_calls": total_inference_calls(len(frames), request.settings.sample_count),
                "sampling_mode": request.sampling_mode.value,
                "frames": frame_records,
            }
            metadata = self._metadata(
                InputKind.VIDEO,
                request.video_path,
                request.model_path,
                request.settings,
                method,
                artifacts={
                    "video_summary": "video_summary.json",
                    "video_frames_csv": "video_frames.csv",
                    "frames": "frames/",
                },
                extra={
                    "video_sampling": {
                        "mode": request.sampling_mode.value,
                        "interval_seconds": request.interval_seconds,
                        "manual_timestamps": request.manual_timestamp_text,
                        "maximum_frames": request.maximum_frames,
                    }
                },
            )
            self.output_manager.write_json(paths.staging_dir / "dashboard_metadata.json", metadata)
            self.output_manager.write_json(paths.staging_dir / "video_summary.json", video_summary)
            self._write_video_csv(paths.staging_dir / "video_frames.csv", frame_records)
            token.raise_if_cancelled()
            published = self.output_manager.publish(paths)
            paths = None
            self._progress(progress, 1.0, "Video experiment complete")
            return ExperimentRunResult(published, load_run(published.directory))
        except (AnalysisCancelled, ProcessingCancelled) as error:
            raise ProcessingCancelled() from error
        except DashboardError:
            raise
        except Exception as error:
            raise DashboardError("VIDEO_EXPERIMENT_FAILED", "Video uncertainty experiment failed.", detail=str(error)) from error
        finally:
            if paths is not None and paths.staging_dir.exists():
                self.output_manager.abort(paths)
            self.controller.finish(token)

    def video_estimate(self, request: VideoExperimentRequest) -> dict[str, object]:
        """Calculate selected frames and inference calls without decoding or inference."""
        info = probe_video(request.video_path)
        if request.sampling_mode is VideoSamplingMode.INTERVAL:
            timestamps = interval_timestamps(info.duration_seconds, request.interval_seconds, request.maximum_frames)
        else:
            timestamps = manual_timestamps(request.manual_timestamp_text, info.duration_seconds, request.maximum_frames)
        return {
            "selected_video_frames": len(timestamps),
            "timestamps_seconds": timestamps,
            "perturbations_per_frame": request.settings.sample_count,
            "inference_calls_per_frame": request.settings.sample_count + 1,
            "estimated_total_inference_calls": total_inference_calls(len(timestamps), request.settings.sample_count),
            "video_duration_seconds": info.duration_seconds,
        }

    def _analyze(
        self,
        image: object,
        detector: object,
        settings: ExperimentSettings,
        token: CancellationToken,
        progress: ProgressCallback | None,
        *,
        progress_start: float,
        progress_span: float,
        prefix: str = "",
    ) -> ImageAnalysis:
        method = method_for(settings.method)

        def core_progress(stage: str, current: int, total: int) -> None:
            if stage == "clean_baseline":
                local = 0.05
                description = "Clean baseline"
            elif stage.startswith("perturbation:"):
                local = 0.05 + 0.75 * current / max(total, 1)
                description = f"Sample {current}/{total}"
            elif stage == "matching":
                local = 0.86
                description = "Matching detections"
            else:
                local = 0.94
                description = "Calculating metrics"
            if prefix:
                description = f"{prefix}: {description}"
            self._progress(progress, progress_start + progress_span * local, description)

        return method.analyze(
            image,  # type: ignore[arg-type]
            detector,  # type: ignore[arg-type]
            settings,
            progress=core_progress,
            cancelled=lambda: token.is_cancelled,
        )

    @staticmethod
    def _summary(
        model_path: Path,
        input_path: Path,
        settings: ExperimentSettings,
        analysis: ImageAnalysis,
    ) -> dict[str, object]:
        return build_summary(
            model_path=model_path,
            image_path=input_path,
            perturbation_count=settings.sample_count,
            seed=settings.seed,
            image_size=settings.image_size,
            confidence=settings.confidence,
            nms_iou=settings.nms_iou,
            match_iou=settings.match_iou,
            device=settings.device.value,
            perturbation_config=PerturbationConfig(),
            sample_metadata=analysis.sample_metadata,
            metrics=analysis.metrics,
        )

    def _metadata(
        self,
        input_kind: InputKind,
        input_path: Path,
        model_path: Path,
        settings: ExperimentSettings,
        method: ExperimentMethod,
        *,
        artifacts: dict[str, object],
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "dashboard_schema_version": DASHBOARD_SCHEMA_VERSION,
            "core_schema_version": "1.0",
            "created_at_utc": self.clock().isoformat(),
            "method_name": method.identity.display_name,
            "method_id": method.identity.identifier,
            "method_version": method.identity.version,
            "input_kind": input_kind.value,
            "input_name": input_path.name,
            "input_path": str(input_path.resolve()),
            "model_name": model_path.name,
            "model_path": str(model_path.resolve()),
            "configuration": settings.to_dict(),
            "artifacts": artifacts,
            **(extra or {}),
        }

    @staticmethod
    def _mean(targets: list[dict[str, object]], key: str) -> float:
        return fmean(float(target[key]) for target in targets) if targets else 0.0

    @staticmethod
    def _write_video_csv(path: Path, records: list[dict[str, object]]) -> None:
        fields = (
            "selection_index",
            "frame_index",
            "timestamp_seconds",
            "target_count",
            "dominant_classes",
            "mean_persistence",
            "mean_confidence",
            "mean_class_agreement",
            "mean_entropy",
            "mean_iou",
            "instability_events",
            "directory",
        )
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = dict(record)
                row["dominant_classes"] = "; ".join(record["dominant_classes"])
                row["instability_events"] = " | ".join(record["instability_events"])
                writer.writerow(row)

    @staticmethod
    def _progress(progress: ProgressCallback | None, fraction: float, description: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), description)
