"""Owned staging, H.264 conversion, validation, and safe cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable

import cv2
import imageio_ffmpeg

from .errors import DashboardError, ProcessingCancelled
from .source_adapter import VideoMetadata


@dataclass(frozen=True)
class JobPaths:
    """Owned paths for one processing job."""

    job_id: str
    staging_dir: Path
    final_dir: Path
    owned_input: Path
    intermediate_video: Path
    encoded_video: Path
    csv_report: Path


@dataclass(frozen=True)
class PublishedOutputs:
    """Final paths returned to the UI."""

    directory: Path
    annotated_video: Path
    csv_report: Path


def safe_stem(value: str) -> str:
    """Return a short Windows-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "video")[:80]


class OutputManager:
    """Manage only files rooted beneath the dashboard outputs directory."""

    def __init__(
        self,
        output_root: Path,
        *,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.output_root = output_root.resolve(strict=False)
        self.staging_root = self.output_root / ".staging"
        self._clock = clock
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def prepare_job(
        self,
        source_path: Path,
        job_id: str,
        *,
        copy_input: bool = True,
    ) -> JobPaths:
        """Create owned staging paths and optionally copy an uploaded input."""
        source = source_path.resolve(strict=True)
        source_stem = safe_stem(source.stem)
        timestamp = self._clock().strftime("%Y%m%d_%H%M%S")
        short_id = safe_stem(job_id)[:12]
        staging_dir = self.staging_root / short_id
        final_dir = self.output_root / f"{source_stem}_{timestamp}_{short_id}"

        if staging_dir.exists():
            self.cleanup_staging(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=False)

        owned_input = staging_dir / f"input{source.suffix.lower() or '.video'}"
        try:
            if copy_input:
                shutil.copy2(source, owned_input)
            else:
                owned_input = source
        except Exception as error:
            self.cleanup_staging(staging_dir)
            raise DashboardError(
                "UPLOAD_COPY_FAILED",
                "The uploaded video could not be copied into safe staging.",
                recovery="Check available disk space and file permissions.",
                detail=str(error),
            ) from error

        return JobPaths(
            job_id=job_id,
            staging_dir=staging_dir,
            final_dir=final_dir,
            owned_input=owned_input,
            intermediate_video=staging_dir / "annotated_intermediate.mp4",
            encoded_video=staging_dir / "annotated.mp4",
            csv_report=staging_dir / "detections.csv",
        )

    def transcode_h264(
        self,
        source: Path,
        destination: Path,
        *,
        cancelled: Callable[[], bool],
    ) -> None:
        """Convert an OpenCV MP4 to browser-compatible H.264 with bundled FFmpeg."""
        try:
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as error:
            raise DashboardError(
                "FFMPEG_UNAVAILABLE",
                "The bundled imageio-ffmpeg executable could not be resolved.",
                recovery="Reinstall imageio-ffmpeg in UAV_YOLO_ENV.",
                detail=str(error),
            ) from error

        command = [
            executable,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creation_flags,
            )
        except Exception as error:
            raise DashboardError(
                "H264_START_FAILED",
                "The H.264 encoder could not be started.",
                detail=str(error),
            ) from error

        while process.poll() is None:
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise ProcessingCancelled()
            time.sleep(0.05)

        stderr = process.stderr.read().strip() if process.stderr else ""
        if process.returncode != 0 or not destination.is_file():
            raise DashboardError(
                "H264_CONVERSION_FAILED",
                "The annotated video could not be converted to H.264.",
                recovery="Check free disk space and retry.",
                detail=stderr or f"FFmpeg exit code {process.returncode}",
            )

    @staticmethod
    def validate_final_video(path: Path, expected: VideoMetadata) -> None:
        """Confirm the final MP4 opens and preserves valid dimensions and FPS."""
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise DashboardError(
                    "OUTPUT_VIDEO_INVALID",
                    "The final annotated MP4 could not be opened.",
                )
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            success, frame = capture.read()
            if (
                not success
                or frame is None
                or width != expected.width
                or height != expected.height
                or abs(fps - expected.fps) > 0.05
            ):
                raise DashboardError(
                    "OUTPUT_VIDEO_INVALID",
                    "The final MP4 failed frame, resolution, or FPS validation.",
                    detail=(
                        f"expected={expected.width}x{expected.height}@{expected.fps}, "
                        f"actual={width}x{height}@{fps}"
                    ),
                )
        finally:
            capture.release()

    def publish(self, paths: JobPaths) -> PublishedOutputs:
        """Remove private staging files and atomically publish successful outputs."""
        for private_file in (paths.owned_input, paths.intermediate_video):
            try:
                if private_file.is_file() and self._is_within(private_file, paths.staging_dir):
                    private_file.unlink()
            except OSError as error:
                raise DashboardError(
                    "OUTPUT_FINALIZE_FAILED",
                    "A private staging file could not be removed.",
                    detail=str(error),
                ) from error

        try:
            os.replace(paths.staging_dir, paths.final_dir)
        except OSError as error:
            raise DashboardError(
                "OUTPUT_FINALIZE_FAILED",
                "Completed outputs could not be published atomically.",
                detail=str(error),
            ) from error

        return PublishedOutputs(
            directory=paths.final_dir,
            annotated_video=paths.final_dir / paths.encoded_video.name,
            csv_report=paths.final_dir / paths.csv_report.name,
        )

    def cleanup_staging(self, target: Path) -> None:
        """Delete only an owned job directory below .staging."""
        resolved_target = target.resolve(strict=False)
        resolved_staging = self.staging_root.resolve(strict=False)
        if resolved_target == resolved_staging or not self._is_within(
            resolved_target,
            resolved_staging,
        ):
            raise DashboardError(
                "UNSAFE_CLEANUP_REFUSED",
                f"Refusing to remove a path outside owned job staging: {target}",
            )
        if resolved_target.exists():
            shutil.rmtree(resolved_target)

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        except ValueError:
            return False
        return True
