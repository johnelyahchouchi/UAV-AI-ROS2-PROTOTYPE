"""Owned staging, deterministic artifacts, and atomic run publication."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

from uav_uncertainty.analysis_engine import ImageAnalysis
from uav_uncertainty.mc_stability_runner import _write_csv, _write_json

from .annotation import annotate_sample, write_annotated_image
from .errors import DashboardError
from .result_models import PublishedRun, RunPaths


def safe_stem(value: str) -> str:
    """Return a short Windows-safe path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "experiment")[:80]


def enriched_sample_metadata(analysis: ImageAnalysis) -> list[dict[str, object]]:
    """Add matched target identities and raw observations to sample metadata."""
    all_target_ids = [metric.target_id for metric in analysis.metrics]
    rows: list[dict[str, object]] = []
    for sample in analysis.samples:
        observations: list[dict[str, object]] = []
        present: list[str] = []
        for cluster in analysis.clusters:
            detection = cluster.observations.get(sample.sample_index)
            if detection is None:
                continue
            target_id = f"target_{cluster.cluster_id}"
            present.append(target_id)
            record = detection.to_dict()
            record["target_id"] = target_id
            observations.append(record)
        present.sort()
        rows.append(
            {
                **sample.metadata(),
                "target_ids_present": present,
                "target_ids_missing": [item for item in all_target_ids if item not in present],
                "detections": observations,
            }
        )
    return rows


class OutputManager:
    """Manage only dashboard-owned files beneath one ignored outputs root."""

    def __init__(
        self,
        output_root: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.output_root = output_root.resolve(strict=False)
        self.staging_root = self.output_root / ".staging"
        self._clock = clock
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def prepare_run(self, input_path: Path, method_id: str, job_id: str) -> RunPaths:
        """Reserve a unique staging and final directory without overwriting runs."""
        timestamp = self._clock().strftime("%Y%m%dT%H%M%S_%fZ")
        short_id = safe_stem(job_id)[:12]
        name = f"{safe_stem(input_path.stem)}_{safe_stem(method_id)}_{timestamp}_{short_id}"
        staging_dir = self.staging_root / short_id
        final_dir = self.output_root / name
        if staging_dir.exists() or final_dir.exists():
            raise DashboardError(
                "OUTPUT_COLLISION",
                "Refusing to overwrite an existing dashboard staging or result directory.",
            )
        staging_dir.mkdir(parents=True, exist_ok=False)
        return RunPaths(job_id, staging_dir, final_dir)

    def write_image_result(
        self,
        directory: Path,
        summary: dict[str, object],
        metadata: dict[str, object],
        analysis: ImageAnalysis,
    ) -> None:
        """Write a complete image/frame result within owned staging."""
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "summary.json", summary)
        _write_csv(directory / "targets.csv", analysis.metrics)
        self.write_json(directory / "dashboard_metadata.json", metadata)
        samples = enriched_sample_metadata(analysis)
        self.write_json(directory / "sample_metadata.json", samples)
        previews = directory / "previews"
        sample_images = directory / "samples"
        for sample in analysis.samples:
            write_annotated_image(
                sample_images / f"sample_{sample.sample_index:03d}.jpg",
                sample.image,
            )
            write_annotated_image(
                previews / f"sample_{sample.sample_index:03d}.jpg",
                annotate_sample(analysis, sample.sample_index),
            )

    @staticmethod
    def write_json(path: Path, payload: object) -> None:
        """Atomically write JSON in the selected directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(payload, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def publish(self, paths: RunPaths) -> PublishedRun:
        """Atomically publish a valid staging directory."""
        metadata_path = paths.staging_dir / "dashboard_metadata.json"
        if not metadata_path.is_file():
            raise DashboardError("OUTPUT_INCOMPLETE", "Dashboard metadata was not generated.")
        try:
            os.replace(paths.staging_dir, paths.final_dir)
        except OSError as error:
            raise DashboardError("OUTPUT_PUBLISH_FAILED", "Completed run could not be published.", detail=str(error)) from error
        return self.published_paths(paths.final_dir)

    @staticmethod
    def published_paths(directory: Path) -> PublishedRun:
        """Resolve known artifacts for a completed run."""
        def existing(name: str) -> Path | None:
            path = directory / name
            return path if path.is_file() else None

        baseline = directory / "previews" / "sample_000.jpg"
        return PublishedRun(
            directory=directory,
            metadata=directory / "dashboard_metadata.json",
            summary=existing("summary.json"),
            targets_csv=existing("targets.csv"),
            sample_metadata=existing("sample_metadata.json"),
            baseline_preview=baseline if baseline.is_file() else None,
            video_frames_csv=existing("video_frames.csv"),
        )

    def abort(self, paths: RunPaths) -> None:
        """Remove only the owned, unpublished staging directory."""
        self._remove_owned_staging(paths.staging_dir)

    def _remove_owned_staging(self, target: Path) -> None:
        resolved = target.resolve(strict=False)
        staging = self.staging_root.resolve(strict=False)
        try:
            resolved.relative_to(staging)
        except ValueError as error:
            raise DashboardError("UNSAFE_CLEANUP_REFUSED", f"Refusing cleanup outside {staging}") from error
        if resolved == staging:
            raise DashboardError("UNSAFE_CLEANUP_REFUSED", "Refusing to remove the staging root.")
        if resolved.exists():
            shutil.rmtree(resolved)
