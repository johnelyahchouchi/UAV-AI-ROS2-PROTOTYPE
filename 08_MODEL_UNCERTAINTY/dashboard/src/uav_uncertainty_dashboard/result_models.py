"""Method-aware data models for persisted dashboard experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DASHBOARD_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RunPaths:
    """Owned staging and publication paths for one experiment."""

    job_id: str
    staging_dir: Path
    final_dir: Path


@dataclass(frozen=True)
class PublishedRun:
    """Paths exposed after atomic publication succeeds."""

    directory: Path
    metadata: Path
    summary: Path | None
    targets_csv: Path | None
    sample_metadata: Path | None
    baseline_preview: Path | None
    video_frames_csv: Path | None = None


@dataclass(frozen=True)
class LoadedExperiment:
    """A completed run reopened without inference."""

    directory: Path
    metadata: dict[str, object]
    summary: dict[str, object]
    samples: list[dict[str, object]]
    video_summary: dict[str, object] | None = None

    @property
    def display_name(self) -> str:
        """Return a compact saved-run label."""
        method = str(self.metadata.get("method_name", "Unknown method"))
        input_name = str(self.metadata.get("input_name", self.directory.name))
        return f"{self.directory.name} — {method} — {input_name}"
