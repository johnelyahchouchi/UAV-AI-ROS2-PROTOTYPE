"""Saved-run loading and UI-ready tables."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from .errors import DashboardError
from .result_models import DASHBOARD_SCHEMA_VERSION, LoadedExperiment


TARGET_HEADERS = (
    "Target ID",
    "Dominant class",
    "Detected / total",
    "Persistence",
    "Mean confidence",
    "Confidence std",
    "Class agreement",
    "Entropy (bits)",
    "Mean IoU",
    "Center std x / y (px)",
    "Size std w / h (px)",
    "Reference source",
)
SAMPLE_HEADERS = (
    "Sample",
    "Family",
    "Parameters",
    "Detections",
    "Targets present",
    "Targets missing",
)
FAMILY_HEADERS = (
    "Family",
    "Samples used",
    "Mean target presence",
    "Total detections",
    "Total target misses",
    "Mean confidence delta vs clean",
    "Observed statement",
)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardError("RESULT_READ_FAILED", f"Could not read saved result: {path}", detail=str(error)) from error


def load_run(directory: Path) -> LoadedExperiment:
    """Load one completed image or video run without rerunning inference."""
    root = directory.resolve(strict=True)
    metadata_path = root / "dashboard_metadata.json"
    if not metadata_path.is_file():
        raise DashboardError("RUN_METADATA_MISSING", f"No dashboard metadata in {root}")
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise DashboardError("RUN_METADATA_INVALID", "Dashboard metadata must be a JSON object.")
    input_kind = str(metadata.get("input_kind", "Image"))
    video_summary: dict[str, object] | None = None
    if input_kind == "Video":
        video_value = _read_json(root / "video_summary.json")
        if not isinstance(video_value, dict):
            raise DashboardError("VIDEO_SUMMARY_INVALID", "Video summary must be an object.")
        video_summary = video_value
        frame_entries = video_summary.get("frames", [])
        if not frame_entries:
            summary: dict[str, object] = {"targets": [], "sampling": {"samples": []}}
            samples: list[dict[str, object]] = []
        else:
            first_dir = root / str(frame_entries[0]["directory"])
            summary_value = _read_json(first_dir / "summary.json")
            samples_value = _read_json(first_dir / "sample_metadata.json")
            summary = dict(summary_value)  # type: ignore[arg-type]
            samples = list(samples_value)  # type: ignore[arg-type]
    else:
        summary_value = _read_json(root / "summary.json")
        samples_value = _read_json(root / "sample_metadata.json")
        summary = dict(summary_value)  # type: ignore[arg-type]
        samples = list(samples_value)  # type: ignore[arg-type]
    return LoadedExperiment(root, metadata, summary, samples, video_summary)


def list_saved_runs(output_root: Path) -> list[LoadedExperiment]:
    """Return valid saved runs in newest-directory-name order."""
    if not output_root.exists():
        return []
    runs: list[LoadedExperiment] = []
    for path in sorted(output_root.iterdir(), reverse=True):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            runs.append(load_run(path))
        except DashboardError:
            continue
    return runs


def target_rows(summary: dict[str, object]) -> list[list[object]]:
    """Convert raw target dictionaries to a sortable display table."""
    rows: list[list[object]] = []
    for target in summary.get("targets", []):
        center = target["bbox_center_std_pixels"]
        size = target["bbox_size_std_pixels"]
        rows.append(
            [
                target["target_id"],
                target["dominant_class"],
                f"{target['detection_count']} / {target['sample_count']}",
                target["detection_persistence"],
                target["confidence_mean"],
                target["confidence_std"],
                target["class_agreement"],
                target["class_entropy_bits"],
                target["mean_iou_to_reference"],
                f"{float(center['x']):.3f} / {float(center['y']):.3f}",
                f"{float(size['x']):.3f} / {float(size['y']):.3f}",
                target["reference_box_source"],
            ]
        )
    return rows


def target_detail(summary: dict[str, object], target_id: str) -> dict[str, object]:
    """Return all uncombined metrics for one selected target."""
    for target in summary.get("targets", []):
        if target.get("target_id") == target_id:
            return dict(target)
    return {}


def sample_rows(samples: list[dict[str, object]]) -> list[list[object]]:
    """Return clean/perturbation observations without causal wording."""
    return [
        [
            sample["sample_index"],
            sample["family"],
            json.dumps(sample.get("parameters", {}), sort_keys=True),
            sample["detection_count"],
            ", ".join(sample.get("target_ids_present", [])),
            ", ".join(sample.get("target_ids_missing", [])),
        ]
        for sample in samples
    ]


def family_rows(samples: list[dict[str, object]]) -> list[list[object]]:
    """Aggregate factual observations by perturbation family."""
    clean = next((item for item in samples if item.get("sample_index") == 0), None)
    clean_confidence = {
        str(item["target_id"]): float(item["confidence"])
        for item in (clean or {}).get("detections", [])
    }
    families = sorted({str(item["family"]) for item in samples if item["family"] != "clean_baseline"})
    all_targets = sorted({str(target) for item in samples for target in item.get("target_ids_present", [])})
    rows: list[list[object]] = []
    for family in families:
        items = [item for item in samples if item["family"] == family]
        presence_total = sum(len(item.get("target_ids_present", [])) for item in items)
        denominator = len(items) * len(all_targets)
        deltas: list[float] = []
        appearances: dict[str, int] = {target: 0 for target in all_targets}
        for item in items:
            for target in item.get("target_ids_present", []):
                appearances[str(target)] += 1
            for detection in item.get("detections", []):
                target_id = str(detection["target_id"])
                if target_id in clean_confidence:
                    deltas.append(float(detection["confidence"]) - clean_confidence[target_id])
        factual = "; ".join(
            f"{target} appeared in {count}/{len(items)} {family} samples"
            for target, count in appearances.items()
            if count
        ) or f"No target appeared in the {len(items)} {family} samples"
        rows.append(
            [
                family,
                len(items),
                presence_total / denominator if denominator else 0.0,
                sum(int(item["detection_count"]) for item in items),
                sum(len(item.get("target_ids_missing", [])) for item in items),
                fmean(deltas) if deltas else None,
                factual,
            ]
        )
    return rows


def overview_markdown(run: LoadedExperiment) -> str:
    """Return a compact, non-calibrated overview."""
    settings = run.metadata.get("configuration", {})
    target_count = len(run.summary.get("targets", []))
    return (
        f"**Input:** {run.metadata.get('input_name', '')}  \n"
        f"**Model:** {run.metadata.get('model_name', '')}  \n"
        f"**Method:** {run.metadata.get('method_name', '')} "
        f"(v{run.metadata.get('method_version', '')})  \n"
        f"**Seed / perturbations / total samples:** {settings.get('seed')} / "
        f"{settings.get('sample_count')} / {settings.get('total_inference_samples')}  \n"
        f"**Target clusters:** {target_count}  \n"
        f"**Detector:** imgsz={settings.get('image_size')}, conf={settings.get('confidence')}, "
        f"NMS IoU={settings.get('nms_iou')}, match IoU={settings.get('match_iou')}, "
        f"device={settings.get('device')}  \n\n"
        "Higher persistence, class agreement, and mean IoU indicate more stable observations. "
        "Lower confidence standard deviation and class entropy indicate less variation.  \n\n"
        "This dashboard measures prediction stability under controlled image perturbations. "
        "It does not measure true detection accuracy without labeled ground truth."
    )


def schema_note(run: LoadedExperiment) -> str:
    """Describe schema/method identity without rejecting future methods."""
    schema = str(run.metadata.get("dashboard_schema_version", "unknown"))
    method = str(run.metadata.get("method_name", "unknown"))
    current = "current" if schema == DASHBOARD_SCHEMA_VERSION else "different/legacy"
    return f"Dashboard schema {schema} ({current}); method: {method}."
