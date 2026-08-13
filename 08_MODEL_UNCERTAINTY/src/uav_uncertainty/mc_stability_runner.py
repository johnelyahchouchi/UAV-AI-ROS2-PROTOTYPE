"""CLI orchestration for V1 Monte Carlo input-perturbation analysis."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterator, Sequence, TextIO

import cv2

from .analysis_engine import analyze_image
from .detector_adapter import UltralyticsDetector
from .perturbations import PerturbationConfig
from .uncertainty_metrics import TargetMetrics


SCHEMA_VERSION = "1.0"
RESULT_FILENAMES = ("summary.json", "targets.csv")


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "image"


def _device_argument(value: str) -> int | str | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized.isdigit():
        return int(normalized)
    return value.strip()


def _warn_if_repository_local_inputs(model_path: Path, image_path: Path) -> None:
    repository_root = _repository_root()
    local_inputs = [
        label
        for label, path in (("model", model_path), ("image", image_path))
        if path.resolve().is_relative_to(repository_root)
    ]
    if local_inputs:
        labels = " and ".join(local_inputs)
        print(
            f"WARNING: Input path(s) for {labels} resolve inside the Git repository. "
            "Do not commit model weights, datasets, or runtime imagery.",
            file=sys.stderr,
        )


def build_summary(
    *,
    model_path: Path,
    image_path: Path,
    perturbation_count: int,
    seed: int,
    image_size: int,
    confidence: float,
    nms_iou: float,
    match_iou: float,
    device: str,
    perturbation_config: PerturbationConfig,
    sample_metadata: list[dict[str, object]],
    metrics: list[TargetMetrics],
) -> dict[str, object]:
    """Build the deterministic machine-readable V1 summary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "monte_carlo_input_perturbation_v1",
        "method_scope": (
            "Robustness analysis under controlled input variations; not Monte Carlo "
            "Dropout and not a calibrated probability of correctness."
        ),
        "input": {
            "model": str(model_path.resolve()),
            "image": str(image_path.resolve()),
        },
        "sampling": {
            "clean_baseline_count": 1,
            "perturbation_count": perturbation_count,
            "total_inference_sample_count": perturbation_count + 1,
            "seed": seed,
            "perturbation_config": perturbation_config.to_dict(),
            "samples": sample_metadata,
        },
        "detector_configuration": {
            "image_size": image_size,
            "confidence_threshold": confidence,
            "nms_iou_threshold": nms_iou,
            "device": device,
        },
        "matching": {
            "strategy": "deterministic_greedy_highest_iou",
            "class_agnostic": True,
            "iou_threshold": match_iou,
            "one_detection_per_cluster_per_sample": True,
            "running_reference": "mean_of_previous_cluster_boxes",
        },
        "metric_definitions": {
            "detection_persistence": "detection_count / total_inference_sample_count",
            "confidence_std": "population standard deviation",
            "class_agreement": "dominant class count / detected observations",
            "class_entropy_bits": "Shannon entropy using log base 2",
            "bbox_variation": "population standard deviation in source-image pixels",
            "mean_iou_to_reference": (
                "mean IoU to clean baseline box when available, otherwise mean box"
            ),
        },
        "target_count": len(metrics),
        "targets": [target.to_dict() for target in metrics],
    }


@contextmanager
def _atomic_text_output(path: Path, *, newline: str) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and atomically replace the target."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_json(path: Path, summary: dict[str, object]) -> None:
    with _atomic_text_output(path, newline="\n") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")


def _write_csv(path: Path, metrics: list[TargetMetrics]) -> None:
    fields = (
        "target_id",
        "sample_count",
        "detection_count",
        "detection_persistence",
        "confidence_mean",
        "confidence_std",
        "dominant_class",
        "class_agreement",
        "class_entropy_bits",
        "class_histogram_json",
        "class_distribution_json",
        "bbox_center_std_x_pixels",
        "bbox_center_std_y_pixels",
        "bbox_width_std_pixels",
        "bbox_height_std_pixels",
        "reference_box_source",
        "reference_x1",
        "reference_y1",
        "reference_x2",
        "reference_y2",
        "mean_iou_to_reference",
        "detected_sample_indices_json",
        "missing_sample_indices_json",
    )
    with _atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for target in metrics:
            x1, y1, x2, y2 = target.reference_bbox_xyxy
            writer.writerow(
                {
                    "target_id": target.target_id,
                    "sample_count": target.sample_count,
                    "detection_count": target.detection_count,
                    "detection_persistence": target.detection_persistence,
                    "confidence_mean": target.confidence_mean,
                    "confidence_std": target.confidence_std,
                    "dominant_class": target.dominant_class,
                    "class_agreement": target.class_agreement,
                    "class_entropy_bits": target.class_entropy_bits,
                    "class_histogram_json": json.dumps(
                        target.class_histogram, sort_keys=True
                    ),
                    "class_distribution_json": json.dumps(
                        target.class_distribution, sort_keys=True
                    ),
                    "bbox_center_std_x_pixels": target.bbox_center_std_pixels.x,
                    "bbox_center_std_y_pixels": target.bbox_center_std_pixels.y,
                    "bbox_width_std_pixels": target.bbox_size_std_pixels.x,
                    "bbox_height_std_pixels": target.bbox_size_std_pixels.y,
                    "reference_box_source": target.reference_box_source,
                    "reference_x1": x1,
                    "reference_y1": y1,
                    "reference_x2": x2,
                    "reference_y2": y2,
                    "mean_iou_to_reference": target.mean_iou_to_reference,
                    "detected_sample_indices_json": json.dumps(
                        target.detected_sample_indices
                    ),
                    "missing_sample_indices_json": json.dumps(
                        target.missing_sample_indices
                    ),
                }
            )


def _prepare_result_paths(
    output_root: Path,
    run_name: str,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Create one run directory and protect existing subsystem-owned results."""
    run_directory = output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    json_path = run_directory / RESULT_FILENAMES[0]
    csv_path = run_directory / RESULT_FILENAMES[1]
    paths = (json_path, csv_path)
    existing = [path.name for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(existing)
        raise FileExistsError(
            f"Refusing to replace existing result file(s) in {run_directory}: "
            f"{names}. Use --overwrite to replace only these owned result files."
        )
    return json_path, csv_path


def _print_console(metrics: list[TargetMetrics], total_samples: int) -> None:
    print("\n=== UAV DETECTOR INPUT-PERTURBATION STABILITY ===")
    print(f"Inference samples: {total_samples} (1 clean + {total_samples - 1} perturbed)")
    if not metrics:
        print("No detections were observed in any sample.")
        return

    for target in metrics:
        print(f"\n{target.target_id.replace('_', ' ').title()}")
        print(f"Dominant class: {target.dominant_class}")
        print(f"Detected: {target.detection_count}/{target.sample_count}")
        print(f"Persistence: {target.detection_persistence:.3f}")
        print(f"Mean confidence: {target.confidence_mean:.3f}")
        print(f"Confidence std: {target.confidence_std:.3f}")
        print(f"Class agreement: {target.class_agreement:.3f}")
        print(f"Class entropy (bits): {target.class_entropy_bits:.3f}")
        print(
            "Center std (px): "
            f"x={target.bbox_center_std_pixels.x:.3f}, "
            f"y={target.bbox_center_std_pixels.y:.3f}"
        )
        print(
            "Size std (px): "
            f"w={target.bbox_size_std_pixels.x:.3f}, "
            f"h={target.bbox_size_std_pixels.y:.3f}"
        )
        print(f"Mean IoU to reference: {target.mean_iou_to_reference:.3f}")


def run_analysis(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run baseline and perturbed inference, then write JSON and CSV summaries."""
    model_path = Path(args.model).expanduser()
    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {image_path}")
    _warn_if_repository_local_inputs(model_path, image_path)

    output_root = Path(args.output_dir).expanduser()
    run_name = (
        f"{_safe_stem(image_path.stem)}_seed{args.seed}_n{args.samples}"
        if args.run_name is None
        else _safe_stem(args.run_name)
    )
    json_path, csv_path = _prepare_result_paths(
        output_root,
        run_name,
        overwrite=args.overwrite,
    )

    detector = UltralyticsDetector(
        model_path,
        image_size=args.imgsz,
        confidence=args.conf,
        nms_iou=args.iou,
        device=_device_argument(args.device),
    )

    config = PerturbationConfig()

    def console_progress(stage: str, current: int, total: int) -> None:
        if stage == "clean_baseline":
            print("Running clean baseline inference...")
        elif stage.startswith("perturbation:"):
            family = stage.partition(":")[2]
            print(f"Running sample {current}/{total}: {family}...")

    analysis = analyze_image(
        image,
        detector,
        sample_count=args.samples,
        seed=args.seed,
        match_iou=args.match_iou,
        perturbation_config=config,
        progress=console_progress,
    )
    total_samples = len(analysis.samples)
    metrics = analysis.metrics
    summary = build_summary(
        model_path=model_path,
        image_path=image_path,
        perturbation_count=args.samples,
        seed=args.seed,
        image_size=args.imgsz,
        confidence=args.conf,
        nms_iou=args.iou,
        match_iou=args.match_iou,
        device=args.device,
        perturbation_config=config,
        sample_metadata=analysis.sample_metadata,
        metrics=metrics,
    )

    _write_json(json_path, summary)
    _write_csv(csv_path, metrics)
    _print_console(metrics, total_samples)
    print(f"\nJSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure detector stability under seeded mild image perturbations. "
            "This V1 does not implement Monte Carlo Dropout."
        )
    )
    parser.add_argument("--model", required=True, help="Existing Ultralytics .pt model")
    parser.add_argument("--image", required=True, help="Source image path")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help=(
            "Number of perturbed samples. Total inference count is N + 1 because "
            "one clean baseline prediction is always included."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="Detector NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device such as auto, 0, or cpu",
    )
    parser.add_argument("--output-dir", default=str(_default_output_root()))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only summary.json and targets.csv in an existing run directory.",
    )
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples < 1:
        parser.error("--samples must be at least 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")
    for name in ("conf", "iou", "match_iou"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0 and at most 1.")
    if not args.device.strip():
        parser.error("--device must not be empty.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    try:
        run_analysis(args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
