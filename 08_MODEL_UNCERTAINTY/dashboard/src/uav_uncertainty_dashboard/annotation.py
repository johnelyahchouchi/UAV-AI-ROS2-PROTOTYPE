"""OpenCV annotations for clean and perturbed samples."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np

from uav_uncertainty.analysis_engine import ImageAnalysis


COLORS = (
    (255, 220, 0),
    (0, 200, 255),
    (255, 80, 180),
    (80, 220, 80),
    (180, 120, 255),
    (255, 160, 40),
)


def _color(cluster_id: int, selected: bool) -> tuple[int, int, int]:
    if selected:
        return (0, 255, 255)
    return COLORS[(cluster_id - 1) % len(COLORS)]


def annotate_sample(
    analysis: ImageAnalysis,
    sample_index: int,
    *,
    selected_target_id: str | None = None,
) -> np.ndarray:
    """Draw matched target IDs, observed classes, and confidence on one sample."""
    if not 0 <= sample_index < len(analysis.samples):
        raise ValueError(f"Sample index outside analysis range: {sample_index}")
    output = analysis.samples[sample_index].image.copy()
    metric_by_id = {metric.target_id: metric for metric in analysis.metrics}
    for cluster in analysis.clusters:
        detection = cluster.observations.get(sample_index)
        if detection is None:
            continue
        target_id = f"target_{cluster.cluster_id}"
        selected = selected_target_id == target_id
        color = _color(cluster.cluster_id, selected)
        thickness = 4 if selected else 2
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        dominant = metric_by_id[target_id].dominant_class
        label = f"{target_id} | {dominant} | {detection.confidence:.3f}"
        text_y = max(18, y1 - 7)
        cv2.putText(
            output,
            label,
            (max(0, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def annotate_detection_records(
    image: np.ndarray,
    detections: list[dict[str, object]],
    *,
    selected_target_id: str | None = None,
) -> np.ndarray:
    """Draw persisted sample records, optionally emphasizing one target."""
    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        target_id = str(detection["target_id"])
        try:
            cluster_id = int(target_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            cluster_id = index
        selected = selected_target_id == target_id
        color = _color(cluster_id, selected)
        thickness = 4 if selected else 2
        x1, y1, x2, y2 = (
            int(round(float(detection[key]))) for key in ("x1", "y1", "x2", "y2")
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label = (
            f"{target_id} | {detection['class_name']} | "
            f"{float(detection['confidence']):.3f}"
        )
        cv2.putText(
            output,
            label,
            (max(0, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def write_annotated_image(path: Path, image: np.ndarray) -> None:
    """Encode and atomically publish one dashboard-owned JPEG preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"OpenCV could not encode annotated preview: {path.name}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(encoded.tobytes())
            output.flush()
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
