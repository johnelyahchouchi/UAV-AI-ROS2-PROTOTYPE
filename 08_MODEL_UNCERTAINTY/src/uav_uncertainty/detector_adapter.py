"""Adapter between Ultralytics results and detector-independent domain types."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .detection_types import Detection


def class_name_for(
    names: Mapping[int, Any] | Sequence[Any],
    class_id: int,
) -> str:
    """Resolve an Ultralytics class name from dict or sequence metadata."""
    try:
        if isinstance(names, Mapping):
            return str(names.get(class_id, f"class_{class_id}"))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return f"class_{class_id}"


def detections_from_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into internal detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = getattr(result, "names", {})
    detections: list[Detection] = []

    for box in boxes:
        coordinates = [float(value) for value in box.xyxy[0].tolist()]
        if len(coordinates) != 4:
            continue
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name_for(names, class_id),
                confidence=float(box.conf[0].item()),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


class UltralyticsDetector:
    """Load one existing Ultralytics checkpoint and expose internal detections."""

    def __init__(
        self,
        model_path: Path,
        *,
        image_size: int,
        confidence: float,
        nms_iou: float,
        device: int | str | None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise ValueError("The model must be an existing Ultralytics .pt file.")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required only for live inference. Run this command "
                "inside the existing UAV YOLO environment."
            ) from error

        try:
            self._model = YOLO(str(model_path.resolve()))
        except Exception as error:
            raise RuntimeError(
                f"Ultralytics could not load the detection model: {model_path}"
            ) from error
        task = getattr(self._model, "task", None)
        if task not in (None, "detect"):
            raise ValueError(f"Model task must be 'detect', not {task!r}.")
        self._arguments: dict[str, object] = {
            "imgsz": image_size,
            "conf": confidence,
            "iou": nms_iou,
            "verbose": False,
            "save": False,
        }
        if device is not None:
            self._arguments["device"] = device

    def detect(self, image: Any) -> list[Detection]:
        """Run ordinary deterministic detector inference on one image array."""
        try:
            results = self._model.predict(source=image, **self._arguments)
        except Exception as error:
            raise RuntimeError(f"Ultralytics inference failed: {error}") from error
        if not results:
            return []
        return detections_from_result(results[0])
