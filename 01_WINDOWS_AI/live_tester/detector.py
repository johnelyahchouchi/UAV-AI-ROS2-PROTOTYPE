"""Trusted Ultralytics detector adapter for the live tester."""

from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any, Callable

from uav_security.model_integrity import (
    ModelIntegrityError,
    load_trusted_yolo,
    verify_trusted_model,
)

from .domain import (
    DetectionResult,
    TesterError,
    class_name_for,
    clip_bbox,
    coordinates,
    scalar,
)


class YoloDetector:
    """Verify a checkpoint before loading it and expose simple detections."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        registry_path: Path | None = None,
        verifier: Callable[..., str] = verify_trusted_model,
        model_loader: Callable[..., Any] = load_trusted_yolo,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device
        self.model_sha256 = verifier(model_path, registry_path)
        try:
            self.model = model_loader(model_path, registry_path=registry_path)
        except ModelIntegrityError:
            raise
        except Exception as error:
            raise TesterError(f"Trusted YOLO model could not be loaded: {error}") from error
        self._prediction_lock = threading.Lock()

    def detect(self, frame: Any) -> list[DetectionResult]:
        """Run one ordinary deterministic inference on a BGR frame."""

        try:
            # V1 can sample the same trusted detector from a background worker.
            # Ultralytics model instances are not a thread-safety boundary, so
            # serialize individual calls while still letting the UI stay live.
            with self._prediction_lock:
                raw_results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False,
                )
        except Exception as error:
            raise TesterError(f"YOLO inference failed: {error}") from error
        if not raw_results:
            return []
        result = raw_results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            raise TesterError("Detector received a frame without valid dimensions")
        frame_height, frame_width = int(shape[0]), int(shape[1])
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        detections: list[DetectionResult] = []
        for box in boxes:
            try:
                bbox = clip_bbox(coordinates(box.xyxy), frame_width, frame_height)
                confidence = scalar(box.conf)
                class_id = int(scalar(box.cls))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if bbox is None or not math.isfinite(confidence):
                continue
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=class_name_for(names, class_id),
                    confidence=max(0.0, min(1.0, confidence)),
                    bbox=bbox,
                )
            )
        return detections
