"""OpenCV detection and low-noise HUD rendering."""

from __future__ import annotations

from typing import Any, Sequence

from .domain import (
    CaptureRegion,
    DetectionResult,
    FrameMetrics,
    clip_bbox,
    safe_overlay_text,
)


class OverlayRenderer:
    """Draw clean detection boxes, labels, and a compact performance HUD."""

    _PALETTE = (
        (0, 220, 255),
        (255, 165, 0),
        (80, 220, 100),
        (255, 120, 210),
        (100, 190, 255),
        (220, 180, 70),
    )

    def __init__(self, cv2_module: Any | None = None) -> None:
        if cv2_module is None:
            import cv2

            cv2_module = cv2
        self.cv2 = cv2_module

    def draw_detections(
        self, frame: Any, detections: Sequence[DetectionResult]
    ) -> Any:
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        for detection in detections:
            clipped = clip_bbox(detection.bbox, frame_width, frame_height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            color = self._PALETTE[detection.class_id % len(self._PALETTE)]
            self.cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = safe_overlay_text(
                f"{detection.class_name} {detection.confidence:.2f}"
            )
            (text_width, text_height), baseline = self.cv2.getTextSize(
                label, self.cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            label_x = min(max(0, x1), max(0, frame_width - text_width - 8))
            label_bottom = y1 - 5 if y1 >= text_height + baseline + 8 else min(
                frame_height - baseline - 1, y1 + text_height + baseline + 8
            )
            label_top = max(0, label_bottom - text_height - baseline - 5)
            self.cv2.rectangle(
                annotated,
                (label_x, label_top),
                (min(frame_width - 1, label_x + text_width + 7), label_bottom + 1),
                color,
                -1,
            )
            self.cv2.putText(
                annotated,
                label,
                (label_x + 3, max(text_height, label_bottom - baseline - 2)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (15, 15, 15),
                1,
                self.cv2.LINE_AA,
            )
        return annotated

    def draw_hud(
        self,
        frame: Any,
        *,
        metrics: FrameMetrics,
        model_name: str,
        region: CaptureRegion,
        device: str,
        detection_count: int,
        source_label: str | None = None,
        paused: bool = False,
        last_inspection: str | None = None,
        uncertainty_available: bool = True,
        mcdo_v2_available: bool = False,
        continuous_uncertainty: bool = False,
    ) -> Any:
        source_text = source_label or (
            f"region {region.left},{region.top} {region.width}x{region.height}"
        )
        lines = [
            "LIVE UAV AI MODEL TESTER",
            f"FPS {metrics.fps:5.1f} | capture {metrics.capture_ms:5.1f} ms | "
            f"infer {metrics.inference_ms:5.1f} ms | frame {metrics.total_ms:5.1f} ms",
            f"model {safe_overlay_text(model_name, 36)} | device {device} | "
            f"detections {detection_count}",
            safe_overlay_text(source_text, 88),
            (
                "Q/ESC quit | P pause | S screenshot | H HUD | uncertainty panels live | U detailed report"
                if continuous_uncertainty and uncertainty_available
                else
                "Q/ESC quit | P pause | S screenshot | H HUD | U uncertainty menu (1 V1 / 2 V2)"
                if uncertainty_available and mcdo_v2_available
                else "Q/ESC quit | P pause | S screenshot | H HUD | U inspect V1 robustness"
                if uncertainty_available
                else "Q/ESC quit | P pause | S screenshot | H HUD"
            ),
        ]
        if last_inspection:
            lines.append(f"Last inspection: {safe_overlay_text(last_inspection, 52)}")
        height = 20 + len(lines) * 23
        width = min(frame.shape[1] - 1, 890)
        if width > 0:
            overlay = frame.copy()
            self.cv2.rectangle(overlay, (0, 0), (width, height), (12, 12, 12), -1)
            self.cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        for index, line in enumerate(lines):
            self.cv2.putText(
                frame,
                line,
                (12, 24 + index * 23),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (235, 235, 235),
                1,
                self.cv2.LINE_AA,
            )
        if paused:
            self.cv2.putText(
                frame,
                "PAUSED",
                (max(10, frame.shape[1] // 2 - 85), max(45, frame.shape[0] // 2)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 220, 255),
                3,
                self.cv2.LINE_AA,
            )
        return frame
