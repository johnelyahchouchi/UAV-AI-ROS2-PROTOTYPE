"""Bridge the live detector to the on-demand V1 robustness inspector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from .domain import DetectionResult, PROJECT_ROOT, safe_overlay_text


UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
if str(UNCERTAINTY_SRC) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_SRC))

from uav_uncertainty.analysis import ImageAnalysis, analyze_image  # noqa: E402
from uav_uncertainty.detection import Detection  # noqa: E402
from uav_uncertainty.presentation import (  # noqa: E402
    interpret_target,
    overall_status,
    persistence_status,
)


@dataclass(frozen=True)
class InspectionView:
    """Rendered analysis plus its concise post-inspection status."""

    frame: Any
    analysis: Any
    status: str


class InspectorProtocol(Protocol):
    """On-demand inspector boundary used by the preview loop."""

    def inspect(self, exact_frame: Any) -> InspectionView:
        """Analyze a defensive copy of one exact captured frame."""

    def render_working(self, exact_frame: Any) -> Any:
        """Return an immediate progress frame before blocking analysis."""


class MethodSelectorProtocol(InspectorProtocol, Protocol):
    """V1 inspector boundary that also renders the V1/V2 choice."""

    def render_selection(
        self, exact_frame: Any, *, v2_sample_count: int = 20
    ) -> Any:
        """Render the V1/V2 method selector when validated V2 is available."""


class LiveDetectionAdapter:
    """Convert live-tester detections into uncertainty-domain detections."""

    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def detect(self, image: Any) -> list[Detection]:
        output: list[Detection] = []
        for item in self.detector.detect(image):
            if not isinstance(item, DetectionResult):
                raise TypeError("Live detector returned an unsupported detection type")
            output.append(
                Detection(
                    class_id=item.class_id,
                    class_name=item.class_name,
                    confidence=item.confidence,
                    bbox=tuple(float(value) for value in item.bbox),
                )
            )
        return output


class RobustnessInspector:
    """Run V1 robustness only when the operator explicitly presses U."""

    def __init__(
        self,
        detector: Any,
        *,
        cv2_module: Any,
        sample_count: int = 10,
        seed: int = 42,
        match_iou: float = 0.50,
    ) -> None:
        self.detector = LiveDetectionAdapter(detector)
        self.cv2 = cv2_module
        self.sample_count = sample_count
        self.seed = seed
        self.match_iou = match_iou

    def render_working(self, exact_frame: Any) -> Any:
        """Overlay a clear status while the first analysis pass starts."""

        frame = exact_frame.copy()
        overlay = frame.copy()
        self.cv2.rectangle(overlay, (0, 0), (frame.shape[1], 92), (12, 12, 12), -1)
        self.cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.cv2.putText(
            frame,
            "UNCERTAINTY INSPECTION - V1 INPUT ROBUSTNESS",
            (18, 35),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 220, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            f"Running 1 clean + {self.sample_count} perturbed inferences...",
            (18, 70),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            self.cv2.LINE_AA,
        )
        return frame

    def render_selection(
        self, exact_frame: Any, *, v2_sample_count: int = 20
    ) -> Any:
        """Render an on-demand method menu over the unchanged frozen frame."""

        frame = exact_frame.copy()
        overlay = frame.copy()
        self.cv2.rectangle(overlay, (0, 0), (frame.shape[1], 178), (12, 12, 12), -1)
        self.cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        lines = (
            ("UNCERTAINTY METHOD - SAME FROZEN FRAME", (0, 220, 255), 0.70, 2),
            (
                f"1 - V1 INPUT ROBUSTNESS (1 clean + {self.sample_count} input variants)",
                (235, 235, 235),
                0.50,
                1,
            ),
            (
                f"2 - V2 MC DROPOUT ({v2_sample_count} same-frame stochastic passes)",
                (235, 235, 235),
                0.50,
                1,
            ),
            ("ESC / SPACE / U cancel and resume", (180, 180, 180), 0.46, 1),
            ("Neither method is a calibrated correctness probability", (180, 180, 180), 0.43, 1),
        )
        for index, (text, color, scale, thickness) in enumerate(lines):
            self.cv2.putText(
                frame,
                text,
                (18, 34 + index * 32),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                self.cv2.LINE_AA,
            )
        return frame

    def inspect(self, exact_frame: Any) -> InspectionView:
        """Analyze the exact frame copy and render transparent per-target metrics."""

        frozen = exact_frame.copy()
        analysis = analyze_image(
            frozen,
            self.detector,
            sample_count=self.sample_count,
            seed=self.seed,
            match_iou=self.match_iou,
        )
        status = overall_status(
            analysis.baseline_metrics,
            perturbed_only_count=len(analysis.perturbed_only_metrics),
        )
        return InspectionView(
            frame=self._render_analysis(frozen, analysis, status),
            analysis=analysis,
            status=status,
        )

    def _render_analysis(
        self, exact_frame: Any, analysis: ImageAnalysis, status: str
    ) -> Any:
        targets = analysis.baseline_metrics
        perturbed_only = analysis.perturbed_only_metrics
        perturbed_only_height = 55 + len(perturbed_only) * 28 if perturbed_only else 0
        canvas_height = max(
            760, 230 + len(targets) * 250 + perturbed_only_height
        )
        image_width = 900
        panel_width = 650
        canvas = np.full((canvas_height, image_width + panel_width, 3), 16, dtype=np.uint8)
        height, width = exact_frame.shape[:2]
        scale = min(image_width / width, (canvas_height - 70) / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = self.cv2.resize(exact_frame, (resized_width, resized_height))
        offset_y = 58 + max(0, (canvas_height - 58 - resized_height) // 2)
        canvas[offset_y : offset_y + resized_height, 0:resized_width] = resized

        palette = ((0, 220, 255), (80, 220, 100), (255, 165, 0), (255, 120, 210))
        for index, cluster in enumerate(analysis.clusters):
            detection = cluster.observations.get(0)
            if detection is None:
                continue
            x1, y1, x2, y2 = detection.bbox
            points = (
                (int(x1 * scale), offset_y + int(y1 * scale)),
                (int(x2 * scale), offset_y + int(y2 * scale)),
            )
            color = palette[index % len(palette)]
            self.cv2.rectangle(canvas, points[0], points[1], color, 2)
            self.cv2.putText(
                canvas,
                f"target_{cluster.cluster_id}",
                (points[0][0], max(offset_y + 18, points[0][1] - 7)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                self.cv2.LINE_AA,
            )

        self._text(canvas, "UNCERTAINTY INSPECTION - V1 ROBUSTNESS", 18, 34, (0, 220, 255), 0.72, 2)
        panel_x = image_width + 20
        self._text(canvas, "METHOD", panel_x, 34, (0, 220, 255), 0.56, 2)
        self._text(canvas, "Input-perturbation robustness", panel_x, 60)
        self._text(canvas, "Not Bayesian uncertainty or probability", panel_x, 83, (160, 160, 160), 0.45)
        self._text(
            canvas,
            f"1 clean + {analysis.perturbation_count} variants | status: {status}",
            panel_x,
            111,
            (80, 220, 100) if status == "STABLE IN V1 TEST" else (0, 180, 255),
            0.48,
            2,
        )

        self._text(
            canvas,
            f"Perturbation-only target clusters: {len(perturbed_only)}",
            panel_x,
            136,
            (0, 180, 255) if perturbed_only else (160, 160, 160),
            0.45,
        )
        y = 170
        if not targets:
            self._text(canvas, "No objects detected in the clean frame.", panel_x, y)
            y += 27
            self._text(
                canvas,
                "Any variant-only clusters below are not clean-frame detections.",
                panel_x,
                y,
                (170, 170, 170),
                0.43,
            )
            y += 30
        for target in targets:
            self.cv2.line(canvas, (panel_x, y), (image_width + panel_width - 20, y), (65, 65, 65), 1)
            y += 27
            self._text(
                canvas,
                f"{target.target_id} | {safe_overlay_text(target.dominant_class, 35)}",
                panel_x,
                y,
                (235, 235, 235),
                0.56,
                2,
            )
            y += 25
            self._text(
                canvas,
                f"Detected: {target.detection_count}/{target.sample_count}   "
                f"persistence {target.detection_persistence:.3f} "
                f"[{persistence_status(target.detection_persistence)}]",
                panel_x,
                y,
            )
            y += 23
            self._text(canvas, f"Confidence: mean {target.confidence_mean:.3f}   std {target.confidence_std:.3f}", panel_x, y)
            y += 23
            self._text(canvas, f"Class: agreement {target.class_agreement:.3f}   entropy {target.class_entropy_bits:.3f} bits", panel_x, y)
            y += 23
            self._text(canvas, f"Localization: mean reference IoU {target.mean_iou_to_reference:.3f}", panel_x, y)
            y += 23
            self._text(canvas, f"Box std: center ({target.bbox_center_std_pixels.x:.1f}, {target.bbox_center_std_pixels.y:.1f}) px", panel_x, y)
            y += 23
            self._text(canvas, f"Box std: size ({target.bbox_size_std_pixels.x:.1f}, {target.bbox_size_std_pixels.y:.1f}) px", panel_x, y)
            y += 23
            evidence = ", ".join(
                f"{name} {share:.2f}" for name, share in target.class_evidence_share.items()
            )
            self._text(canvas, f"Evidence share: {safe_overlay_text(evidence, 65)}", panel_x, y, (190, 190, 190), 0.43)
            y += 23
            for line in _wrap_text(interpret_target(target), 66):
                self._text(canvas, line, panel_x, y, (0, 210, 255), 0.43)
                y += 20
            y += 9
        if perturbed_only:
            self.cv2.line(
                canvas,
                (panel_x, y),
                (image_width + panel_width - 20, y),
                (65, 65, 65),
                1,
            )
            y += 25
            self._text(
                canvas,
                "PERTURBATION-ONLY CLUSTERS (not detected in clean frame)",
                panel_x,
                y,
                (0, 180, 255),
                0.46,
                2,
            )
            for target in perturbed_only:
                y += 27
                self._text(
                    canvas,
                    f"{target.target_id} | "
                    f"{safe_overlay_text(target.dominant_class, 32)} | "
                    f"observed {target.detection_count}/{target.sample_count} samples",
                    panel_x,
                    y,
                    (210, 210, 210),
                    0.43,
                )
        self._text(
            canvas,
            "P / U / SPACE resume | S saves this inspection | Q exits",
            panel_x,
            canvas_height - 22,
            (235, 235, 235),
            0.48,
            1,
        )
        return canvas

    def _text(
        self,
        image: Any,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = (225, 225, 225),
        scale: float = 0.47,
        thickness: int = 1,
    ) -> None:
        self.cv2.putText(
            image,
            safe_overlay_text(text, 95),
            (x, y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self.cv2.LINE_AA,
        )


def _wrap_text(text: str, width: int) -> tuple[str, ...]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)
