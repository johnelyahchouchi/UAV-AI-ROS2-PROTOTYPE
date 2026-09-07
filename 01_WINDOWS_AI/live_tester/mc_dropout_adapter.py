"""Live-tester presentation adapter for validated MC Dropout V2."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from .domain import PROJECT_ROOT, safe_overlay_text
from .uncertainty_adapter import InspectionView


UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
if str(UNCERTAINTY_SRC) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_SRC))

from uav_uncertainty.mc_dropout_ultralytics import (  # noqa: E402
    TrustedUltralyticsMCDORunner,
)
from uav_uncertainty.mc_dropout_v2 import (  # noqa: E402
    MCDOFrameAnalysis,
    MCDOV2Config,
    run_mcdo_frame,
)


class MCDOV2LiveInspector:
    """Run V2 only on an operator-selected copy of one exact frozen frame."""

    def __init__(self, runner: Any, *, cv2_module: Any, config: MCDOV2Config) -> None:
        self.runner = runner
        self.cv2 = cv2_module
        self.config = config

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        *,
        device: str,
        confidence: float,
        nms_iou: float,
        image_size: int,
        sample_count: int,
        match_iou: float,
        cv2_module: Any,
        registry_path: Path | None = None,
    ) -> "MCDOV2LiveInspector":
        """Securely load and validate the external V2 checkpoint."""

        config = MCDOV2Config(
            sample_count=sample_count,
            match_iou=match_iou,
            confidence=confidence,
            nms_iou=nms_iou,
            image_size=image_size,
        )
        runner = TrustedUltralyticsMCDORunner(
            model_path,
            device=device,
            config=config,
            registry_path=registry_path,
        )
        return cls(runner, cv2_module=cv2_module, config=config)

    @property
    def model_sha256(self) -> str:
        """Return the digest verified before checkpoint deserialization."""

        return str(self.runner.model_sha256)

    @property
    def architecture_report(self) -> Any:
        """Return the fail-closed architecture/state report."""

        return self.runner.architecture_report

    def render_working(self, exact_frame: Any) -> Any:
        """Show the scientifically distinct same-frame V2 work state."""

        frame = exact_frame.copy()
        overlay = frame.copy()
        self.cv2.rectangle(overlay, (0, 0), (frame.shape[1], 116), (12, 12, 12), -1)
        self.cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        self.cv2.putText(
            frame,
            "UNCERTAINTY INSPECTION - V2 MC DROPOUT",
            (18, 34),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 220, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            f"Running {self.config.sample_count} stochastic passes on the same exact frame...",
            (18, 69),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 235, 235),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            "6 Dropout2d p=0.20 active; BatchNorm remains in evaluation mode",
            (18, 98),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (175, 175, 175),
            1,
            self.cv2.LINE_AA,
        )
        return frame

    def inspect(self, exact_frame: Any) -> InspectionView:
        """Analyze and render one unchanged frame with the trusted V2 model."""

        frozen = exact_frame.copy()
        analysis = run_mcdo_frame(frozen, self.runner, self.config)
        return InspectionView(
            frame=self._render_analysis(frozen, analysis),
            analysis=analysis,
            status=analysis.status,
        )

    def _render_analysis(self, exact_frame: Any, analysis: MCDOFrameAnalysis) -> Any:
        # Reserve every rendered row plus spacing before placing the fixed footer.
        # The count-dependent rows are the winner and evidence distributions.
        target_heights = [
            344
            + 20 * len(target.winner_class_distribution)
            + 20 * len(target.class_evidence_share or {})
            for target in analysis.targets
        ]
        canvas_height = max(820, 198 + sum(target_heights) + 74)
        image_width = 850
        panel_width = 700
        canvas = np.full((canvas_height, image_width + panel_width, 3), 16, dtype=np.uint8)
        height, width = exact_frame.shape[:2]
        scale = min(image_width / width, (canvas_height - 70) / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = self.cv2.resize(exact_frame, (resized_width, resized_height))
        offset_y = 58 + max(0, (canvas_height - 58 - resized_height) // 2)
        canvas[offset_y : offset_y + resized_height, 0:resized_width] = resized

        palette = ((0, 220, 255), (80, 220, 100), (255, 165, 0), (255, 120, 210))
        for index, target in enumerate(analysis.targets):
            if target.reference_bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = target.reference_bbox_xyxy
            first = (int(x1 * scale), offset_y + int(y1 * scale))
            second = (int(x2 * scale), offset_y + int(y2 * scale))
            color = palette[index % len(palette)]
            self.cv2.rectangle(canvas, first, second, color, 2)
            self.cv2.putText(
                canvas,
                target.target_id,
                (first[0], max(offset_y + 18, first[1] - 7)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                self.cv2.LINE_AA,
            )

        self._text(
            canvas,
            "UNCERTAINTY INSPECTION - V2 MC DROPOUT",
            18,
            34,
            (0, 220, 255),
            0.72,
            2,
        )
        panel_x = image_width + 20
        self._text(canvas, "METHOD", panel_x, 32, (0, 220, 255), 0.55, 2)
        self._text(canvas, "MC Dropout model uncertainty", panel_x, 57)
        self._text(canvas, "Same exact frame", panel_x, 80)
        self._text(
            canvas,
            f"{analysis.sample_count} stochastic model passes | 6 Dropout2d p=0.20",
            panel_x,
            103,
        )
        self._text(canvas, "BatchNorm remains in evaluation mode", panel_x, 126)
        self._text(
            canvas,
            "Approximate model / epistemic uncertainty",
            panel_x,
            149,
            (160, 160, 160),
            0.44,
        )
        self._text(
            canvas,
            "Not calibrated probability of correctness",
            panel_x,
            170,
            (160, 160, 160),
            0.44,
        )
        y = 198
        if not analysis.targets:
            self._text(canvas, "No objects detected in any stochastic pass.", panel_x, y)
            y += 25
            self._text(canvas, "Winner agreement: N/A", panel_x, y)
            y += 23
            self._text(canvas, "Winner entropy: N/A", panel_x, y)
            y += 23
            self._text(canvas, "Evidence share: N/A", panel_x, y)

        for target in analysis.targets:
            self.cv2.line(
                canvas,
                (panel_x, y),
                (image_width + panel_width - 20, y),
                (65, 65, 65),
                1,
            )
            y += 27
            self._text(
                canvas,
                f"{target.target_id} | {safe_overlay_text(target.dominant_winner_class or 'N/A', 35)}",
                panel_x,
                y,
                (235, 235, 235),
                0.56,
                2,
            )
            y += 25
            self._text(
                canvas,
                f"Detected: {target.detected_count}/{target.sample_count} | persistence {target.persistence:.3f}",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                f"Existence: {target.existence_status} | Class: {target.classification_status} | Localization: {target.localization_status}",
                panel_x,
                y,
                (0, 210, 255) if target.classification_status == "UNCERTAIN" else (220, 220, 220),
                0.44,
            )
            y += 25
            self._text(canvas, "Winner classes:", panel_x, y, (190, 190, 190), 0.43)
            for class_name, count in target.winner_class_distribution.items():
                y += 20
                self._text(
                    canvas,
                    f"  {safe_overlay_text(class_name, 34)}: {count}/{target.detected_count}",
                    panel_x,
                    y,
                    (210, 210, 210),
                    0.43,
                )
            y += 23
            self._text(
                canvas,
                f"Winner agreement: {_format_metric(target.winner_class_agreement)} | entropy {_format_metric(target.winner_class_entropy_bits)} bits",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                f"Competition: {target.competition_count}/{target.detected_count} | rate {_format_metric(target.competition_rate)}",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                f"Winner confidence: mean {_format_metric(target.winner_confidence_mean)} | std {_format_metric(target.winner_confidence_std)}",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                f"Predicted-box consistency: mean reference IoU {_format_metric(target.mean_reference_iou)} | min {_format_metric(target.minimum_reference_iou)}",
                panel_x,
                y,
                scale=0.43,
            )
            y += 23
            center = target.bbox_center_std_pixels
            size = target.bbox_size_std_pixels
            self._text(
                canvas,
                "Center std: N/A" if center is None else f"Center std: {center.x:.1f}, {center.y:.1f} px",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                "Size std: N/A" if size is None else f"Size std: {size.x:.1f} x {size.y:.1f} px",
                panel_x,
                y,
            )
            y += 24
            self._text(canvas, "EVIDENCE SHARE (not probability):", panel_x, y, (190, 190, 190), 0.43)
            if target.class_evidence_share is None:
                y += 20
                self._text(canvas, "  N/A", panel_x, y, (190, 190, 190), 0.43)
            else:
                for class_name, share in target.class_evidence_share.items():
                    y += 20
                    self._text(
                        canvas,
                        f"  {safe_overlay_text(class_name, 34)} evidence share: {share:.3f}",
                        panel_x,
                        y,
                        (190, 190, 190),
                        0.43,
                    )
            y += 22
            self._text(
                canvas,
                f"Evidence entropy: {_format_metric(target.evidence_entropy_bits)} bits",
                panel_x,
                y,
            )
            y += 23
            self._text(
                canvas,
                safe_overlay_text(target.interpretation, 78),
                panel_x,
                y,
                (0, 210, 255),
                0.43,
            )
            y += 25

        self._text(
            canvas,
            f"Interpretation: {analysis.status}",
            panel_x,
            canvas_height - 47,
            (0, 220, 255),
            0.48,
            2,
        )
        self._text(
            canvas,
            "P / U / SPACE resume | S saves this inspection | Q exits",
            panel_x,
            canvas_height - 22,
            (235, 235, 235),
            0.48,
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
            safe_overlay_text(text, 105),
            (x, y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self.cv2.LINE_AA,
        )


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
