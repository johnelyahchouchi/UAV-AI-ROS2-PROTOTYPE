"""Live-tester presentation adapter for validated MC Dropout V2."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from .domain import PROJECT_ROOT
from .presentation_ui import (
    FontResolver,
    PillowCanvas,
    Rect,
    THEME,
    calculate_presentation_layout,
    draw_page_footer,
    draw_page_header,
    draw_status_pill,
    finite_metric,
    paginate,
    presentation_canvas_size,
    ratio_text,
    render_working_overlay,
    share_text,
    status_color,
)
from .uncertainty_adapter import InspectionView


UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
if str(UNCERTAINTY_SRC) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_SRC))

from uav_uncertainty.mc_dropout_ultralytics import (  # noqa: E402
    TrustedUltralyticsMCDORunner,
)
from uav_uncertainty.mc_dropout_v2 import (  # noqa: E402
    MCDOFrameAnalysis,
    MCDOTargetResult,
    MCDOV2Config,
    run_mcdo_frame,
)


def _target_is_stable(target: MCDOTargetResult) -> bool:
    return (
        target.existence_status == "STABLE"
        and target.classification_status == "STABLE"
        and target.localization_status == "STABLE"
    )


def mcdo_frame_summary(
    targets: Sequence[MCDOTargetResult],
) -> tuple[int, int, str, str]:
    """Return stable/review counts and a dimension-consistent frame reason."""

    total = len(targets)
    stable = sum(_target_is_stable(target) for target in targets)
    review = total - stable
    if not targets:
        return (0, 0, "REVIEW", "FRAME REVIEW — no stochastic detections were observed.")
    if review == 0:
        return (
            stable,
            0,
            "STABLE",
            f"FRAME STABLE — all {total} target(s) are stable across the model passes.",
        )

    checks = (
        (
            "unstable existence",
            lambda target: target.existence_status == "UNSTABLE / REVIEW",
        ),
        (
            "unstable localization",
            lambda target: target.localization_status == "UNSTABLE / REVIEW",
        ),
        (
            "uncertain classification",
            lambda target: target.classification_status == "UNCERTAIN",
        ),
        (
            "variable model output",
            lambda target: not _target_is_stable(target),
        ),
    )
    for description, predicate in checks:
        count = sum(bool(predicate(target)) for target in targets)
        if count:
            verb = "shows" if count == 1 else "show"
            return (
                stable,
                review,
                "REVIEW",
                f"FRAME REVIEW — {count} of {total} targets {verb} {description}.",
            )
    raise AssertionError("review targets must have a visible review reason")


class MCDOV2LiveInspector:
    """Run V2 only on an operator-selected copy of one exact frozen frame."""

    def __init__(
        self,
        runner: Any,
        *,
        cv2_module: Any,
        config: MCDOV2Config,
        font_resolver: FontResolver | None = None,
    ) -> None:
        self.runner = runner
        self.cv2 = cv2_module
        self.config = config
        self.fonts = font_resolver or FontResolver()

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

        return render_working_overlay(
            exact_frame,
            title="V2 — MC DROPOUT MODEL UNCERTAINTY",
            primary=(
                f"Running {self.config.sample_count} stochastic model passes on the "
                "same exact frame…"
            ),
            secondary=(
                "6 Dropout2d · p = 0.20 · BatchNorm = evaluation · approximate "
                "epistemic signal"
            ),
            fonts=self.fonts,
        )

    def inspect(self, exact_frame: Any) -> InspectionView:
        """Analyze and render one unchanged frame with the trusted V2 model."""

        frozen = exact_frame.copy()
        analysis = run_mcdo_frame(frozen, self.runner, self.config)
        pages = self._render_analysis_pages(frozen, analysis)
        return InspectionView(
            frame=pages[0], pages=pages, analysis=analysis, status=analysis.status
        )

    def _render_analysis(self, exact_frame: Any, analysis: MCDOFrameAnalysis) -> Any:
        """Compatibility wrapper returning the first responsive page."""

        return self._render_analysis_pages(exact_frame, analysis)[0]

    def _render_analysis_pages(
        self, exact_frame: np.ndarray, analysis: MCDOFrameAnalysis
    ) -> tuple[np.ndarray, ...]:
        chunks = paginate(analysis.targets, 2)
        stable_count, review_count, overall, reason = mcdo_frame_summary(
            analysis.targets
        )
        rendered: list[np.ndarray] = []
        for page_index, chunk in enumerate(chunks, start=1):
            width, height = presentation_canvas_size(exact_frame.shape)
            layout = calculate_presentation_layout(width, height, len(chunk))
            base = np.full((height, width, 3), THEME.background[::-1], dtype=np.uint8)
            canvas = PillowCanvas(base, fonts=self.fonts)
            placement = canvas.paste_frozen_frame(
                exact_frame, layout.image, radius=round(12 * layout.scale)
            )
            self._draw_reference_boxes(canvas, placement, analysis.targets)
            draw_page_header(
                canvas,
                layout,
                title="V2 — MC DROPOUT UNCERTAINTY",
                subtitle=(
                    f"Same frozen frame · {analysis.sample_count} stochastic passes · "
                    "6 Dropout2d · p = 0.20 · BatchNorm = evaluation"
                ),
                method_tag="EPISTEMIC SIGNAL",
            )
            self._draw_summary(
                canvas,
                layout.summary,
                total=len(analysis.targets),
                stable=stable_count,
                review=review_count,
                overall=overall,
                reason=reason,
                scale=layout.scale,
            )
            if chunk:
                for card, target in zip(layout.cards, chunk):
                    self._draw_target_card(
                        canvas, card, target=target, scale=layout.scale
                    )
            else:
                self._draw_empty_card(canvas, layout.cards[0], scale=layout.scale)
            draw_page_footer(
                canvas,
                layout,
                page=page_index,
                page_count=len(chunks),
            )
            rendered.append(canvas.render())
        return tuple(rendered)

    def _draw_reference_boxes(
        self,
        canvas: PillowCanvas,
        placement: Any,
        targets: Sequence[MCDOTargetResult],
    ) -> None:
        palette = (THEME.accent, THEME.stable, THEME.review, (208, 118, 222))
        for index, target in enumerate(targets):
            if target.reference_bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = target.reference_bbox_xyxy
            left = placement.x + round(x1 * placement.scale)
            top = placement.y + round(y1 * placement.scale)
            right = placement.x + round(x2 * placement.scale)
            bottom = placement.y + round(y2 * placement.scale)
            left = max(placement.x, min(left, placement.x + placement.width - 2))
            right = max(left + 1, min(right, placement.x + placement.width - 1))
            top = max(placement.y, min(top, placement.y + placement.height - 2))
            bottom = max(top + 1, min(bottom, placement.y + placement.height - 1))
            color = palette[index % len(palette)]
            canvas.draw.rectangle(
                (left, top, right, bottom),
                outline=(*color, 255),
                width=max(2, round(2 * min(placement.scale, 2.0))),
            )
            label_y = max(placement.y + 4, top - 22)
            label_x = min(left, placement.x + placement.width - 92)
            canvas.rectangle(
                Rect(label_x, label_y, 92, 20), fill=(*color, 225), radius=4
            )
            canvas.text(
                target.target_id.replace("_", " ").upper(),
                label_x + 6,
                label_y + 3,
                size=10.5,
                fill=(8, 14, 24),
                bold=True,
                max_width=80,
            )

    def _draw_summary(
        self,
        canvas: PillowCanvas,
        rect: Rect,
        *,
        total: int,
        stable: int,
        review: int,
        overall: str,
        reason: str,
        scale: float,
    ) -> None:
        canvas.rectangle(
            rect,
            fill=(*THEME.surface, 255),
            outline=(*THEME.border, 255),
            radius=round(12 * scale),
        )
        padding = round(14 * scale)
        x = rect.x + padding
        canvas.text(
            "FRAME SUMMARY",
            x,
            rect.y + round(12 * scale),
            size=11 * scale,
            fill=THEME.muted,
            bold=True,
        )
        draw_status_pill(
            canvas,
            overall,
            x=rect.right - round(112 * scale),
            y=rect.y + round(10 * scale),
            size=10.5 * scale,
            max_width=round(100 * scale),
        )
        metrics_y = rect.y + round(43 * scale)
        values = (("TARGETS", total), ("STABLE", stable), ("REVIEW", review))
        column_width = (rect.width - 2 * padding) // 3
        for index, (label, value) in enumerate(values):
            column_x = x + index * column_width
            canvas.text(value, column_x, metrics_y, size=20 * scale, fill=THEME.text, bold=True)
            canvas.text(
                label,
                column_x,
                metrics_y + round(27 * scale),
                size=9.5 * scale,
                fill=THEME.muted,
                bold=True,
            )
        reason_y = rect.y + round(88 * scale)
        for line_index, line in enumerate(
            canvas.wrapped_lines(
                reason,
                size=10.5 * scale,
                max_width=rect.width - 2 * padding,
                max_lines=2,
                bold=True,
            )
        ):
            canvas.text(
                line,
                x,
                reason_y + line_index * round(15 * scale),
                size=10.5 * scale,
                fill=status_color(overall),
                bold=True,
            )

    def _draw_target_card(
        self,
        canvas: PillowCanvas,
        rect: Rect,
        *,
        target: MCDOTargetResult,
        scale: float,
    ) -> None:
        canvas.rectangle(
            rect,
            fill=(*THEME.surface, 255),
            outline=(*THEME.border, 255),
            radius=round(12 * scale),
        )
        padding = round(13 * scale)
        x = rect.x + padding
        max_width = rect.width - 2 * padding
        class_name = target.dominant_winner_class or "N/A"
        canvas.text(
            f"{target.target_id.replace('_', ' ').upper()}  |  {class_name}",
            x,
            rect.y + round(10 * scale),
            size=13.5 * scale,
            fill=THEME.text,
            bold=True,
            max_width=max_width,
        )
        statuses = (
            ("EXISTENCE", target.existence_status),
            ("CLASSIFICATION", target.classification_status),
            ("LOCALIZATION", target.localization_status),
        )
        columns_y = rect.y + round(38 * scale)
        column_width = max_width // 3
        for index, (label, value) in enumerate(statuses):
            column_x = x + index * column_width
            canvas.text(
                label,
                column_x,
                columns_y,
                size=8.2 * scale,
                fill=THEME.muted,
                bold=True,
                max_width=column_width - 5,
            )
            draw_status_pill(
                canvas,
                value,
                x=column_x,
                y=columns_y + round(13 * scale),
                size=8.7 * scale,
                max_width=column_width - round(6 * scale),
            )

        line_height = round(16 * scale)
        metrics_y = rect.y + round(78 * scale)
        center = target.bbox_center_std_pixels
        size = target.bbox_size_std_pixels
        center_text = "N/A" if center is None else f"{center.x:.1f}/{center.y:.1f} px"
        size_text = "N/A" if size is None else f"{size.x:.1f}/{size.y:.1f} px"
        lines = (
            (
                f"Persistence: {target.detected_count}/{target.sample_count} "
                f"({target.persistence:.3f})  ·  confidence "
                f"{finite_metric(target.winner_confidence_mean)} ± "
                f"{finite_metric(target.winner_confidence_std)}"
            ),
            (
                "Winner mix: "
                + ratio_text(
                    target.winner_class_distribution,
                    denominator=max(1, target.detected_count),
                )
            ),
            (
                f"Class entropy {finite_metric(target.winner_class_entropy_bits)} bits"
                f"  ·  competition {finite_metric(target.competition_rate)}"
            ),
            (
                f"Mean reference IoU {finite_metric(target.mean_reference_iou)}  ·  "
                f"min {finite_metric(target.minimum_reference_iou)}"
            ),
            (
                f"Center σ {center_text}"
                f"  ·  Width/height σ {size_text}"
            ),
            (
                f"Evidence share (not probability): {share_text(target.class_evidence_share)}  ·  "
                f"entropy {finite_metric(target.evidence_entropy_bits)} bits"
            ),
        )
        for index, line in enumerate(lines):
            canvas.text(
                line,
                x,
                metrics_y + index * line_height,
                size=(11.0 if index < 4 else 9.7) * scale,
                fill=THEME.text if index < 4 else THEME.muted,
                max_width=max_width,
            )
        interpretation_y = rect.bottom - round(39 * scale)
        for index, line in enumerate(
            canvas.wrapped_lines(
                target.interpretation,
                size=9.2 * scale,
                max_width=max_width,
                max_lines=2,
                bold=True,
            )
        ):
            canvas.text(
                line,
                x,
                interpretation_y + index * round(14 * scale),
                size=9.2 * scale,
                fill=status_color(
                    "STABLE" if _target_is_stable(target) else "REVIEW"
                ),
                bold=True,
            )

    def _draw_empty_card(
        self, canvas: PillowCanvas, rect: Rect, *, scale: float
    ) -> None:
        canvas.rectangle(
            rect,
            fill=(*THEME.surface, 255),
            outline=(*THEME.border, 255),
            radius=round(12 * scale),
        )
        x = rect.x + round(18 * scale)
        canvas.text(
            "NO STOCHASTIC DETECTIONS",
            x,
            rect.y + round(24 * scale),
            size=16 * scale,
            fill=THEME.text,
            bold=True,
        )
        canvas.text(
            "Existence, classification, and localization metrics are unavailable for this frame.",
            x,
            rect.y + round(60 * scale),
            size=11 * scale,
            fill=THEME.muted,
            max_width=rect.width - round(36 * scale),
        )
        canvas.text(
            "This is an approximate epistemic signal, not a calibrated correctness probability.",
            x,
            rect.y + round(86 * scale),
            size=10 * scale,
            fill=THEME.review,
            max_width=rect.width - round(36 * scale),
        )
