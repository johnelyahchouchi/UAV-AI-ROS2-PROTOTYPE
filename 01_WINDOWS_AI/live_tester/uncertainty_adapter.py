"""Bridge the live detector to the on-demand V1 robustness inspector."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Protocol

import numpy as np

from .domain import DetectionResult, PROJECT_ROOT
from .presentation_ui import (
    FontResolver,
    PillowCanvas,
    Rect,
    THEME,
    calculate_presentation_layout,
    draw_page_footer,
    draw_page_header,
    draw_status_pill,
    paginate,
    presentation_canvas_size,
    ratio_text,
    render_method_selector,
    render_working_overlay,
    share_text,
    status_color,
)


UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
if str(UNCERTAINTY_SRC) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_SRC))

from uav_uncertainty.analysis import ImageAnalysis, analyze_image  # noqa: E402
from uav_uncertainty.detection import Detection  # noqa: E402
from uav_uncertainty.metrics import TargetMetrics  # noqa: E402
from uav_uncertainty.presentation import (  # noqa: E402
    interpret_target,
    overall_status,
    persistence_status,
)


@dataclass(frozen=True)
class InspectionView:
    """Rendered analysis, optional pages, and concise post-inspection status."""

    frame: Any
    analysis: Any
    status: str
    pages: tuple[Any, ...] = ()


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


def _v1_class_status(target: TargetMetrics) -> str:
    if target.class_agreement >= 0.95:
        return "STABLE"
    if target.class_agreement >= 0.80:
        return "INPUT-SENSITIVE"
    return "REVIEW"


def _v1_localization_status(target: TargetMetrics) -> str:
    if target.mean_iou_to_reference >= 0.80:
        return "STABLE"
    if target.mean_iou_to_reference >= 0.60:
        return "INPUT-SENSITIVE"
    return "REVIEW"


def _v1_target_is_stable(target: TargetMetrics) -> bool:
    return (
        persistence_status(target.detection_persistence) == "STABLE"
        and _v1_class_status(target) == "STABLE"
        and _v1_localization_status(target) == "STABLE"
        and target.confidence_std < 0.10
    )


def _v1_frame_reason(
    analysis: ImageAnalysis, status: str, stable_count: int, review_count: int
) -> str:
    total = len(analysis.baseline_metrics) + len(analysis.perturbed_only_metrics)
    if not total:
        return "No detections were observed in the clean or perturbed inputs."
    if analysis.perturbed_only_metrics:
        count = len(analysis.perturbed_only_metrics)
        noun = "cluster appeared" if count == 1 else "clusters appeared"
        return f"FRAME REVIEW — {count} target {noun} only after input perturbation."
    if status == "STABLE IN V1 TEST":
        return f"FRAME STABLE — all {stable_count} target(s) remained robust in this test."
    if review_count:
        return f"FRAME REVIEW — {review_count} of {total} target(s) changed across inputs."
    return f"FRAME REVIEW — {status.lower()} across the tested inputs."


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
        font_resolver: FontResolver | None = None,
    ) -> None:
        self.detector = LiveDetectionAdapter(detector)
        self.cv2 = cv2_module
        self.sample_count = sample_count
        self.seed = seed
        self.match_iou = match_iou
        self.fonts = font_resolver or FontResolver()

    def render_working(self, exact_frame: Any) -> Any:
        """Overlay a clear status while the first analysis pass starts."""

        return render_working_overlay(
            exact_frame,
            title="V1 — INPUT-PERTURBATION ROBUSTNESS",
            primary=f"Running 1 clean + {self.sample_count} perturbed inputs…",
            secondary=(
                "The input changes; this is not Bayesian uncertainty or a calibrated "
                "probability of correctness."
            ),
            fonts=self.fonts,
        )

    def render_selection(
        self, exact_frame: Any, *, v2_sample_count: int = 20
    ) -> Any:
        """Render a centered method menu over the unchanged frozen frame."""

        return render_method_selector(
            exact_frame,
            v1_sample_count=self.sample_count,
            v2_sample_count=v2_sample_count,
            fonts=self.fonts,
        )

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
        pages = self._render_analysis_pages(frozen, analysis, status)
        return InspectionView(
            frame=pages[0], pages=pages, analysis=analysis, status=status
        )

    def _render_analysis(
        self, exact_frame: Any, analysis: ImageAnalysis, status: str
    ) -> Any:
        """Compatibility wrapper returning the first responsive page."""

        return self._render_analysis_pages(exact_frame, analysis, status)[0]

    def _render_analysis_pages(
        self, exact_frame: np.ndarray, analysis: ImageAnalysis, status: str
    ) -> tuple[np.ndarray, ...]:
        items = tuple(("baseline", target) for target in analysis.baseline_metrics) + tuple(
            ("variant", target) for target in analysis.perturbed_only_metrics
        )
        chunks = paginate(items, 2)
        stable_count = sum(
            _v1_target_is_stable(target) for target in analysis.baseline_metrics
        )
        review_count = len(items) - stable_count
        rendered: list[np.ndarray] = []
        for page_index, chunk in enumerate(chunks, start=1):
            width, height = presentation_canvas_size(exact_frame.shape)
            layout = calculate_presentation_layout(width, height, len(chunk))
            base = np.full((height, width, 3), THEME.background[::-1], dtype=np.uint8)
            canvas = PillowCanvas(base, fonts=self.fonts)
            placement = canvas.paste_frozen_frame(
                exact_frame, layout.image, radius=round(12 * layout.scale)
            )
            self._draw_reference_boxes(canvas, placement, analysis)
            draw_page_header(
                canvas,
                layout,
                title="V1 — INPUT-PERTURBATION ROBUSTNESS",
                subtitle=(
                    f"Same frozen frame · 1 clean + {analysis.perturbation_count} "
                    "perturbed inputs · not a correctness probability"
                ),
                method_tag="INPUT STABILITY",
            )
            self._draw_summary(
                canvas,
                layout.summary,
                analysis=analysis,
                status=status,
                stable_count=stable_count,
                review_count=review_count,
                scale=layout.scale,
            )
            if chunk:
                for card, (kind, target) in zip(layout.cards, chunk):
                    self._draw_target_card(
                        canvas,
                        card,
                        target=target,
                        variant_only=kind == "variant",
                        scale=layout.scale,
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
        self, canvas: PillowCanvas, placement: Any, analysis: ImageAnalysis
    ) -> None:
        palette = (THEME.accent, THEME.stable, THEME.review, (208, 118, 222))
        for index, cluster in enumerate(analysis.clusters):
            detection = cluster.observations.get(0)
            if detection is None:
                continue
            x1, y1, x2, y2 = detection.bbox
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
                Rect(label_x, label_y, 92, 20),
                fill=(*color, 225),
                radius=4,
            )
            canvas.text(
                f"TARGET {cluster.cluster_id}",
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
        analysis: ImageAnalysis,
        status: str,
        stable_count: int,
        review_count: int,
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
        total = len(analysis.baseline_metrics) + len(analysis.perturbed_only_metrics)
        overall = (
            "STABLE"
            if status == "STABLE IN V1 TEST"
            else "NO TARGETS" if total == 0 else "REVIEW"
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
        values = (("TARGETS", total), ("STABLE", stable_count), ("REVIEW", review_count))
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
        reason = _v1_frame_reason(analysis, status, stable_count, review_count)
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
        target: TargetMetrics,
        variant_only: bool,
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
        prefix = "VARIANT-ONLY" if variant_only else target.target_id.replace("_", " ").upper()
        canvas.text(
            f"{prefix}  |  {target.dominant_class}",
            x,
            rect.y + round(10 * scale),
            size=13.5 * scale,
            fill=THEME.text,
            bold=True,
            max_width=max_width,
        )
        statuses = (
            (("CLEAN EXISTENCE", "REVIEW"), ("CLASS", "OBSERVED"), ("LOCATION", "OBSERVED"))
            if variant_only
            else (
                ("EXISTENCE", persistence_status(target.detection_persistence)),
                ("CLASS", _v1_class_status(target)),
                ("LOCATION", _v1_localization_status(target)),
            )
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
        detected_label = "Observed" if variant_only else "Persistence"
        lines = (
            (
                f"{detected_label}: {target.detection_count}/{target.sample_count} "
                f"({target.detection_persistence:.3f})  ·  confidence "
                f"{target.confidence_mean:.3f} ± {target.confidence_std:.3f}"
            ),
            f"Class mix: {ratio_text(target.class_histogram, denominator=target.detection_count)}",
            f"Class agreement {target.class_agreement:.3f}  ·  entropy {target.class_entropy_bits:.3f} bits",
            f"Mean reference IoU {target.mean_iou_to_reference:.3f}",
            (
                f"Center σ {target.bbox_center_std_pixels.x:.1f}/{target.bbox_center_std_pixels.y:.1f} px"
                f"  ·  width/height σ {target.bbox_size_std_pixels.x:.1f}/{target.bbox_size_std_pixels.y:.1f} px"
            ),
            f"Evidence share (not probability): {share_text(target.class_evidence_share)}",
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
        interpretation = (
            "Appeared only after input perturbation; no clean-frame target exists for comparison."
            if variant_only
            else interpret_target(target)
        )
        interpretation_y = rect.bottom - round(39 * scale)
        for index, line in enumerate(
            canvas.wrapped_lines(
                interpretation,
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
                fill=THEME.review if variant_only else status_color(statuses[0][1]),
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
            "NO TARGETS OBSERVED",
            x,
            rect.y + round(24 * scale),
            size=16 * scale,
            fill=THEME.text,
            bold=True,
        )
        canvas.text(
            "No clean-frame or perturbation-only detections were available to summarize.",
            x,
            rect.y + round(60 * scale),
            size=11 * scale,
            fill=THEME.muted,
            max_width=rect.width - round(36 * scale),
        )
