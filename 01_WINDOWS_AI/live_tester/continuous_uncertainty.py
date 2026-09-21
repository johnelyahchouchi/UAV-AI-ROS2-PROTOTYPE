"""Asynchronous uncertainty sampling and a unified live presentation workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from .domain import FrameMetrics, safe_overlay_text
from .explanation import (
    MAX_VISUAL_SAMPLES, FramePreview, SamplePreview, draw_preview,
    frame_preview, sample_preview,
)
from .presentation_ui import (
    FontResolver,
    PillowCanvas,
    Rect,
    THEME,
    draw_status_pill,
    presentation_canvas_size,
)


@dataclass(frozen=True)
class MethodSnapshot:
    """Immutable display state for one continuously sampled method."""

    method: str
    title: str
    scope: str
    state: str
    status: str
    lines: tuple[str, ...] = ()
    frame_number: int | None = None
    duration_ms: float | None = None
    updated_at: float | None = None
    sampled_at: float | None = None
    sampled_utc: str | None = None
    reference: FramePreview | None = None
    previews: tuple[SamplePreview, ...] = ()
    completed: int = 0
    total: int = 0


@dataclass(frozen=True)
class ContinuousSnapshot:
    """Latest V1 and V2 states rendered beside the live detector."""

    v1: MethodSnapshot
    v2: MethodSnapshot
    running: bool = False


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _metric(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _summarize_v1(view: Any, frame_number: int, duration_ms: float, now: float) -> MethodSnapshot:
    analysis = view.analysis
    baseline = tuple(getattr(analysis, "baseline_metrics", ()))
    variant_only = tuple(getattr(analysis, "perturbed_only_metrics", ()))
    targets = baseline + variant_only
    persistence = _mean([target.detection_persistence for target in targets])
    agreement = _mean([target.class_agreement for target in targets])
    reference_iou = _mean([target.mean_iou_to_reference for target in targets])
    center_std = _mean(
        [
            (target.bbox_center_std_pixels.x + target.bbox_center_std_pixels.y) / 2.0
            for target in targets
        ]
    )
    size_std = _mean(
        [
            (target.bbox_size_std_pixels.x + target.bbox_size_std_pixels.y) / 2.0
            for target in targets
        ]
    )
    samples = int(getattr(analysis, "perturbation_count", 0)) + 1
    status = str(view.status)
    return MethodSnapshot(
        method="v1",
        title="V1 INPUT STABILITY",
        scope="Input-perturbation robustness",
        state="READY",
        status=status,
        lines=(
            f"Targets {len(targets)} | clean {len(baseline)} | variant-only {len(variant_only)}",
            f"Mean persistence {_metric(persistence)} across {samples} inputs",
            f"Class agreement {_metric(agreement)} | reference IoU {_metric(reference_iou)}",
            f"Center variation {_metric(center_std, 1)} px | size variation {_metric(size_std, 1)} px",
            "Input changes between samples; this is not a correctness probability.",
        ),
        frame_number=frame_number,
        duration_ms=duration_ms,
        updated_at=now,
    )


def _summarize_v2(view: Any, frame_number: int, duration_ms: float, now: float) -> MethodSnapshot:
    analysis = view.analysis
    targets = tuple(getattr(analysis, "targets", ()))
    persistence = _mean([target.persistence for target in targets])
    agreement = _mean(
        [
            target.winner_class_agreement
            for target in targets
            if target.winner_class_agreement is not None
        ]
    )
    reference_iou = _mean(
        [
            target.mean_reference_iou
            for target in targets
            if target.mean_reference_iou is not None
        ]
    )
    confidence_std = _mean(
        [
            target.winner_confidence_std
            for target in targets
            if target.winner_confidence_std is not None
        ]
    )
    sample_count = int(getattr(analysis, "sample_count", 0))
    return MethodSnapshot(
        method="v2",
        title="V2 MC DROPOUT",
        scope="Approximate epistemic signal",
        state="READY",
        status=str(view.status),
        lines=(
            f"Targets {len(targets)} across {sample_count} unchanged-frame passes",
            f"Mean persistence {_metric(persistence)} | class agreement {_metric(agreement)}",
            f"Reference IoU {_metric(reference_iou)} | confidence std {_metric(confidence_std)}",
            "6 Dropout2d at p=0.20 | BatchNorm remains in evaluation mode.",
            "Model output variation; this is not a correctness probability.",
        ),
        frame_number=frame_number,
        duration_ms=duration_ms,
        updated_at=now,
    )


class ContinuousUncertaintyController:
    """Analyze periodic frame copies without blocking capture or building a backlog."""

    def __init__(
        self,
        v1_inspector: Any,
        v2_inspector: Any | None,
        *,
        interval_seconds: float,
        v2_unavailable_reason: str = "Validated V2 checkpoint is not configured",
        clock: Callable[[], float] = time.perf_counter,
        result_sink: Callable[[Any, MethodSnapshot], None] | None = None,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("Continuous uncertainty interval must be positive")
        self.v1_inspector = v1_inspector
        self.v2_inspector = v2_inspector
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.result_sink = result_sink
        self._lock = threading.Lock()
        self._closed = False
        self._running = False
        self._next_due = 0.0
        self._thread: threading.Thread | None = None
        self._snapshot = ContinuousSnapshot(
            v1=MethodSnapshot(
                "v1",
                "V1 INPUT STABILITY",
                "Input-perturbation robustness",
                "WAITING",
                "WAITING FOR FIRST SAMPLE",
            ),
            v2=(
                MethodSnapshot(
                    "v2",
                    "V2 MC DROPOUT",
                    "Approximate epistemic signal",
                    "WAITING",
                    "WAITING FOR FIRST SAMPLE",
                )
                if v2_inspector is not None
                else MethodSnapshot(
                    "v2",
                    "V2 MC DROPOUT",
                    "Approximate epistemic signal",
                    "UNAVAILABLE",
                    "UNAVAILABLE",
                    (safe_overlay_text(v2_unavailable_reason, 86),),
                )
            ),
        )

    def submit(
        self, exact_frame: Any, *, frame_number: int, now: float | None = None,
        captured_at: float | None = None, captured_utc: str | None = None,
    ) -> bool:
        """Start one V1/V2 cycle when due; return false while prior work is active."""

        submitted_at = self.clock() if now is None else float(now)
        sampled_at = submitted_at if captured_at is None else captured_at
        sampled_utc = captured_utc or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            if self._closed or self._running or submitted_at < self._next_due:
                return False
            self._running = True
            self._next_due = submitted_at + self.interval_seconds
            self._snapshot = ContinuousSnapshot(
                v1=replace(
                    self._snapshot.v1,
                    state="ANALYZING",
                    status="ANALYZING SNAPSHOT",
                    frame_number=frame_number,
                    lines=(), duration_ms=None, updated_at=None,
                    sampled_at=sampled_at, sampled_utc=sampled_utc,
                    reference=None, previews=(), completed=0, total=0,
                ),
                v2=(
                    replace(
                        self._snapshot.v2,
                        state="QUEUED",
                        status="QUEUED AFTER V1",
                        frame_number=frame_number,
                        lines=(), duration_ms=None, updated_at=None,
                        sampled_at=sampled_at, sampled_utc=sampled_utc,
                        reference=None, previews=(), completed=0, total=0,
                    )
                    if self.v2_inspector is not None
                    else self._snapshot.v2
                ),
                running=True,
            )
        frozen = exact_frame.copy()
        self._thread = threading.Thread(
            target=self._run_cycle,
            args=(frozen, frame_number),
            name="uav-continuous-uncertainty",
            daemon=True,
        )
        self._thread.start()
        return True

    def _inspect(self, inspector: Any, frame: Any, observer: Callable[[Any], None]) -> Any:
        streaming = getattr(inspector, "analyze_with_observer", None)
        if callable(streaming):
            return streaming(frame, observer=observer)
        analyze = getattr(inspector, "analyze", None)
        return analyze(frame) if callable(analyze) else inspector.inspect(frame)

    def _run_method(
        self,
        inspector: Any,
        frame: Any,
        *,
        method: str,
        frame_number: int,
    ) -> MethodSnapshot:
        started = self.clock()
        with self._lock:
            initial = getattr(self._snapshot, method)
        previews: list[SamplePreview] = []
        reference = None
        completed = total = 0

        def observe(observation: Any) -> None:
            nonlocal completed, total
            previews.append(sample_preview(observation))
            del previews[:-MAX_VISUAL_SAMPLES]
            completed, total = observation.sample_index + 1, observation.total
            with self._lock:
                if self._closed:
                    return
                state = replace(
                    initial, state="ANALYZING", status=f"Completed {completed} / {total}",
                    reference=reference, previews=tuple(previews),
                    completed=completed, total=total,
                )
                self._snapshot = replace(self._snapshot, **{method: state})

        def attach(state: MethodSnapshot) -> MethodSnapshot:
            return replace(
                state, sampled_at=initial.sampled_at, sampled_utc=initial.sampled_utc,
                reference=reference, previews=tuple(previews), completed=completed, total=total,
            )

        def publish(view: Any, state: MethodSnapshot) -> MethodSnapshot:
            if self.result_sink is not None:
                try:
                    self.result_sink(view, state)
                except Exception as error:
                    print(f"Extraction journal write failed: {error}", flush=True)
                    return replace(state, status=f"{state.status} | EXPORT FAILED")
            return state

        try:
            reference = frame_preview(frame)
            view = self._inspect(inspector, frame.copy(), observe)
            finished = self.clock()
            duration_ms = max(0.0, (finished - started) * 1000.0)
            summarizer = _summarize_v1 if method == "v1" else _summarize_v2
            return publish(view, attach(summarizer(view, frame_number, duration_ms, finished)))
        except Exception as error:
            finished = self.clock()
            return publish(None, attach(MethodSnapshot(
                method=method,
                title="V1 INPUT STABILITY" if method == "v1" else "V2 MC DROPOUT",
                scope=(
                    "Input-perturbation robustness"
                    if method == "v1"
                    else "Approximate epistemic signal"
                ),
                state="FAILED",
                status="ANALYSIS FAILED",
                lines=(safe_overlay_text(error, 86), "Live detection continues; review configuration and logs."),
                frame_number=frame_number,
                duration_ms=max(0.0, (finished - started) * 1000.0),
                updated_at=finished,
            )))

    def _run_cycle(self, frame: Any, frame_number: int) -> None:
        v1 = self._run_method(
            self.v1_inspector, frame, method="v1", frame_number=frame_number
        )
        with self._lock:
            v2 = self._snapshot.v2
            if self.v2_inspector is not None and not self._closed:
                v2 = replace(v2, state="ANALYZING", status="ANALYZING SNAPSHOT")
            self._snapshot = ContinuousSnapshot(v1=v1, v2=v2, running=True)

        if self.v2_inspector is not None and not self._closed:
            v2 = self._run_method(
                self.v2_inspector, frame, method="v2", frame_number=frame_number
            )
        with self._lock:
            self._running = False
            self._snapshot = ContinuousSnapshot(v1=v1, v2=v2, running=False)

    def snapshot(self) -> ContinuousSnapshot:
        """Return the latest immutable display state."""

        with self._lock:
            return self._snapshot

    def close(self) -> None:
        """Stop accepting new frame samples; active inference finishes in the daemon."""

        with self._lock:
            self._closed = True


class ContinuousWorkspaceRenderer:
    """Render live detections and separate V1/V2 cards in one polished frame."""

    def __init__(self, *, fonts: FontResolver | None = None) -> None:
        self.fonts = fonts or FontResolver()
        self.zoomed = False

    def render(
        self,
        live_frame: np.ndarray,
        snapshot: ContinuousSnapshot,
        *,
        metrics: FrameMetrics,
        model_name: str,
        device: str,
        source_label: str,
        detection_count: int,
        interval_seconds: float,
        paused: bool,
        now: float,
        exact_inspection_available: bool,
    ) -> np.ndarray:
        width, height = presentation_canvas_size(live_frame.shape)
        scale = height / 720.0
        margin = round(16 * scale)
        gap = round(12 * scale)
        header_height = round(72 * scale)
        footer_height = round(35 * scale)
        body_y = margin + header_height
        body_bottom = height - margin - footer_height
        body_height = body_bottom - body_y
        usable_width = width - 2 * margin - gap
        video_width = round(usable_width * 0.56)
        video_rect = Rect(margin, body_y, video_width, body_height - round(136 * scale))
        panel_x = video_rect.right + gap
        panel_width = width - margin - panel_x
        panel_gap = gap
        panel_height = (body_height - panel_gap) // 2
        v1_rect = Rect(panel_x, body_y, panel_width, panel_height)
        v2_rect = Rect(panel_x, v1_rect.bottom + panel_gap, panel_width, panel_height)

        base = np.full((height, width, 3), THEME.background[::-1], dtype=np.uint8)
        canvas = PillowCanvas(base, fonts=self.fonts)
        canvas.text(
            "LIVE PERCEPTION + UNCERTAINTY",
            margin,
            margin + round(2 * scale),
            size=22 * scale,
            fill=THEME.text,
            bold=True,
        )
        canvas.text(
            f"One base detection pass per displayed frame | sampled V1/V2 refresh every {interval_seconds:g} s",
            margin,
            margin + round(35 * scale),
            size=10.5 * scale,
            fill=THEME.muted,
            max_width=round(width * 0.69),
        )
        draw_status_pill(
            canvas,
            "PAUSED" if paused else "LIVE",
            x=width - margin - round(92 * scale),
            y=margin + round(9 * scale),
            size=10.5 * scale,
            max_width=round(92 * scale),
        )
        canvas.paste_frozen_frame(live_frame, video_rect, radius=round(12 * scale))
        guide = Rect(margin, video_rect.bottom + gap, video_width, body_bottom - video_rect.bottom - gap)
        canvas.rectangle(guide, fill=(*THEME.surface, 255), radius=round(12 * scale))
        guide_lines = (
            "HOW TO READ THE EXPERIMENTS",
            "V1 changes the input. V2 keeps the input fixed and changes dropout masks.",
            "Panels replay completed samples; their source frame is older than the live video.",
            "Z: toggle 2x center zoom (view only). Cyan: selected output; amber: V2 pass history.",
            "Measurements describe variation, not a calibrated probability of correctness.",
        )
        for index, line in enumerate(guide_lines):
            canvas.text(
                line, guide.x + round(14 * scale), guide.y + round((10 + index * 21) * scale),
                size=(11 if index == 0 else 9.5) * scale,
                fill=THEME.accent if index == 0 else THEME.muted,
                bold=index == 0, max_width=guide.width - round(28 * scale),
            )
        self._draw_method_card(canvas, v1_rect, snapshot.v1, scale=scale, now=now)
        self._draw_method_card(canvas, v2_rect, snapshot.v2, scale=scale, now=now)

        footer = (
            "Q/ESC quit | P pause | S save | H HUD | Z zoom | U report"
            if exact_inspection_available
            else "Q/ESC quit | P pause | S save | H HUD | Z zoom"
        )
        canvas.text(
            footer,
            margin,
            body_bottom + round(9 * scale),
            size=9.5 * scale,
            fill=THEME.muted,
        )
        operational = (
            f"{safe_overlay_text(model_name, 28)} | {device} | {detection_count} detections | "
            f"{metrics.fps:.1f} FPS | {safe_overlay_text(source_label, 42)}"
        )
        canvas.text(
            operational,
            width - margin,
            body_bottom + round(9 * scale),
            size=9.5 * scale,
            fill=THEME.text,
            anchor="ra",
            max_width=round(width * 0.55),
        )
        return canvas.render()

    def _draw_method_card(
        self,
        canvas: PillowCanvas,
        rect: Rect,
        state: MethodSnapshot,
        *,
        scale: float,
        now: float,
    ) -> None:
        canvas.rectangle(
            rect,
            fill=(*THEME.surface, 255),
            outline=(*THEME.border, 255),
            radius=round(12 * scale),
        )
        pad = round(14 * scale)
        x = rect.x + pad
        max_width = rect.width - 2 * pad
        canvas.text(
            state.title,
            x,
            rect.y + round(12 * scale),
            size=14 * scale,
            fill=THEME.text,
            bold=True,
            max_width=max_width - round(100 * scale),
        )
        draw_status_pill(
            canvas,
            state.state,
            x=rect.right - pad - round(96 * scale),
            y=rect.y + round(10 * scale),
            size=8.5 * scale,
            max_width=round(96 * scale),
        )
        def text(value: str, offset: int, *, accent: bool = False) -> None:
            canvas.text(
                value, x, rect.y + round(offset * scale), size=9.5 * scale,
                fill=THEME.accent if accent else THEME.muted, max_width=max_width,
            )

        if state.sampled_at is None:
            text(state.scope, 38)
            text(state.status, 72, accent=True)
            for index, line in enumerate(state.lines[:4]):
                text(line, 100 + index * 24)
            return

        age = max(0.0, now - state.sampled_at)
        utc = (state.sampled_utc or "UTC unavailable").replace("T", " ").replace("+00:00", " UTC")
        text(f"Frame {state.frame_number} | {utc}", 36)
        duration = "" if state.duration_ms is None else f" | analysis {state.duration_ms:.0f} ms"
        text(f"Source age {age:.1f}s{duration}", 53)

        if not state.previews or state.reference is None:
            text(state.status, 94, accent=True)
            for index, line in enumerate(state.lines[:5]):
                text(line, 122 + index * 22)
            return

        # Replay only real completed observations; never invent progress or model passes.
        index = int(now / 0.9) % len(state.previews)
        selected = state.previews[index]
        zoom = "2x center / view only" if self.zoomed else "Full frame"
        text(f"{zoom} | Replay completed sample {selected.sample_index + 1}/{selected.total}", 73, accent=True)
        gap = round(8 * scale)
        preview_width = (max_width - gap) // 2
        left = Rect(x, rect.y + round(108 * scale), preview_width, round(78 * scale))
        right = Rect(left.right + gap, left.y, preview_width, left.height)
        text("Clean reference" if state.method == "v1" else "Unchanged input", 92)
        canvas.text(
            selected.family if state.method == "v1" else f"Stochastic pass {selected.sample_index + 1}",
            right.x, rect.y + round(92 * scale), size=9.5 * scale,
            fill=THEME.text, max_width=preview_width,
        )
        draw_preview(canvas, state.reference, left, zoomed=self.zoomed)
        draw_preview(
            canvas, selected.pixels, right, zoomed=self.zoomed, boxes=selected.boxes,
            history=state.previews if state.method == "v2" else (),
        )
        detail = selected.parameters if state.method == "v1" else "Same pixels | dropout active | BatchNorm eval"
        text(f"{detail} | detections {selected.detection_count}", 190)
        retained = f" | latest {len(state.previews)} shown" if state.completed > len(state.previews) else ""
        text(f"Completed {state.completed}/{state.total}{retained} | {state.status}", 210, accent=True)
        bar_y = rect.y + round(229 * scale)
        bar_height = max(2, round(3 * scale))
        canvas.rectangle(Rect(x, bar_y, max_width, bar_height), fill=(*THEME.border, 255))
        if state.total:
            bar_width = max(1, round(max_width * min(1.0, state.completed / state.total)))
            canvas.rectangle(Rect(x, bar_y, bar_width, bar_height), fill=(*THEME.accent, 255))
        # Final aggregate values appear only after matching/metrics finish for this frame.
        if state.state == "READY":
            for line_index, line in enumerate(state.lines[1:3]):
                text(line, 238 + line_index * 18)
        elif state.state == "FAILED":
            text(state.lines[0] if state.lines else "Analysis failed", 240)
        else:
            text("Collecting real results; aggregate metrics pending.", 240)
