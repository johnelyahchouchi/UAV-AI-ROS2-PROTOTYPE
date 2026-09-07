"""Frame processing, performance tracking, preview loop, and screenshots."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Any, Callable

from .configuration import choose_preview_position
from .domain import (
    CaptureRegion,
    DetectorProtocol,
    FrameMetrics,
    FrameOutcome,
    RendererProtocol,
    TesterError,
    VideoSourceEnded,
    WINDOW_NAME,
    filter_detections,
)
from .sources import MSSScreenSource, VideoFileSource, scaled_preview_size
from .uncertainty_adapter import InspectorProtocol, MethodSelectorProtocol


class PerformanceTracker:
    """Track rolling capture, inference, total time, and displayed FPS."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise TesterError("Metrics window size must be positive")
        self._capture_ms: deque[float] = deque(maxlen=window_size)
        self._inference_ms: deque[float] = deque(maxlen=window_size)
        self._total_ms: deque[float] = deque(maxlen=window_size)
        self._frame_intervals: deque[float] = deque(maxlen=window_size)
        self._last_frame_timestamp: float | None = None

    def record(
        self,
        *,
        capture_ms: float,
        inference_ms: float,
        total_ms: float,
        frame_timestamp: float,
    ) -> FrameMetrics:
        values = (capture_ms, inference_ms, total_ms, frame_timestamp)
        if not all(math.isfinite(float(value)) for value in values):
            raise TesterError("Frame timings must be finite")
        if capture_ms < 0 or inference_ms < 0 or total_ms < 0:
            raise TesterError("Frame timings cannot be negative")
        if self._last_frame_timestamp is not None:
            interval = frame_timestamp - self._last_frame_timestamp
            if interval > 0:
                self._frame_intervals.append(interval)
        self._last_frame_timestamp = frame_timestamp
        self._capture_ms.append(float(capture_ms))
        self._inference_ms.append(float(inference_ms))
        self._total_ms.append(float(total_ms))
        return self.snapshot()

    def snapshot(self) -> FrameMetrics:
        """Return current rolling means."""

        def average(values: deque[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        average_interval = average(self._frame_intervals)
        return FrameMetrics(
            fps=(1.0 / average_interval) if average_interval > 0 else 0.0,
            capture_ms=average(self._capture_ms),
            inference_ms=average(self._inference_ms),
            total_ms=average(self._total_ms),
        )

    def reset_interval(self) -> None:
        """Exclude a pause or blocking inspection from the next FPS interval."""

        self._last_frame_timestamp = None


class FrameProcessor:
    """Coordinate one-pass inference, filtering, metrics, and rendering."""

    def __init__(
        self,
        detector: DetectorProtocol,
        renderer: RendererProtocol,
        *,
        allowed_classes: frozenset[str] | None,
        model_name: str,
        region: CaptureRegion,
        device: str,
        source_label: str | None = None,
        uncertainty_available: bool = True,
        mcdo_v2_available: bool = False,
        tracker: PerformanceTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.detector = detector
        self.renderer = renderer
        self.allowed_classes = allowed_classes
        self.model_name = model_name
        self.region = region
        self.device = device
        self.source_label = source_label
        self.uncertainty_available = uncertainty_available
        self.mcdo_v2_available = mcdo_v2_available
        self.tracker = tracker or PerformanceTracker()
        self.clock = clock
        self.last_inspection: str | None = None

    def process(
        self,
        frame: Any,
        *,
        capture_ms: float,
        frame_started_at: float,
        frame_timestamp: float,
        hud_visible: bool,
    ) -> FrameOutcome:
        inference_started = self.clock()
        detections = self.detector.detect(frame)
        inference_ms = (self.clock() - inference_started) * 1000.0
        displayed = filter_detections(detections, self.allowed_classes)
        annotated = self.renderer.draw_detections(frame, displayed)
        total_ms = (self.clock() - frame_started_at) * 1000.0
        metrics = self.tracker.record(
            capture_ms=capture_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
            frame_timestamp=frame_timestamp,
        )
        if hud_visible:
            self.renderer.draw_hud(
                annotated,
                metrics=metrics,
                model_name=self.model_name,
                region=self.region,
                device=self.device,
                detection_count=len(displayed),
                source_label=self.source_label,
                last_inspection=self.last_inspection,
                uncertainty_available=self.uncertainty_available,
                mcdo_v2_available=self.mcdo_v2_available,
            )
        return FrameOutcome(annotated, displayed, metrics)


class ScreenshotStore:
    """Create unique timestamped screenshots without overwriting output."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = Path(output_directory).expanduser()

    def next_path(self, when: datetime | None = None) -> Path:
        timestamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
        base = self.output_directory / f"live_screen_{timestamp}.png"
        candidate = base
        counter = 1
        while candidate.exists():
            candidate = base.with_name(f"{base.stem}_{counter}{base.suffix}")
            counter += 1
        return candidate

    def save(self, frame: Any, cv2_module: Any) -> Path:
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TesterError(
                f"Could not create screenshot directory: {self.output_directory}"
            ) from error
        path = self.next_path()
        try:
            saved = bool(cv2_module.imwrite(str(path), frame))
        except Exception as error:
            raise TesterError(f"Could not save screenshot: {error}") from error
        if not saved:
            raise TesterError(f"Could not save screenshot: {path}")
        return path


def run_test_frame(
    frame_path: Path,
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    cv2_module: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Run one image through loading, inference, rendering, and save."""

    try:
        resolved = frame_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise TesterError(f"Test frame does not exist: {frame_path}") from error
    frame = cv2_module.imread(str(resolved))
    if frame is None:
        raise TesterError(f"OpenCV could not decode test frame: {resolved}")
    started = clock()
    outcome = processor.process(
        frame,
        capture_ms=0.0,
        frame_started_at=started,
        frame_timestamp=started,
        hud_visible=True,
    )
    return screenshot_store.save(outcome.annotated_frame, cv2_module)


def _perform_uncertainty_inspection(
    inspector: InspectorProtocol,
    frozen: Any,
    *,
    method_label: str,
    failure_status: str,
    processor: FrameProcessor,
    cv2_module: Any,
) -> Any | None:
    """Run one blocking on-demand method while keeping the live session recoverable."""

    working = inspector.render_working(frozen)
    cv2_module.imshow(WINDOW_NAME, working)
    cv2_module.waitKey(1)
    print(f"Running {method_label} on the frozen frame...")
    try:
        inspection = inspector.inspect(frozen)
    except Exception as error:
        processor.last_inspection = failure_status
        print(f"{method_label} failed: {error}")
        return None
    processor.last_inspection = inspection.status
    print(f"Uncertainty inspection complete: {inspection.status}")
    return inspection.frame


def _run_preview_loop(
    source: Any,
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    target_period: float,
    cv2_module: Any,
    uncertainty_inspector: MethodSelectorProtocol | None,
    mcdo_v2_inspector: InspectorProtocol | None,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    paused = False
    hud_visible = True
    last_raw_frame: Any | None = None
    last_frame: Any | None = None
    inspection_frame: Any | None = None
    selection_frame: Any | None = None
    selection_frozen: Any | None = None
    last_detection_count = 0

    with source:
        while True:
            loop_started = clock()
            if not paused:
                capture_started = clock()
                try:
                    frame = source.read()
                except VideoSourceEnded:
                    print("Video reached the end; closing preview.")
                    break
                capture_ms = (clock() - capture_started) * 1000.0
                last_raw_frame = frame.copy()
                outcome = processor.process(
                    frame,
                    capture_ms=capture_ms,
                    frame_started_at=loop_started,
                    frame_timestamp=loop_started,
                    hud_visible=hud_visible,
                )
                last_frame = outcome.annotated_frame
                last_detection_count = len(outcome.detections)

            if selection_frame is not None:
                display_frame = selection_frame.copy()
            elif inspection_frame is not None:
                display_frame = inspection_frame.copy()
            elif last_frame is not None:
                display_frame = last_frame.copy()
                if paused and hud_visible:
                    processor.renderer.draw_hud(
                        display_frame,
                        metrics=processor.tracker.snapshot(),
                        model_name=processor.model_name,
                        region=processor.region,
                        device=processor.device,
                        detection_count=last_detection_count,
                        source_label=processor.source_label,
                        paused=True,
                        last_inspection=processor.last_inspection,
                        uncertainty_available=getattr(
                            processor,
                            "uncertainty_available",
                            uncertainty_inspector is not None,
                        ),
                        mcdo_v2_available=getattr(
                            processor,
                            "mcdo_v2_available",
                            mcdo_v2_inspector is not None,
                        ),
                    )
            else:
                display_frame = None
            if display_frame is not None:
                cv2_module.imshow(WINDOW_NAME, display_frame)

            key = cv2_module.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if selection_frame is not None:
                if key in (27, 32, ord("u"), ord("U"), ord("p"), ord("P")):
                    selection_frame = None
                    selection_frozen = None
                    paused = False
                    processor.tracker.reset_interval()
                    print("Uncertainty selection cancelled; resumed")
                elif key in (ord("1"), ord("2")) and selection_frozen is not None:
                    if key == ord("1"):
                        selected = uncertainty_inspector
                        method_label = "V1 input-perturbation robustness"
                        failure_status = "V1 INSPECTION FAILED"
                    else:
                        selected = mcdo_v2_inspector
                        method_label = "V2 MC Dropout model uncertainty"
                        failure_status = "V2 INSPECTION FAILED"
                    frozen = selection_frozen.copy()
                    selection_frame = None
                    selection_frozen = None
                    if selected is None:
                        processor.last_inspection = failure_status
                        print(f"{method_label} is unavailable")
                    else:
                        inspection_frame = _perform_uncertainty_inspection(
                            selected,
                            frozen,
                            method_label=method_label,
                            failure_status=failure_status,
                            processor=processor,
                            cv2_module=cv2_module,
                        )
                elif key in (ord("h"), ord("H")):
                    hud_visible = not hud_visible
                    print("HUD on" if hud_visible else "HUD off")
                elif key in (ord("s"), ord("S")):
                    saved_path = screenshot_store.save(selection_frame, cv2_module)
                    print(f"Screenshot saved: {saved_path}")
            elif key == 27:
                break
            elif key in (ord("p"), ord("P"), 32):
                if inspection_frame is not None:
                    inspection_frame = None
                    paused = False
                    processor.tracker.reset_interval()
                    print("Resumed after uncertainty inspection")
                elif key != 32:
                    paused = not paused
                    if not paused:
                        processor.tracker.reset_interval()
                    print("Paused" if paused else "Resumed")
            elif key in (ord("u"), ord("U")):
                if inspection_frame is not None:
                    inspection_frame = None
                    paused = False
                    processor.tracker.reset_interval()
                    print("Resumed after uncertainty inspection")
                elif uncertainty_inspector is None:
                    print("Uncertainty inspection is disabled.")
                elif last_raw_frame is None:
                    print("No frame is available to inspect yet.")
                else:
                    paused = True
                    frozen = last_raw_frame.copy()
                    if mcdo_v2_inspector is not None:
                        v2_samples = int(
                            getattr(
                                getattr(mcdo_v2_inspector, "config", None),
                                "sample_count",
                                20,
                            )
                        )
                        selection_frozen = frozen
                        selection_frame = uncertainty_inspector.render_selection(
                            frozen, v2_sample_count=v2_samples
                        )
                        print(
                            "Frozen-frame uncertainty menu: 1 = V1 input robustness, "
                            "2 = V2 MC Dropout, ESC/Space = resume"
                        )
                    else:
                        print(
                            "V2 unavailable - configure UAV_MCDO_V2_MODEL_PATH; "
                            "running V1 directly"
                        )
                        inspection_frame = _perform_uncertainty_inspection(
                            uncertainty_inspector,
                            frozen,
                            method_label="V1 input-perturbation robustness",
                            failure_status="INSPECTION FAILED",
                            processor=processor,
                            cv2_module=cv2_module,
                        )
            elif key in (ord("h"), ord("H")):
                hud_visible = not hud_visible
                print("HUD on" if hud_visible else "HUD off")
            elif key in (ord("s"), ord("S")):
                frame_to_save = inspection_frame if inspection_frame is not None else last_frame
                if frame_to_save is None:
                    print("No frame is available to save yet.")
                else:
                    saved_path = screenshot_store.save(frame_to_save, cv2_module)
                    print(f"Screenshot saved: {saved_path}")

            try:
                if cv2_module.getWindowProperty(WINDOW_NAME, cv2_module.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass
            if target_period > 0 and not paused:
                delay = target_period - (clock() - loop_started)
                if delay > 0:
                    sleeper(delay)


def run_live_preview(
    region: CaptureRegion,
    monitors: tuple[CaptureRegion, ...] | list[CaptureRegion],
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    max_fps: float,
    cv2_module: Any,
    uncertainty_inspector: MethodSelectorProtocol | None = None,
    mcdo_v2_inspector: InspectorProtocol | None = None,
    source_factory: Callable[[CaptureRegion], Any] = MSSScreenSource,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Run a keyboard-controlled screen capture preview."""

    try:
        preview_width, preview_height = scaled_preview_size(region)
        window_left, window_top, overlap = choose_preview_position(
            region, monitors, preview_width, preview_height
        )
        cv2_module.namedWindow(WINDOW_NAME, cv2_module.WINDOW_NORMAL)
        cv2_module.resizeWindow(WINDOW_NAME, preview_width, preview_height)
        cv2_module.moveWindow(WINDOW_NAME, window_left, window_top)
        if overlap:
            print(
                "Warning: no non-overlapping preview position is available; "
                "move the preview to another monitor if possible."
            )
        _run_preview_loop(
            source_factory(region),
            processor,
            screenshot_store,
            target_period=(1.0 / max_fps) if max_fps > 0 else 0.0,
            cv2_module=cv2_module,
            uncertainty_inspector=uncertainty_inspector,
            mcdo_v2_inspector=mcdo_v2_inspector,
            clock=clock,
            sleeper=sleeper,
        )
    except KeyboardInterrupt:
        print("Interrupted; closing preview.")
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"OpenCV preview failed: {error}") from error
    finally:
        try:
            cv2_module.destroyAllWindows()
        except Exception:
            pass


def run_video_preview(
    source: VideoFileSource,
    processor: FrameProcessor,
    screenshot_store: ScreenshotStore,
    *,
    max_fps: float,
    cv2_module: Any,
    uncertainty_inspector: MethodSelectorProtocol | None = None,
    mcdo_v2_inspector: InspectorProtocol | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Play a local video through one-pass detection until EOF or exit."""

    preview_width, preview_height = scaled_preview_size(processor.region)
    try:
        cv2_module.namedWindow(WINDOW_NAME, cv2_module.WINDOW_NORMAL)
        cv2_module.resizeWindow(WINDOW_NAME, preview_width, preview_height)
        playback_fps = source.metadata.fps
        if max_fps > 0:
            playback_fps = min(playback_fps, max_fps)
        _run_preview_loop(
            source,
            processor,
            screenshot_store,
            target_period=1.0 / playback_fps,
            cv2_module=cv2_module,
            uncertainty_inspector=uncertainty_inspector,
            mcdo_v2_inspector=mcdo_v2_inspector,
            clock=clock,
            sleeper=sleeper,
        )
    except KeyboardInterrupt:
        print("Interrupted; closing preview.")
    except TesterError:
        raise
    except Exception as error:
        raise TesterError(f"OpenCV video preview failed: {error}") from error
    finally:
        try:
            cv2_module.destroyAllWindows()
        except Exception:
            pass
