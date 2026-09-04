"""Thread-safe model caching and inference-device selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Iterator

import torch
from ultralytics import YOLO

from .configuration import DeviceChoice
from .errors import DashboardError
from uav_security.model_integrity import ModelIntegrityError, verify_trusted_model


@dataclass(frozen=True)
class ModelIdentity:
    """File identity used to invalidate a cached model safely."""

    canonical_path: Path
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved user choice and exact Ultralytics device argument."""

    choice: DeviceChoice
    argument: int | str
    display_name: str


@dataclass(frozen=True)
class ModelHandle:
    """A loaded model and whether this call populated the cache."""

    model: Any
    identity: ModelIdentity
    loaded_now: bool


class ModelManager:
    """Cache one unchanged model and serialize mutable inference state."""

    def __init__(
        self,
        loader: Callable[[str], Any] = YOLO,
        torch_module: Any = torch,
        verifier: Callable[[Path], str] = verify_trusted_model,
    ) -> None:
        self._loader = loader
        self._torch = torch_module
        self._verifier = verifier
        self._cache_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._cached_identity: ModelIdentity | None = None
        self._cached_model: Any | None = None

    @staticmethod
    def identity_for(model_path: Path) -> ModelIdentity:
        """Return canonical path, size, and modification time."""
        try:
            canonical = model_path.expanduser().resolve(strict=True)
            stat = canonical.stat()
        except OSError as error:
            raise DashboardError(
                "MODEL_NOT_FOUND",
                f"Cannot access the selected model: {model_path}",
                detail=str(error),
            ) from error
        return ModelIdentity(
            canonical_path=canonical,
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )

    def get_model(self, model_path: Path) -> ModelHandle:
        """Load a changed model once or return the cached instance."""
        identity = self.identity_for(model_path)
        with self._cache_lock:
            if identity == self._cached_identity and self._cached_model is not None:
                return ModelHandle(self._cached_model, identity, loaded_now=False)

            try:
                self._verifier(identity.canonical_path)
                model = self._loader(str(identity.canonical_path))
            except ModelIntegrityError as error:
                raise DashboardError(
                    "MODEL_INTEGRITY_FAILED",
                    str(error),
                    recovery="Add an independently verified SHA-256 to the trusted registry.",
                ) from error
            except Exception as error:
                raise DashboardError(
                    "MODEL_LOAD_FAILED",
                    f"Ultralytics could not load {identity.canonical_path.name}.",
                    recovery="Select a valid detection-model .pt file.",
                    detail=str(error),
                ) from error

            task = getattr(model, "task", None)
            if task not in (None, "detect"):
                raise DashboardError(
                    "MODEL_TASK_UNSUPPORTED",
                    f"The selected model task is {task!r}; this dashboard requires a detection model.",
                    recovery="Select an Ultralytics detection checkpoint.",
                )

            self._cached_identity = identity
            self._cached_model = model
            return ModelHandle(model, identity, loaded_now=True)

    def resolve_device(self, choice: DeviceChoice) -> DeviceInfo:
        """Resolve Auto, GPU 0, or CPU without silent explicit fallback."""
        if choice is DeviceChoice.CPU:
            return DeviceInfo(choice, "cpu", "CPU")

        try:
            cuda_available = bool(self._torch.cuda.is_available())
            device_count = int(self._torch.cuda.device_count())
        except Exception as error:
            raise DashboardError(
                "GPU_QUERY_FAILED",
                "PyTorch could not query CUDA device availability.",
                detail=str(error),
            ) from error

        if cuda_available and device_count > 0:
            try:
                gpu_name = str(self._torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = "CUDA device 0"
            return DeviceInfo(choice, 0, f"GPU 0 — {gpu_name}")

        if choice is DeviceChoice.GPU_0:
            raise DashboardError(
                "GPU_0_UNAVAILABLE",
                "GPU 0 was selected, but CUDA device 0 is not available.",
                recovery=(
                    "Run the dashboard with the verified CUDA-enabled UAV_YOLO_ENV "
                    "or explicitly select CPU."
                ),
            )
        return DeviceInfo(choice, "cpu", "CPU — CUDA unavailable")

    def describe_device(self, choice: DeviceChoice) -> str:
        """Return a user-facing resolved-device description."""
        info = self.resolve_device(choice)
        return f"**Resolved inference device:** {info.display_name}"

    @staticmethod
    def reset_inference_state(model: Any) -> None:
        """Prevent tracker/predictor state from leaking between unrelated videos."""
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
        if hasattr(model, "predictor"):
            model.predictor = None

    @contextmanager
    def acquire(self, model_path: Path) -> Iterator[ModelHandle]:
        """Lock model inference for the duration of one video."""
        with self._run_lock:
            handle = self.get_model(model_path)
            self.reset_inference_state(handle.model)
            try:
                yield handle
            finally:
                self.reset_inference_state(handle.model)
