"""Single-job coordination and cooperative cancellation."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid

from .errors import DashboardError, ProcessingCancelled


@dataclass
class CancellationToken:
    """Cancellation state for one image, video, or batch job."""

    job_id: str
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise ProcessingCancelled()


class ProcessingController:
    """Permit one GPU experiment job at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: CancellationToken | None = None

    def begin(self) -> CancellationToken:
        """Start a job or reject parallel processing."""
        with self._lock:
            if self._active is not None:
                raise DashboardError(
                    "PROCESSING_ALREADY_RUNNING",
                    "Another uncertainty experiment is already running.",
                )
            self._active = CancellationToken(uuid.uuid4().hex)
            return self._active

    def finish(self, token: CancellationToken) -> None:
        """Clear the token if it still belongs to the active job."""
        with self._lock:
            if self._active is token:
                self._active = None

    def request_cancel(self) -> bool:
        """Request cancellation between inference calls."""
        with self._lock:
            if self._active is None:
                return False
            self._active.cancel()
            return True

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active is not None
