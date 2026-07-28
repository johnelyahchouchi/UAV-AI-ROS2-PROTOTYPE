"""Thread-safe cooperative cancellation and single-job coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid

from .errors import DashboardError, ProcessingCancelled


@dataclass
class CancellationToken:
    """Cancellation state associated with one processing job."""

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
    """Allow one active job and expose a safe cancellation request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: CancellationToken | None = None

    def begin(self) -> CancellationToken:
        """Create the active token or reject a parallel job."""
        with self._lock:
            if self._active is not None:
                raise DashboardError(
                    "PROCESSING_ALREADY_RUNNING",
                    "Another video is already being processed.",
                    recovery="Cancel it or wait for it to finish.",
                )
            self._active = CancellationToken(job_id=uuid.uuid4().hex)
            return self._active

    def finish(self, token: CancellationToken) -> None:
        """Clear the active token only when it belongs to the caller."""
        with self._lock:
            if self._active is token:
                self._active = None

    def request_cancel(self) -> bool:
        """Request cooperative cancellation of the active job."""
        with self._lock:
            if self._active is None:
                return False
            self._active.cancel()
            return True

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active is not None
