"""User-facing error types for the model test dashboard."""

from __future__ import annotations


class DashboardError(RuntimeError):
    """An expected failure with a stable code and recovery guidance."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        recovery: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery
        self.detail = detail

    def user_message(self) -> str:
        """Return a concise message suitable for the dashboard."""
        parts = [f"[{self.code}] {self.message}"]
        if self.recovery:
            parts.append(f"Action: {self.recovery}")
        return "\n\n".join(parts)


class ProcessingCancelled(DashboardError):
    """Raised when a user cooperatively cancels an active job."""

    def __init__(self) -> None:
        super().__init__(
            "PROCESSING_CANCELLED",
            "Processing was cancelled safely.",
            recovery="Choose settings and press Start Processing to run again.",
        )
