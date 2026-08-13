"""User-facing dashboard errors."""

from __future__ import annotations


class DashboardError(RuntimeError):
    """A validated dashboard failure with optional recovery guidance."""

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
        """Return a concise message suitable for the local UI."""
        parts = [self.message]
        if self.recovery:
            parts.append(self.recovery)
        return " ".join(parts)


class ProcessingCancelled(DashboardError):
    """Cooperative cancellation observed between inference calls."""

    def __init__(self) -> None:
        super().__init__(
            "PROCESSING_CANCELLED",
            "Processing cancelled. Partial output was removed safely.",
        )
