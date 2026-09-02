"""Adapter-specific validation and capacity errors."""


class AdapterError(ValueError):
    """Base error for deterministic adapter failures."""


class AdapterConfigurationError(AdapterError):
    """Raised when the adapter policy is malformed."""


class AdapterEventError(AdapterError):
    """Raised when a normalized event is invalid or cannot be applied."""


class AdapterSnapshotError(AdapterError):
    """Raised when current state cannot form a valid Phase 2 snapshot."""


class AdapterLimitError(AdapterError):
    """Raised before a configured bounded resource would be exceeded."""
