"""Explicit uncertainty-method capabilities.

The validated V2 runtime is implemented, but it is enabled only after an external
trusted checkpoint has been loaded and its architecture has passed validation.
Repeating ordinary deterministic YOLO inference is never treated as MC Dropout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MCDropoutCapability:
    """Runtime readiness and scientific requirements for V2."""

    runtime_implemented: bool
    available: bool
    method: str
    reason: str
    requirements: tuple[str, ...]


class MCDropoutInspector(Protocol):
    """Contract implemented by a validated, checkpoint-backed V2 inspector."""

    def inspect(self, exact_frame: Any) -> object:
        """Run repeated stochastic passes on one unchanged frame."""


def mc_dropout_capability(
    *, validated_checkpoint_loaded: bool = False
) -> MCDropoutCapability:
    """Report runtime availability without probing or deserializing a checkpoint."""

    return MCDropoutCapability(
        runtime_implemented=True,
        available=validated_checkpoint_loaded,
        method="v2_mc_dropout_model_uncertainty",
        reason=(
            "Validated external V2 checkpoint loaded and architecture accepted"
            if validated_checkpoint_loaded
            else "Runtime is implemented; configure and trust the validated external "
            "V2 checkpoint before enabling it"
        ),
        requirements=(
            "keep the model globally in evaluation mode",
            "keep every BatchNorm layer in evaluation mode",
            "activate only forward-connected Dropout2d layers",
            "repeat inference on the same exact frame",
            "validate the checkpoint and stochastic execution path",
        ),
    )
