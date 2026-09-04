"""Pure validation helpers for sender CLI and control-panel boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math


class InputValidationError(ValueError):
    """Raised when a launch argument falls outside its accepted domain."""


def validate_ip(value: object) -> str:
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError as error:
        raise InputValidationError("Target must be a valid IPv4 or IPv6 address") from error


def validate_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise InputValidationError(f"{name} must be an integer")
    try:
        parsed = int(str(value).strip(), 10)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise InputValidationError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def validate_finite_float(
    value: object, name: str, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name} must be a number") from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise InputValidationError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def validate_binary_flag(value: object, name: str) -> int:
    if value in (0, "0"):
        return 0
    if value in (1, "1"):
        return 1
    raise InputValidationError(f"{name} must be 0 or 1")


@dataclass(frozen=True)
class SenderSettings:
    target: str
    port: int
    confidence: float
    iou: float
    image_size: int
    stride: int
    send_width: int
    show: int
    military_only: int


def validate_sender_settings(
    *,
    target: object,
    port: object,
    confidence: object,
    iou: object,
    image_size: object,
    stride: object,
    send_width: object,
    show: object,
    military_only: object,
) -> SenderSettings:
    """Validate every scalar value before it reaches socket or subprocess code."""

    return SenderSettings(
        target=validate_ip(target),
        port=validate_integer(port, "Port", 1, 65_535),
        confidence=validate_finite_float(confidence, "Confidence", 0.0, 1.0),
        iou=validate_finite_float(iou, "IoU", 0.0, 1.0),
        image_size=validate_integer(image_size, "Image size", 32, 4096),
        stride=validate_integer(stride, "Stride", 1, 10_000),
        send_width=validate_integer(send_width, "Send width", 32, 4096),
        show=validate_binary_flag(show, "Show"),
        military_only=validate_binary_flag(military_only, "Military-only"),
    )
