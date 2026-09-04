"""Validated central security limits and network configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


PROTOCOL_VERSION = 2
DEFAULT_BIND_ADDRESS = "127.0.0.1"
DEFAULT_PORT = 5010
DEFAULT_ALLOWED_CIDRS = "127.0.0.0/8,::1/128"


class SecurityConfigurationError(ValueError):
    """Raised when security configuration is missing or unsafe."""


def _bounded_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as error:
        raise SecurityConfigurationError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise SecurityConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _bounded_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise SecurityConfigurationError(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise SecurityConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True)
class SecurityLimits:
    """Resource ceilings used before untrusted input is allocated or decoded."""

    max_header_size: int = 256 * 1024
    max_jpeg_size: int = 16 * 1024 * 1024
    max_detections: int = 512
    max_string_length: int = 256
    max_image_width: int = 4096
    max_image_height: int = 4096
    max_image_pixels: int = 16_000_000
    socket_read_timeout: float = 5.0
    listener_timeout: float = 1.0
    listen_backlog: int = 5
    max_archive_members: int = 100_000
    max_archive_member_size: int = 2 * 1024 * 1024 * 1024
    max_archive_size: int = 50 * 1024 * 1024 * 1024
    max_archive_ratio: int = 1_000
    max_metadata_size: int = 1024 * 1024

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "SecurityLimits":
        """Load optional bounded overrides, rejecting invalid weakening attempts."""

        values = os.environ if env is None else env
        return cls(
            max_header_size=_bounded_int(
                values, "UAV_MAX_HEADER_SIZE", cls.max_header_size, 1024, 4 * 1024 * 1024
            ),
            max_jpeg_size=_bounded_int(
                values, "UAV_MAX_JPEG_SIZE", cls.max_jpeg_size, 1024, 128 * 1024 * 1024
            ),
            max_detections=_bounded_int(
                values, "UAV_MAX_DETECTIONS", cls.max_detections, 1, 10_000
            ),
            max_string_length=_bounded_int(
                values, "UAV_MAX_STRING_LENGTH", cls.max_string_length, 16, 4096
            ),
            max_image_width=_bounded_int(
                values, "UAV_MAX_IMAGE_WIDTH", cls.max_image_width, 64, 32_768
            ),
            max_image_height=_bounded_int(
                values, "UAV_MAX_IMAGE_HEIGHT", cls.max_image_height, 64, 32_768
            ),
            max_image_pixels=_bounded_int(
                values, "UAV_MAX_IMAGE_PIXELS", cls.max_image_pixels, 4096, 268_435_456
            ),
            socket_read_timeout=_bounded_float(
                values, "UAV_SOCKET_READ_TIMEOUT", cls.socket_read_timeout, 0.25, 300.0
            ),
            listener_timeout=_bounded_float(
                values, "UAV_LISTENER_TIMEOUT", cls.listener_timeout, 0.1, 60.0
            ),
            listen_backlog=_bounded_int(
                values, "UAV_LISTEN_BACKLOG", cls.listen_backlog, 1, 128
            ),
            max_archive_members=_bounded_int(
                values, "UAV_MAX_ARCHIVE_MEMBERS", cls.max_archive_members, 1, 1_000_000
            ),
            max_archive_member_size=_bounded_int(
                values,
                "UAV_MAX_ARCHIVE_MEMBER_SIZE",
                cls.max_archive_member_size,
                1024,
                16 * 1024 * 1024 * 1024,
            ),
            max_archive_size=_bounded_int(
                values,
                "UAV_MAX_ARCHIVE_SIZE",
                cls.max_archive_size,
                1024,
                500 * 1024 * 1024 * 1024,
            ),
            max_archive_ratio=_bounded_int(
                values, "UAV_MAX_ARCHIVE_RATIO", cls.max_archive_ratio, 2, 10_000
            ),
            max_metadata_size=_bounded_int(
                values, "UAV_MAX_METADATA_SIZE", cls.max_metadata_size, 1024, 64 * 1024 * 1024
            ),
        )
