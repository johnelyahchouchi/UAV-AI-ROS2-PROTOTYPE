"""Strict external video-source classification and resolution."""

from __future__ import annotations

import subprocess
from typing import Callable
from urllib.parse import urlparse


YOUTUBE_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
)


class SourceResolutionError(RuntimeError):
    """Raised when an approved external source cannot be resolved safely."""


def is_trusted_youtube_url(source: str) -> bool:
    """Return true only for HTTP(S) URLs on an exact approved YouTube host."""

    try:
        parsed = urlparse(source)
        host = parsed.hostname.lower() if parsed.hostname else ""
    except (AttributeError, ValueError):
        return False
    return parsed.scheme.lower() in {"http", "https"} and host in YOUTUBE_HOSTS


def source_log_label(source: str) -> str:
    """Describe a source without logging URL credentials, query tokens, or full paths."""

    try:
        parsed = urlparse(source)
        if parsed.hostname and parsed.scheme:
            return f"{parsed.scheme.lower()}://{parsed.hostname}/..."
    except (AttributeError, ValueError):
        return "configured source"
    normalized = str(source).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else "configured source"


def resolve_video_source(
    source: str,
    *,
    timeout: float = 30.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resolve an approved YouTube URL through yt-dlp or return a local source."""

    if not is_trusted_youtube_url(source):
        return source
    try:
        completed = runner(
            ["yt-dlp", "-g", "-f", "best", "--", source],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise SourceResolutionError("yt-dlp timed out while resolving the source") from error
    except subprocess.CalledProcessError as error:
        raise SourceResolutionError("yt-dlp could not resolve the source") from error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise SourceResolutionError("yt-dlp returned no playable stream URL")
    return lines[0]
