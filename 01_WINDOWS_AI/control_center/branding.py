"""Editable branding configuration for the desktop control center."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


CONTROL_CENTER_ROOT = Path(__file__).resolve().parent
BRANDING_CONFIG_PATH = CONTROL_CENTER_ROOT / "branding.json"


@dataclass(frozen=True)
class Branding:
    """Text and logo shown in the application header."""

    title: str = "UAV Perception and Mission Control"
    subtitle: str = "Research Prototype Dashboard"
    author: str = "JOHN EL YAHCHOUCHI"
    internship_period: str = "June - September 2026 internship"
    logo_path: Path = CONTROL_CENTER_ROOT / "assets" / "additess_logo_dark.png"
    logo_max_width: int = 380
    logo_max_height: int = 70


def _text_value(payload: Mapping[str, Any], name: str, default: str) -> str:
    value = str(payload.get(name, default)).strip()
    return (value or default).replace(chr(0x2014), "-")


def _integer_value(
    payload: Mapping[str, Any],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(payload.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def load_branding(path: Path = BRANDING_CONFIG_PATH) -> Branding:
    """Load editable branding, falling back safely when the file is invalid."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    defaults = Branding()
    logo_value = str(payload.get("logo_path", "assets/additess_logo_dark.png")).strip()
    logo_path = Path(logo_value).expanduser() if logo_value else defaults.logo_path
    if not logo_path.is_absolute():
        logo_path = path.parent / logo_path

    return Branding(
        title=_text_value(payload, "title", defaults.title),
        subtitle=_text_value(payload, "subtitle", defaults.subtitle),
        author=_text_value(payload, "author", defaults.author),
        internship_period=_text_value(
            payload, "internship_period", defaults.internship_period
        ),
        logo_path=logo_path.resolve(),
        logo_max_width=_integer_value(
            payload, "logo_max_width", defaults.logo_max_width, 120, 600
        ),
        logo_max_height=_integer_value(
            payload, "logo_max_height", defaults.logo_max_height, 30, 160
        ),
    )
