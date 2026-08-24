"""Display-only formatting and explanatory text for dashboard results."""

from __future__ import annotations

from math import isfinite
from statistics import fmean
from typing import Sequence


DISPLAY_DECIMALS = 3
NOT_AVAILABLE = "N/A"
FRAME_LOCAL_TARGET_NOTE = (
    "Target IDs are local to each sampled frame. They are not cross-frame tracking IDs."
)


def format_metric(value: object | None) -> str:
    """Format one visible metric without modifying its stored value."""
    if value is None:
        return NOT_AVAILABLE
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if not isfinite(number):
        return NOT_AVAILABLE
    return f"{number:.{DISPLAY_DECIMALS}f}"


def mean_metric(targets: Sequence[dict[str, object]], key: str) -> float | None:
    """Return an unrounded mean, or None when no target clusters exist."""
    if not targets:
        return None
    return fmean(float(target[key]) for target in targets)


def target_id_scope_note(is_video: bool) -> str:
    """Return the frame-local target-ID warning only for video results."""
    if not is_video:
        return ""
    return f"> **Frame-local IDs:** {FRAME_LOCAL_TARGET_NOTE}"


def entropy_status_markdown(
    summary: dict[str, object],
    *,
    is_video: bool = False,
) -> str:
    """Explain empty and all-zero entropy views without changing raw entropy."""
    targets = list(summary.get("targets", []))
    if not targets:
        return (
            "**Class entropy: N/A**  \n"
            "No detected targets to calculate class entropy."
        )
    entropies = [float(target["class_entropy_bits"]) for target in targets]
    if all(value == 0.0 for value in entropies):
        scope = "sampled frame" if is_video else "image"
        return (
            "**Class entropy: 0.000 for all targets**  \n"
            f"No class disagreement observed in this {scope}."
        )
    return ""


def review_flags_markdown(events: Sequence[str]) -> str:
    """Present transparent diagnostic events as a prominent review list."""
    if not events:
        return "### Review flags\n\nNo dashboard diagnostic rule was triggered."

    category_rules = (
        ("no target clusters", "No detections"),
        ("appears only under perturbation", "Perturbation-only detection"),
        ("missing in", "Missing detections"),
        ("class disagreement", "Class disagreement"),
        ("overlapping/alternative", "Overlap diagnostic"),
        ("box variation", "Bounding-box variation"),
    )
    lines = ["### Review flags", "", "Observed diagnostic events for review; these are not automatic failures:", ""]
    for event in events:
        normalized = event.lower()
        category = next(
            (label for phrase, label in category_rules if phrase in normalized),
            "Observed instability",
        )
        lines.append(f"- **{category}** — {event}")
    return "\n".join(lines)


def display_overlap_rows(rows: Sequence[Sequence[object]]) -> list[list[object]]:
    """Round only the visible numeric cells of overlap diagnostics."""
    displayed: list[list[object]] = []
    for row in rows:
        values = list(row)
        for index in range(4, 9):
            values[index] = format_metric(values[index])
        displayed.append(values)
    return displayed


def display_comparison_rows(rows: Sequence[Sequence[object]]) -> list[list[object]]:
    """Round comparison-table metrics while leaving export rows untouched."""
    displayed: list[list[object]] = []
    for row in rows:
        values = list(row)
        for index in (8, 9, 11, 12, 13, 14, 15):
            values[index] = format_metric(values[index])
        displayed.append(values)
    return displayed
