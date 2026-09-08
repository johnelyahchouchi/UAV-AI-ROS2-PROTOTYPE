"""Shared high-quality presentation primitives for uncertainty inspection.

The scientific adapters provide the values.  This module only controls how
those values are laid out and drawn on a final-resolution presentation frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


Color = tuple[int, int, int]


def _ascii_fallback(value: str) -> str:
    """Keep the last-resort bitmap font readable when Unicode is unavailable."""

    replacements = {
        "—": "-",
        "–": "-",
        "·": "|",
        "σ": "std",
        "±": "+/-",
        "…": "...",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    return value.encode("ascii", errors="replace").decode("ascii")


@dataclass(frozen=True)
class PresentationTheme:
    """Restrained, projection-friendly colors in Pillow RGB order."""

    background: Color = (9, 15, 25)
    surface: Color = (17, 27, 43)
    surface_raised: Color = (24, 37, 57)
    border: Color = (57, 74, 99)
    text: Color = (238, 243, 250)
    muted: Color = (160, 174, 194)
    accent: Color = (65, 196, 224)
    stable: Color = (62, 203, 139)
    review: Color = (245, 181, 66)
    failure: Color = (238, 91, 91)


THEME = PresentationTheme()


@dataclass(frozen=True)
class Rect:
    """Integer rectangle used by the responsive layout and its tests."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )


@dataclass(frozen=True)
class PresentationLayout:
    """Responsive 16:9 split layout for a single rendered page."""

    width: int
    height: int
    scale: float
    margin: int
    header: Rect
    image: Rect
    summary: Rect
    cards: tuple[Rect, ...]
    footer: Rect
    cards_per_page: int


@dataclass(frozen=True)
class FramePlacement:
    """Coordinates and scale of the uncropped frozen frame within its panel."""

    x: int
    y: int
    width: int
    height: int
    scale: float


def presentation_canvas_size(frame_shape: Sequence[int]) -> tuple[int, int]:
    """Select one of the presentation-tested 16:9 output resolutions."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")
    if width >= 1800 or height >= 1000:
        return (1920, 1080)
    if width >= 1450 or height >= 820:
        return (1600, 900)
    return (1280, 720)


def calculate_presentation_layout(
    width: int, height: int, visible_card_count: int
) -> PresentationLayout:
    """Calculate a 60/40 frame-and-analytics layout without card overlap."""

    if width <= 0 or height <= 0:
        raise ValueError("presentation dimensions must be positive")
    if visible_card_count < 0:
        raise ValueError("visible_card_count cannot be negative")
    scale = height / 720.0
    margin = max(14, round(16 * scale))
    gap = max(10, round(12 * scale))
    header_height = max(58, round(62 * scale))
    footer_height = max(32, round(34 * scale))
    content_y = margin + header_height
    content_bottom = height - margin - footer_height
    content_height = content_bottom - content_y
    available_width = width - 2 * margin - gap
    image_width = round(available_width * 0.59)
    panel_width = available_width - image_width
    summary_height = min(round(132 * scale), max(104, content_height // 3))
    image = Rect(margin, content_y, image_width, content_height)
    panel_x = image.right + gap
    summary = Rect(panel_x, content_y, panel_width, summary_height)
    cards_y = summary.bottom + gap
    cards_height = content_bottom - cards_y
    cards_per_page = 2
    slot_count = max(1, min(cards_per_page, visible_card_count or 1))
    card_height = (cards_height - gap * (slot_count - 1)) // slot_count
    cards = tuple(
        Rect(panel_x, cards_y + index * (card_height + gap), panel_width, card_height)
        for index in range(slot_count)
    )
    return PresentationLayout(
        width=width,
        height=height,
        scale=scale,
        margin=margin,
        header=Rect(margin, margin, width - 2 * margin, header_height),
        image=image,
        summary=summary,
        cards=cards,
        footer=Rect(margin, content_bottom, width - 2 * margin, footer_height),
        cards_per_page=cards_per_page,
    )


def _font_candidates(*, bold: bool) -> tuple[str, ...]:
    """Return platform-aware candidates without embedding a user-specific path."""

    filenames = (
        ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    )
    candidates: list[str] = []
    if sys.platform.startswith("win"):
        windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
        candidates.extend(str(windows_root / "Fonts" / name) for name in filenames)
    elif sys.platform == "darwin":
        mac_names = ("Helvetica.ttc", "Arial.ttf")
        candidates.extend(str(Path("/System/Library/Fonts") / name) for name in mac_names)
        candidates.extend(str(Path("/Library/Fonts") / name) for name in mac_names)
    else:
        linux_roots = (
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/liberation2"),
        )
        candidates.extend(str(root / name) for root in linux_roots for name in filenames)
    # Pillow can resolve its packaged DejaVu fonts by family filename.
    candidates.extend(filenames)
    return tuple(dict.fromkeys(candidates))


class FontResolver:
    """Resolve TrueType UI fonts dynamically, with a safe Pillow fallback."""

    def __init__(
        self,
        *,
        regular_candidates: Sequence[str] | None = None,
        bold_candidates: Sequence[str] | None = None,
        truetype_loader: Callable[..., Any] = ImageFont.truetype,
        default_loader: Callable[..., Any] = ImageFont.load_default,
    ) -> None:
        self.regular_candidates = tuple(regular_candidates or _font_candidates(bold=False))
        self.bold_candidates = tuple(bold_candidates or _font_candidates(bold=True))
        self._truetype_loader = truetype_loader
        self._default_loader = default_loader
        self.resolved_sources: dict[bool, str] = {}

    @lru_cache(maxsize=32)
    def font(self, size: int, *, bold: bool = False) -> Any:
        """Return a scalable system font or Pillow's portable default."""

        size = max(8, int(size))
        candidates = self.bold_candidates if bold else self.regular_candidates
        for candidate in candidates:
            try:
                font = self._truetype_loader(candidate, size=size)
            except (OSError, TypeError, ValueError):
                continue
            self.resolved_sources[bold] = candidate
            return font
        self.resolved_sources[bold] = "Pillow default"
        try:
            return self._default_loader(size=size)
        except TypeError:
            return self._default_loader()


class PillowCanvas:
    """Draw anti-aliased text and cards on an OpenCV BGR frame."""

    def __init__(
        self,
        bgr_frame: np.ndarray,
        *,
        fonts: FontResolver | None = None,
    ) -> None:
        if not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3:
            raise ValueError("PillowCanvas requires a BGR NumPy image")
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        # Keep an opaque RGB backing image so ImageDraw's RGBA mode composites
        # translucent fills instead of leaving unflattened alpha that can hide text.
        self.image = Image.fromarray(rgb).convert("RGB")
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        self.fonts = fonts or FontResolver()

    def font(self, size: float, *, bold: bool = False) -> Any:
        return self.fonts.font(round(size), bold=bold)

    def rectangle(
        self,
        rect: Rect,
        *,
        fill: tuple[int, ...],
        outline: tuple[int, ...] | None = None,
        width: int = 1,
        radius: int = 0,
    ) -> None:
        box = (rect.x, rect.y, rect.right - 1, rect.bottom - 1)
        if radius > 0:
            self.draw.rounded_rectangle(
                box, radius=radius, fill=fill, outline=outline, width=width
            )
        else:
            self.draw.rectangle(box, fill=fill, outline=outline, width=width)

    def line(
        self, points: Sequence[tuple[int, int]], *, fill: tuple[int, ...], width: int
    ) -> None:
        self.draw.line(points, fill=fill, width=width)

    def text_width(self, value: str, font: Any) -> int:
        try:
            box = self.draw.textbbox((0, 0), value, font=font)
        except UnicodeEncodeError:
            box = self.draw.textbbox((0, 0), _ascii_fallback(value), font=font)
        return max(0, box[2] - box[0])

    def ellipsize(self, value: object, font: Any, max_width: int) -> str:
        text = str(value).replace("\n", " ").strip()
        if max_width <= 0 or self.text_width(text, font) <= max_width:
            return text
        suffix = "…"
        while text and self.text_width(text + suffix, font) > max_width:
            text = text[:-1]
        return text.rstrip() + suffix if text else suffix

    def text(
        self,
        value: object,
        x: int,
        y: int,
        *,
        size: float,
        fill: tuple[int, ...] = THEME.text,
        bold: bool = False,
        max_width: int | None = None,
        anchor: str | None = None,
    ) -> None:
        font = self.font(size, bold=bold)
        rendered = str(value)
        if max_width is not None:
            rendered = self.ellipsize(rendered, font, max_width)
        try:
            self.draw.text((x, y), rendered, font=font, fill=fill, anchor=anchor)
        except UnicodeEncodeError:
            self.draw.text(
                (x, y),
                _ascii_fallback(rendered),
                font=font,
                fill=fill,
                anchor=anchor,
            )

    def wrapped_lines(
        self,
        value: object,
        *,
        size: float,
        max_width: int,
        max_lines: int,
        bold: bool = False,
    ) -> tuple[str, ...]:
        font = self.font(size, bold=bold)
        words = str(value).replace("\n", " ").split()
        if not words:
            return ()
        lines: list[str] = []
        current = ""
        for word_index, word in enumerate(words):
            candidate = f"{current} {word}".strip()
            if current and self.text_width(candidate, font) > max_width:
                lines.append(current)
                current = word
                if len(lines) == max_lines:
                    lines[-1] = self.ellipsize(
                        " ".join([lines[-1], current, *words[word_index + 1 :]]),
                        font,
                        max_width,
                    )
                    return tuple(lines)
            else:
                current = candidate
        if current and len(lines) < max_lines:
            lines.append(self.ellipsize(current, font, max_width))
        return tuple(lines[:max_lines])

    def paste_frozen_frame(
        self, bgr_frame: np.ndarray, rect: Rect, *, radius: int
    ) -> FramePlacement:
        """Fill the panel attractively while retaining one uncropped exact frame."""

        source = Image.fromarray(np.ascontiguousarray(bgr_frame[:, :, ::-1])).convert("RGB")
        source_width, source_height = source.size
        resampling = getattr(Image, "Resampling", Image).LANCZOS

        cover_scale = max(rect.width / source_width, rect.height / source_height)
        cover_size = (
            max(1, round(source_width * cover_scale)),
            max(1, round(source_height * cover_scale)),
        )
        cover = source.resize(cover_size, resampling)
        cover_x = max(0, (cover.width - rect.width) // 2)
        cover_y = max(0, (cover.height - rect.height) // 2)
        cover = cover.crop((cover_x, cover_y, cover_x + rect.width, cover_y + rect.height))
        cover = ImageEnhance.Brightness(cover).enhance(0.30).filter(
            ImageFilter.GaussianBlur(max(2.0, rect.height / 110.0))
        )
        self.image.paste(cover, (rect.x, rect.y))

        contain_scale = min(rect.width / source_width, rect.height / source_height)
        fitted_size = (
            max(1, round(source_width * contain_scale)),
            max(1, round(source_height * contain_scale)),
        )
        fitted = source.resize(fitted_size, resampling)
        x = rect.x + (rect.width - fitted.width) // 2
        y = rect.y + (rect.height - fitted.height) // 2
        self.image.paste(fitted, (x, y))
        self.draw.rounded_rectangle(
            (rect.x, rect.y, rect.right - 1, rect.bottom - 1),
            radius=radius,
            outline=(*THEME.border, 255),
            width=max(1, radius // 5),
        )
        return FramePlacement(x, y, fitted.width, fitted.height, contain_scale)

    def render(self) -> np.ndarray:
        rgb = np.asarray(self.image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])


def status_color(status: str) -> Color:
    """Map display-only status words to a restrained semantic color."""

    normalized = status.upper()
    if "FAIL" in normalized or "UNSTABLE" in normalized:
        return THEME.failure
    if normalized == "STABLE" or normalized.endswith(" STABLE"):
        return THEME.stable
    if any(
        marker in normalized
        for marker in ("REVIEW", "UNCERTAIN", "VARIABLE", "SENSITIVE")
    ):
        return THEME.review
    return THEME.muted


def draw_status_pill(
    canvas: PillowCanvas,
    status: str,
    *,
    x: int,
    y: int,
    size: float,
    max_width: int,
) -> int:
    """Draw one compact status pill and return its width."""

    font = canvas.font(size, bold=True)
    label = canvas.ellipsize(status, font, max_width - 16)
    width = min(max_width, canvas.text_width(label, font) + 16)
    height = max(18, round(size * 1.55))
    color = status_color(status)
    canvas.rectangle(
        Rect(x, y, width, height),
        fill=(*color, 38),
        outline=(*color, 190),
        radius=height // 2,
    )
    canvas.text(
        label,
        x + width // 2,
        y + height // 2,
        size=size,
        fill=color,
        bold=True,
        anchor="mm",
    )
    return width


def draw_page_header(
    canvas: PillowCanvas,
    layout: PresentationLayout,
    *,
    title: str,
    subtitle: str,
    method_tag: str,
) -> None:
    scale = layout.scale
    canvas.text(
        title,
        layout.header.x,
        layout.header.y,
        size=25 * scale,
        fill=THEME.text,
        bold=True,
        max_width=round(layout.header.width * 0.72),
    )
    canvas.text(
        subtitle,
        layout.header.x,
        layout.header.y + round(34 * scale),
        size=12.5 * scale,
        fill=THEME.muted,
        max_width=round(layout.header.width * 0.78),
    )
    font = canvas.font(11.5 * scale, bold=True)
    tag_width = canvas.text_width(method_tag, font) + round(24 * scale)
    tag_height = round(27 * scale)
    tag_x = layout.header.right - tag_width
    canvas.rectangle(
        Rect(tag_x, layout.header.y + round(7 * scale), tag_width, tag_height),
        fill=(*THEME.accent, 28),
        outline=(*THEME.accent, 170),
        radius=tag_height // 2,
    )
    canvas.text(
        method_tag,
        tag_x + tag_width // 2,
        layout.header.y + round(7 * scale) + tag_height // 2,
        size=11.5 * scale,
        fill=THEME.accent,
        bold=True,
        anchor="mm",
    )


def draw_page_footer(
    canvas: PillowCanvas,
    layout: PresentationLayout,
    *,
    page: int,
    page_count: int,
) -> None:
    scale = layout.scale
    controls = "P / U / SPACE  Resume    S  Screenshot    Q  Exit"
    if page_count > 1:
        controls = "A / D or [ / ]  Pages    " + controls
    canvas.text(
        controls,
        layout.footer.x,
        layout.footer.y + round(10 * scale),
        size=11.5 * scale,
        fill=THEME.muted,
        max_width=round(layout.footer.width * 0.82),
    )
    if page_count > 1:
        canvas.text(
            f"PAGE {page} / {page_count}",
            layout.footer.right,
            layout.footer.y + round(10 * scale),
            size=11.5 * scale,
            fill=THEME.text,
            bold=True,
            anchor="ra",
        )


def render_working_overlay(
    exact_frame: np.ndarray,
    *,
    title: str,
    primary: str,
    secondary: str,
    fonts: FontResolver | None = None,
) -> np.ndarray:
    """Render a sharp progress card directly on the frozen frame resolution."""

    canvas = PillowCanvas(exact_frame.copy(), fonts=fonts)
    width, height = canvas.image.size
    scale = max(0.72, min(width / 1280.0, height / 720.0))
    margin = round(24 * scale)
    card_height = min(height - 2 * margin, round(132 * scale))
    canvas.rectangle(
        Rect(margin, margin, width - 2 * margin, card_height),
        fill=(10, 17, 28, 228),
        outline=(*THEME.border, 220),
        width=max(1, round(2 * scale)),
        radius=round(14 * scale),
    )
    x = margin + round(20 * scale)
    available = width - 2 * margin - round(40 * scale)
    canvas.text(
        title,
        x,
        margin + round(16 * scale),
        size=21 * scale,
        fill=THEME.accent,
        bold=True,
        max_width=available,
    )
    canvas.text(
        primary,
        x,
        margin + round(54 * scale),
        size=14 * scale,
        fill=THEME.text,
        max_width=available,
    )
    canvas.text(
        secondary,
        x,
        margin + round(82 * scale),
        size=11.5 * scale,
        fill=THEME.muted,
        max_width=available,
    )
    return canvas.render()


def render_method_selector(
    exact_frame: np.ndarray,
    *,
    v1_sample_count: int,
    v2_sample_count: int,
    fonts: FontResolver | None = None,
) -> np.ndarray:
    """Render the centered V1/V2 method card over one unchanged frozen frame."""

    canvas = PillowCanvas(exact_frame.copy(), fonts=fonts)
    width, height = canvas.image.size
    canvas.rectangle(Rect(0, 0, width, height), fill=(4, 9, 17, 188))
    scale = max(0.72, min(width / 1280.0, height / 720.0))
    card_width = min(width - round(36 * scale), round(900 * scale))
    card_height = min(height - round(36 * scale), round(520 * scale))
    card = Rect(
        (width - card_width) // 2,
        (height - card_height) // 2,
        card_width,
        card_height,
    )
    radius = round(18 * scale)
    canvas.rectangle(
        card,
        fill=(*THEME.surface, 245),
        outline=(*THEME.border, 255),
        width=max(1, round(2 * scale)),
        radius=radius,
    )
    center_x = card.x + card.width // 2
    canvas.text(
        "UNCERTAINTY METHOD",
        center_x,
        card.y + round(42 * scale),
        size=27 * scale,
        fill=THEME.text,
        bold=True,
        anchor="ma",
    )
    canvas.text(
        "Same frozen frame",
        center_x,
        card.y + round(84 * scale),
        size=13 * scale,
        fill=THEME.muted,
        anchor="ma",
    )

    option_margin = round(34 * scale)
    option_x = card.x + option_margin
    option_width = card.width - 2 * option_margin
    option_height = round(112 * scale)
    first_y = card.y + round(120 * scale)
    option_gap = round(14 * scale)
    options = (
        (
            "1",
            "V1 — INPUT ROBUSTNESS",
            f"1 clean + {v1_sample_count} perturbed inputs",
        ),
        (
            "2",
            "V2 — MC DROPOUT",
            f"{v2_sample_count} stochastic model passes",
        ),
    )
    for index, (number, label, description) in enumerate(options):
        option = Rect(
            option_x,
            first_y + index * (option_height + option_gap),
            option_width,
            option_height,
        )
        canvas.rectangle(
            option,
            fill=(*THEME.surface_raised, 255),
            outline=(*THEME.border, 255),
            width=max(1, round(2 * scale)),
            radius=round(13 * scale),
        )
        number_size = round(58 * scale)
        number_rect = Rect(
            option.x + round(18 * scale),
            option.y + (option.height - number_size) // 2,
            number_size,
            number_size,
        )
        canvas.rectangle(
            number_rect,
            fill=(*THEME.accent, 38),
            outline=(*THEME.accent, 230),
            radius=round(12 * scale),
        )
        canvas.text(
            number,
            number_rect.x + number_rect.width // 2,
            number_rect.y + number_rect.height // 2,
            size=25 * scale,
            fill=THEME.accent,
            bold=True,
            anchor="mm",
        )
        text_x = number_rect.right + round(20 * scale)
        canvas.text(
            label,
            text_x,
            option.y + round(25 * scale),
            size=19 * scale,
            fill=THEME.text,
            bold=True,
            max_width=option.right - text_x - round(18 * scale),
        )
        canvas.text(
            description,
            text_x,
            option.y + round(62 * scale),
            size=13 * scale,
            fill=THEME.muted,
            max_width=option.right - text_x - round(18 * scale),
        )

    disclaimer_y = card.y + round(386 * scale)
    canvas.text(
        "Neither method is a calibrated probability of correctness.",
        center_x,
        disclaimer_y,
        size=11.5 * scale,
        fill=THEME.muted,
        anchor="ma",
        max_width=card.width - 2 * option_margin,
    )
    canvas.text(
        "ESC / SPACE / U — Resume",
        center_x,
        card.y + round(448 * scale),
        size=13 * scale,
        fill=THEME.text,
        bold=True,
        anchor="ma",
        max_width=card.width - 2 * option_margin,
    )
    return canvas.render()


def paginate(items: Iterable[Any], page_size: int) -> tuple[tuple[Any, ...], ...]:
    """Return deterministic non-empty pages for presentation rendering."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    values = tuple(items)
    if not values:
        return ((),)
    return tuple(
        values[index : index + page_size] for index in range(0, len(values), page_size)
    )


def ratio_text(mapping: Any, *, denominator: int, limit: int = 3) -> str:
    """Format a bounded class distribution without implying probability."""

    if not mapping:
        return "N/A"
    values = list(mapping.items())
    rendered = [f"{name} {value}/{denominator}" for name, value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"+{len(values) - limit} more")
    return " · ".join(rendered)


def share_text(mapping: Any, *, limit: int = 2) -> str:
    """Format evidence shares explicitly as evidence, not correctness."""

    if not mapping:
        return "N/A"
    values = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
    rendered = [f"{name} {share:.2f}" for name, share in values[:limit]]
    if len(values) > limit:
        rendered.append(f"+{len(values) - limit} more")
    return " · ".join(rendered)


def finite_metric(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"
