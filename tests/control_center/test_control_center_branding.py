from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = PROJECT_ROOT / "01_WINDOWS_AI"
if str(WINDOWS_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(WINDOWS_AI_ROOT))

from control_center.app import _prepare_brand_logo_image  # noqa: E402
from control_center.branding import Branding, load_branding  # noqa: E402


CONTROL_CENTER_ROOT = WINDOWS_AI_ROOT / "control_center"


def test_committed_branding_matches_dashboard_header() -> None:
    branding = load_branding()

    assert branding.title == "UAV Perception and Mission Control"
    assert branding.subtitle == "Research Prototype Dashboard"
    assert branding.author == "JOHN EL YAHCHOUCHI"
    assert branding.internship_period == "June - September 2026 internship"
    assert branding.logo_path == (
        CONTROL_CENTER_ROOT / "assets" / "additess_logo_dark.png"
    ).resolve()
    assert branding.logo_max_width == 380
    assert branding.logo_max_height == 70


def test_branding_can_be_changed_without_editing_application_code(
    tmp_path: Path,
) -> None:
    logo = tmp_path / "brand.png"
    Image.new("RGB", (20, 10), "white").save(logo)
    config = tmp_path / "branding.json"
    config.write_text(
        json.dumps(
            {
                "title": "Flight Lab",
                "subtitle": "Operator Console",
                "author": "Example Author",
                "internship_period": "Summer 2026",
                "logo_path": logo.name,
                "logo_max_width": 420,
                "logo_max_height": 80,
            }
        ),
        encoding="utf-8",
    )

    branding = load_branding(config)

    assert branding == Branding(
        title="Flight Lab",
        subtitle="Operator Console",
        author="Example Author",
        internship_period="Summer 2026",
        logo_path=logo.resolve(),
        logo_max_width=420,
        logo_max_height=80,
    )


def test_invalid_branding_file_uses_safe_defaults(tmp_path: Path) -> None:
    config = tmp_path / "branding.json"
    config.write_text("not json", encoding="utf-8")

    branding = load_branding(config)

    assert branding.title == Branding().title
    assert branding.author == Branding().author
    assert branding.logo_path == (
        tmp_path / "assets" / "additess_logo_dark.png"
    ).resolve()


def test_branding_text_normalizes_em_dash(tmp_path: Path) -> None:
    config = tmp_path / "branding.json"
    config.write_text(
        json.dumps({"title": f"UAV{chr(0x2014)}Control"}),
        encoding="utf-8",
    )

    assert load_branding(config).title == "UAV-Control"


def test_logo_asset_is_a_valid_wide_image() -> None:
    logo = CONTROL_CENTER_ROOT / "assets" / "additess_logo_dark.png"

    with Image.open(logo) as image:
        image.verify()
    with Image.open(logo) as image:
        assert image.width > image.height
        assert image.width >= 500


def test_dark_logo_is_matted_and_scaled_for_the_header() -> None:
    branding = load_branding()

    logo = _prepare_brand_logo_image(branding)

    assert logo.mode == "RGBA"
    assert logo.width <= branding.logo_max_width
    assert logo.height <= branding.logo_max_height
    assert logo.width > logo.height * 5
    assert logo.getchannel("A").getextrema() == (0, 255)


def test_control_center_visible_text_contains_no_em_dash() -> None:
    for path in (
        CONTROL_CENTER_ROOT / "app.py",
        CONTROL_CENTER_ROOT / "branding.json",
    ):
        assert chr(0x2014) not in path.read_text(encoding="utf-8")
