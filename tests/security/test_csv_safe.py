import pytest

from uav_security.csv_safe import sanitize_csv_cell, sanitize_csv_row


@pytest.mark.parametrize(
    "payload",
    [
        "=WEBSERVICE(\"https://example.invalid\")",
        "=HYPERLINK(\"https://example.invalid\")",
        "+cmd",
        "-cmd",
        "@something",
        "\tpayload",
        "\rpayload",
    ],
)
def test_formula_capable_strings_are_neutralized(payload):
    assert sanitize_csv_cell(payload) == "'" + payload


def test_normal_text_and_numbers_are_unchanged():
    assert sanitize_csv_row(["military_tank", 7, 0.9]) == ["military_tank", 7, 0.9]
