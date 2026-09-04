import math

import pytest

from uav_security.input_validation import InputValidationError, validate_sender_settings


def valid_values(**overrides):
    values = {
        "target": "127.0.0.1",
        "port": "5010",
        "confidence": "0.25",
        "iou": "0.45",
        "image_size": "960",
        "stride": "1",
        "send_width": "960",
        "show": "1",
        "military_only": "1",
    }
    values.update(overrides)
    return values


def test_valid_control_panel_values_are_preserved():
    settings = validate_sender_settings(**valid_values())
    assert settings.target == "127.0.0.1"
    assert settings.port == 5010
    assert settings.confidence == 0.25


@pytest.mark.parametrize("target", ["not-an-ip", "127.0.0.1; whoami", ""])
def test_invalid_ip_is_rejected(target):
    with pytest.raises(InputValidationError, match="valid"):
        validate_sender_settings(**valid_values(target=target))


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", 0),
        ("port", 65536),
        ("confidence", math.nan),
        ("iou", math.inf),
        ("image_size", 0),
        ("stride", 0),
        ("send_width", -1),
        ("show", 2),
        ("military_only", "true"),
    ],
)
def test_invalid_numeric_and_flag_values_are_rejected(field, value):
    with pytest.raises(InputValidationError):
        validate_sender_settings(**valid_values(**{field: value}))
