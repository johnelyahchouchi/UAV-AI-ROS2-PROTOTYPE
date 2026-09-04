"""Neutralize spreadsheet formulas without changing numeric CSV cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_cell(value: Any) -> Any:
    """Prefix potentially executable spreadsheet strings with an apostrophe."""

    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_csv_row(row: Any) -> Any:
    """Return a sanitized mapping or positional CSV row."""

    if isinstance(row, Mapping):
        return {key: sanitize_csv_cell(value) for key, value in row.items()}
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return [sanitize_csv_cell(value) for value in row]
    raise TypeError("CSV row must be a mapping or non-string sequence")


def sanitize_csv_rows(rows: Any):
    """Yield sanitized rows lazily for writerows()."""

    for row in rows:
        yield sanitize_csv_row(row)
