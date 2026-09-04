from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from uav_security.config import SecurityLimits
from uav_security.safe_zip import UnsafeArchiveError, safe_extract_zip, safe_read_member


def make_zip(path: Path, members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_valid_archive_extracts_inside_destination(tmp_path: Path):
    archive = make_zip(tmp_path / "valid.zip", {"dataset/data.yaml": b"names: [tank]"})
    output = tmp_path / "output"
    extracted = safe_extract_zip(archive, output)
    assert extracted == [(output / "dataset" / "data.yaml").resolve()]
    assert extracted[0].read_bytes() == b"names: [tank]"


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:\\escape.txt"])
def test_traversal_and_absolute_paths_are_rejected(tmp_path: Path, name: str):
    archive = make_zip(tmp_path / "bad.zip", {name: b"bad"})
    with pytest.raises(UnsafeArchiveError, match="path"):
        safe_extract_zip(archive, tmp_path / "output")


def test_excessive_member_count_is_rejected(tmp_path: Path):
    archive = make_zip(tmp_path / "many.zip", {"one": b"1", "two": b"2"})
    with pytest.raises(UnsafeArchiveError, match="too many"):
        safe_extract_zip(
            archive,
            tmp_path / "output",
            limits=SecurityLimits(max_archive_members=1),
        )


def test_oversized_declared_member_is_rejected(tmp_path: Path):
    archive = make_zip(tmp_path / "large.zip", {"large.bin": b"12345"}, zipfile.ZIP_STORED)
    with pytest.raises(UnsafeArchiveError, match="size budget"):
        safe_extract_zip(
            archive,
            tmp_path / "output",
            limits=SecurityLimits(max_archive_member_size=4),
        )


def test_suspicious_compression_ratio_is_rejected(tmp_path: Path):
    archive = make_zip(tmp_path / "ratio.zip", {"zeros.bin": b"0" * 10_000})
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        safe_extract_zip(
            archive,
            tmp_path / "output",
            limits=SecurityLimits(max_archive_ratio=2),
        )


def test_metadata_read_has_independent_cap(tmp_path: Path):
    archive_path = make_zip(tmp_path / "metadata.zip", {"data.yaml": b"12345"})
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(UnsafeArchiveError, match="too large"):
            safe_read_member(archive, "data.yaml", maximum_size=4)
