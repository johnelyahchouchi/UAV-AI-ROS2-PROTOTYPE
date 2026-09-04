"""Bounded, traversal-safe extraction and metadata reads for dataset ZIPs."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .config import SecurityLimits


class UnsafeArchiveError(ValueError):
    """Raised before extraction when a ZIP violates a safety budget."""


def _normalized_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise UnsafeArchiveError("ZIP member has an empty or NUL-containing name")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchiveError(f"ZIP member uses an absolute path: {name!r}")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"ZIP member has an unsafe path: {name!r}")
    return path


def _validate_member_type(info: zipfile.ZipInfo) -> None:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise UnsafeArchiveError(f"ZIP member is not a regular file/directory: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise UnsafeArchiveError(f"Encrypted ZIP members are unsupported: {info.filename!r}")


def validate_zip(
    archive: zipfile.ZipFile,
    *,
    limits: SecurityLimits | None = None,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate every member and aggregate budget before any extraction begins."""

    budget = limits or SecurityLimits.from_environment()
    infos = archive.infolist()
    if len(infos) > budget.max_archive_members:
        raise UnsafeArchiveError("ZIP contains too many members")
    total_size = 0
    targets: set[str] = set()
    validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in infos:
        relative = _normalized_member_path(info.filename)
        _validate_member_type(info)
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeArchiveError("ZIP contains a negative declared size")
        if info.file_size > budget.max_archive_member_size:
            raise UnsafeArchiveError(f"ZIP member exceeds size budget: {info.filename!r}")
        total_size += info.file_size
        if total_size > budget.max_archive_size:
            raise UnsafeArchiveError("ZIP exceeds total uncompressed size budget")
        if info.file_size and (
            info.compress_size == 0
            or info.file_size / info.compress_size > budget.max_archive_ratio
        ):
            raise UnsafeArchiveError(f"ZIP member has a suspicious compression ratio: {info.filename!r}")
        key = relative.as_posix().casefold()
        if key in targets:
            raise UnsafeArchiveError(f"ZIP contains a duplicate output path: {info.filename!r}")
        targets.add(key)
        validated.append((info, relative))
    return validated


def safe_extract_zip(
    source: str | Path | zipfile.ZipFile,
    destination: str | Path,
    *,
    limits: SecurityLimits | None = None,
) -> list[Path]:
    """Extract validated regular files while enforcing observed byte counts."""

    budget = limits or SecurityLimits.from_environment()
    destination_root = Path(destination).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    context = nullcontext(source) if isinstance(source, zipfile.ZipFile) else zipfile.ZipFile(source, "r")
    extracted: list[Path] = []
    with context as archive:
        validated = validate_zip(archive, limits=budget)
        observed_total = 0
        for info, relative in validated:
            target = destination_root.joinpath(*relative.parts).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as error:
                raise UnsafeArchiveError(f"ZIP member escapes destination: {info.filename!r}") from error
            if info.is_dir() or info.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            observed_member = 0
            with archive.open(info, "r") as source_stream, target.open("xb") as target_stream:
                while chunk := source_stream.read(1024 * 1024):
                    observed_member += len(chunk)
                    observed_total += len(chunk)
                    if observed_member > budget.max_archive_member_size:
                        raise UnsafeArchiveError("ZIP member exceeded its declared safety budget")
                    if observed_total > budget.max_archive_size:
                        raise UnsafeArchiveError("ZIP exceeded its total safety budget")
                    target_stream.write(chunk)
            if observed_member != info.file_size:
                raise UnsafeArchiveError(f"ZIP member size changed while reading: {info.filename!r}")
            extracted.append(target)
    return extracted


def safe_read_member(
    archive: zipfile.ZipFile,
    member: str | zipfile.ZipInfo,
    *,
    maximum_size: int | None = None,
) -> bytes:
    """Read a small ZIP metadata member with a strict pre-read and observed cap."""

    cap = maximum_size or SecurityLimits.from_environment().max_metadata_size
    info = archive.getinfo(member) if isinstance(member, str) else member
    _normalized_member_path(info.filename)
    _validate_member_type(info)
    if info.file_size < 0 or info.file_size > cap:
        raise UnsafeArchiveError(f"ZIP metadata member is too large: {info.filename!r}")
    with archive.open(info, "r") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap or len(data) != info.file_size:
        raise UnsafeArchiveError(f"ZIP metadata member exceeded its size cap: {info.filename!r}")
    return data
