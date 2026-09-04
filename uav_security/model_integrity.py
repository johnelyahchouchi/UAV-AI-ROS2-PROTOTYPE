"""Fail-closed SHA-256 verification for pickle-bearing YOLO checkpoints."""

from __future__ import annotations

import csv
import hashlib
import hmac
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Any


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
MODEL_ERROR = (
    "Model integrity verification failed. The checkpoint hash is not present "
    "in the trusted model registry. Do not load untrusted .pt checkpoints."
)


class ModelIntegrityError(RuntimeError):
    """Raised before deserialization when a checkpoint is not explicitly trusted."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_registry_path() -> Path:
    configured = os.environ.get("UAV_TRUSTED_MODEL_REGISTRY")
    if configured:
        return Path(configured).expanduser()
    return repository_root() / "00_PROJECT_GUIDE" / "ACTIVE_MODEL_HASHES.csv"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise ModelIntegrityError(f"Cannot read model checkpoint: {candidate}") from error
    return digest.hexdigest()


def trusted_hashes(registry_path: str | Path | None = None) -> frozenset[str]:
    """Read only valid SHA-256 values from the explicit trust registry."""

    path = default_registry_path() if registry_path is None else Path(registry_path)
    try:
        with path.expanduser().open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ModelIntegrityError("Trusted model registry has no header")
            hash_field = next(
                (field for field in reader.fieldnames if field.lower() == "sha256"), None
            )
            if hash_field is None:
                raise ModelIntegrityError("Trusted model registry has no SHA256 column")
            hashes = {
                value.lower()
                for row in reader
                if (value := str(row.get(hash_field, "")).strip())
                and SHA256_PATTERN.fullmatch(value)
            }
    except OSError as error:
        raise ModelIntegrityError(f"Cannot read trusted model registry: {path}") from error
    if not hashes:
        raise ModelIntegrityError("Trusted model registry contains no valid SHA-256 values")
    return frozenset(hashes)


def verify_trusted_model(
    path: str | Path,
    registry_path: str | Path | None = None,
) -> str:
    """Return the verified digest or fail before YOLO/PyTorch can deserialize it."""

    candidate = Path(path).expanduser()
    if candidate.suffix.lower() != ".pt":
        raise ModelIntegrityError("Only explicitly trusted .pt checkpoints are supported")
    digest = sha256_file(candidate)
    approved = trusted_hashes(registry_path)
    if not any(hmac.compare_digest(digest, expected) for expected in approved):
        raise ModelIntegrityError(MODEL_ERROR)
    return digest


def load_trusted_yolo(
    path: str | Path,
    *,
    registry_path: str | Path | None = None,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    """Verify a local checkpoint and only then invoke the Ultralytics loader."""

    candidate = Path(path).expanduser()
    verify_trusted_model(candidate, registry_path)
    candidate = candidate.resolve(strict=True)
    if loader is None:
        from ultralytics import YOLO

        loader = YOLO
    return loader(str(candidate))
