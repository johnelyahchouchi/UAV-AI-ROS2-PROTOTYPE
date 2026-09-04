from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from uav_security.model_integrity import (
    ModelIntegrityError,
    load_trusted_yolo,
    sha256_file,
    verify_trusted_model,
)


def write_registry(path: Path, digests: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["FileName", "SHA256", "FullName"])
        writer.writeheader()
        for index, digest in enumerate(digests):
            writer.writerow(
                {"FileName": f"model-{index}.pt", "SHA256": digest, "FullName": "ignored"}
            )


def test_approved_sha256_is_accepted_before_loader(tmp_path: Path):
    model = tmp_path / "approved.pt"
    model.write_bytes(b"known checkpoint")
    registry = tmp_path / "trusted.csv"
    write_registry(registry, [hashlib.sha256(model.read_bytes()).hexdigest()])
    calls = []
    loaded = load_trusted_yolo(
        model, registry_path=registry, loader=lambda path: calls.append(path) or "model"
    )
    assert loaded == "model"
    assert calls == [str(model.resolve())]


def test_modified_checkpoint_is_rejected_without_calling_loader(tmp_path: Path):
    model = tmp_path / "modified.pt"
    model.write_bytes(b"original")
    registry = tmp_path / "trusted.csv"
    write_registry(registry, [sha256_file(model)])
    model.write_bytes(b"modified")
    called = False

    def loader(_path):
        nonlocal called
        called = True

    with pytest.raises(ModelIntegrityError, match="not present"):
        load_trusted_yolo(model, registry_path=registry, loader=loader)
    assert called is False


def test_unknown_checkpoint_is_rejected(tmp_path: Path):
    model = tmp_path / "unknown.pt"
    model.write_bytes(b"unknown")
    registry = tmp_path / "trusted.csv"
    write_registry(registry, ["0" * 64])
    with pytest.raises(ModelIntegrityError, match="not present"):
        verify_trusted_model(model, registry)


def test_wrong_extension_is_rejected(tmp_path: Path):
    model = tmp_path / "model.pth"
    model.write_bytes(b"pickle")
    registry = tmp_path / "trusted.csv"
    write_registry(registry, [hashlib.sha256(model.read_bytes()).hexdigest()])
    with pytest.raises(ModelIntegrityError, match=".pt"):
        verify_trusted_model(model, registry)


def test_missing_checkpoint_is_rejected_without_calling_loader(tmp_path: Path):
    registry = tmp_path / "trusted.csv"
    write_registry(registry, ["0" * 64])
    called = False

    def loader(_path):
        nonlocal called
        called = True

    with pytest.raises(ModelIntegrityError, match="Cannot read"):
        load_trusted_yolo(tmp_path / "missing.pt", registry_path=registry, loader=loader)
    assert called is False
