from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from uav_model_dashboard.errors import DashboardError
from uav_model_dashboard.output_manager import OutputManager, safe_stem


def test_safe_stem_removes_windows_unsafe_characters() -> None:
    assert safe_stem(" tank videos : test? ") == "tank_videos_test"


def test_prepare_and_cleanup_are_confined_to_staging(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    manager = OutputManager(
        tmp_path / "outputs",
        clock=lambda: datetime(2026, 7, 28, 12, 0, 0),
    )
    paths = manager.prepare_job(source, "abc123")
    assert paths.owned_input.read_bytes() == b"video"
    assert paths.staging_dir.parent == manager.staging_root
    manager.cleanup_staging(paths.staging_dir)
    assert not paths.staging_dir.exists()

    with pytest.raises(DashboardError) as raised:
        manager.cleanup_staging(tmp_path)
    assert raised.value.code == "UNSAFE_CLEANUP_REFUSED"


def test_publish_does_not_delete_external_input_when_copy_disabled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"external-video")
    manager = OutputManager(
        tmp_path / "outputs",
        clock=lambda: datetime(2026, 7, 28, 12, 0, 0),
    )
    paths = manager.prepare_job(source, "job7", copy_input=False)
    paths.encoded_video.write_bytes(b"encoded")
    paths.csv_report.write_text("header\n", encoding="utf-8")
    published = manager.publish(paths)

    assert source.read_bytes() == b"external-video"
    assert published.annotated_video.read_bytes() == b"encoded"
    assert published.csv_report.is_file()
