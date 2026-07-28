from __future__ import annotations

from pathlib import Path

import pytest

from uav_model_dashboard.errors import DashboardError
from uav_model_dashboard.source_adapter import UploadedVideoSource


def test_valid_video_metadata_and_frame_sequence(synthetic_video: Path) -> None:
    with UploadedVideoSource(synthetic_video) as source:
        assert source.metadata.width == 64
        assert source.metadata.height == 48
        assert source.metadata.fps == pytest.approx(10.0)
        frames = list(source.frames())

    assert [frame.number for frame in frames] == [1, 2, 3, 4]
    assert frames[0].timestamp_seconds == 0.0
    assert frames[-1].timestamp_seconds >= 0.0


def test_corrupt_video_reports_clear_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_text("not a video", encoding="utf-8")
    with pytest.raises(DashboardError) as raised:
        UploadedVideoSource(corrupt)
    assert raised.value.code == "VIDEO_OPEN_FAILED"
