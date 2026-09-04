from __future__ import annotations

import subprocess

import pytest

from uav_security.source_urls import (
    SourceResolutionError,
    is_trusted_youtube_url,
    resolve_video_source,
    source_log_label,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=abc",
        "https://www.youtube.com/watch?v=abc",
        "http://m.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
    ],
)
def test_real_youtube_urls_are_accepted(url):
    assert is_trusted_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/?next=youtube.com",
        "https://youtube.com.attacker.example/video",
        "file://youtube.com/video",
        "youtube.com/watch?v=abc",
    ],
)
def test_lookalike_or_non_http_urls_are_rejected(url):
    assert not is_trusted_youtube_url(url)


def test_ytdlp_uses_option_terminator_and_timeout():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="https://stream.example/video\n")

    source = "https://youtu.be/abc"
    assert resolve_video_source(source, runner=runner, timeout=7) == "https://stream.example/video"
    assert calls[0][0][-2:] == ["--", source]
    assert calls[0][1]["timeout"] == 7


def test_empty_ytdlp_output_is_rejected():
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="\n")

    with pytest.raises(SourceResolutionError, match="no playable"):
        resolve_video_source("https://youtube.com/watch?v=x", runner=runner)


def test_source_log_label_removes_credentials_queries_and_full_paths():
    assert source_log_label("rtsp://user:secret@camera.example/live?token=value") == (
        "rtsp://camera.example/..."
    )
    assert source_log_label(r"C:\private\mission\video.mp4") == "video.mp4"
