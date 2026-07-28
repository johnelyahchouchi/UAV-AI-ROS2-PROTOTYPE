from __future__ import annotations

from uav_model_dashboard import app


def test_dashboard_builds_with_upload_only_video_and_required_defaults() -> None:
    demo = app.build_app()
    config = demo.config
    config_text = str(config)
    input_videos = [
        component
        for component in config["components"]
        if component["type"] == "video"
        and component["props"].get("label") == "Input video"
    ]
    assert "UAV Model Test Dashboard" in config_text
    assert len(input_videos) == 1
    assert input_videos[0]["props"]["sources"] == ["upload"]
    assert "0.5" in config_text
    assert "BoT-SORT tracking" in config_text
    assert "GPU 0" in config_text
