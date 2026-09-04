from __future__ import annotations

import csv
from pathlib import Path

import pytest

from conftest import FakeModel
from uav_model_dashboard.configuration import (
    DeviceChoice,
    InferenceMode,
    ProcessingSettings,
)
from uav_model_dashboard.errors import ProcessingCancelled
from uav_model_dashboard.model_manager import ModelManager
from uav_model_dashboard.output_manager import OutputManager
from uav_model_dashboard.processing_control import ProcessingController
from uav_model_dashboard.video_processor import ProcessingRequest, VideoProcessor


class CpuCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


class CpuTorch:
    cuda = CpuCuda()


def build_processor(
    tmp_path: Path,
    model: FakeModel,
    controller: ProcessingController,
    *,
    table_limit: int = 10_000,
) -> VideoProcessor:
    return VideoProcessor(
        ModelManager(
            loader=lambda _: model,
            torch_module=CpuTorch(),
            verifier=lambda _: "verified",
        ),
        OutputManager(tmp_path / "outputs"),
        controller,
        table_limit=table_limit,
    )


def test_detection_pipeline_produces_complete_video_csv_and_summary(
    tmp_path: Path,
    synthetic_video: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    model = FakeModel()
    controller = ProcessingController()
    processor = build_processor(tmp_path, model, controller, table_limit=2)
    settings = ProcessingSettings(
        confidence=0.50,
        iou=0.45,
        image_size=640,
        device=DeviceChoice.CPU,
        mode=InferenceMode.DETECTION,
    )

    result = processor.process(
        ProcessingRequest(synthetic_video, model_path, settings)
    )

    assert result.outputs.annotated_video.is_file()
    assert result.outputs.csv_report.is_file()
    assert result.summary["processed_frames"] == 4
    assert result.summary["total_detections"] == 4
    assert result.summary["inference_device"] == "CPU"
    assert result.summary["detection_table_truncated"] is True
    assert len(result.detection_rows) == 2
    assert result.class_count_rows == [["military_tank", 4]]
    assert len(model.predict_calls) == 4
    assert all(call["device"] == "cpu" for call in model.predict_calls)

    with result.outputs.csv_report.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert all(row["track_id"] == "" for row in rows)


def test_botsort_pipeline_passes_tracking_arguments(
    tmp_path: Path,
    synthetic_video: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    model = FakeModel()
    controller = ProcessingController()
    processor = build_processor(tmp_path, model, controller)
    settings = ProcessingSettings(
        device=DeviceChoice.CPU,
        mode=InferenceMode.BOTSORT,
    )
    result = processor.process(
        ProcessingRequest(synthetic_video, model_path, settings)
    )

    assert len(model.track_calls) == 4
    assert all(call["tracker"] == "botsort.yaml" for call in model.track_calls)
    assert all(call["persist"] is True for call in model.track_calls)
    with result.outputs.csv_report.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["track_id"] for row in rows} == {"7"}


def test_cancellation_removes_partial_output(
    tmp_path: Path,
    synthetic_video: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    model = FakeModel()
    controller = ProcessingController()
    processor = build_processor(tmp_path, model, controller)
    settings = ProcessingSettings(
        device=DeviceChoice.CPU,
        mode=InferenceMode.DETECTION,
    )

    def cancel_after_first_frame(_: float, *, desc: str) -> None:
        if desc.startswith("Frame 1"):
            controller.request_cancel()

    with pytest.raises(ProcessingCancelled):
        processor.process(
            ProcessingRequest(synthetic_video, model_path, settings),
            progress=cancel_after_first_frame,
        )

    output_root = tmp_path / "outputs"
    assert list((output_root / ".staging").iterdir()) == []
    assert [
        path
        for path in output_root.iterdir()
        if path.name != ".staging"
    ] == []
