from __future__ import annotations

from conftest import FakeResult
import numpy as np

from uav_model_dashboard.detection_records import (
    CSV_FIELDS,
    DetectionAccumulator,
    records_from_result,
)


def test_detection_record_has_required_fields_without_track_id() -> None:
    result = FakeResult(np.zeros((48, 64, 3), dtype=np.uint8))
    records = records_from_result(result, frame_number=3, timestamp_seconds=0.2)
    assert len(records) == 1
    row = records[0].to_csv_row()
    assert tuple(row) == CSV_FIELDS
    assert row["frame_number"] == 3
    assert row["timestamp_seconds"] == 0.2
    assert row["track_id"] == ""
    assert row["class_name"] == "military_tank"
    assert row["confidence"] == 0.75
    assert [row[key] for key in ("x1", "y1", "x2", "y2")] == [
        1.0,
        2.0,
        30.0,
        40.0,
    ]


def test_tracking_record_preserves_track_id() -> None:
    result = FakeResult(
        np.zeros((48, 64, 3), dtype=np.uint8),
        track_id=17,
    )
    record = records_from_result(result, 1, 0.0)[0]
    assert record.track_id == 17


def test_accumulator_sorts_counts_and_handles_empty_confidence() -> None:
    accumulator = DetectionAccumulator()
    assert accumulator.average_confidence is None
    assert accumulator.maximum_confidence is None

    result = FakeResult(np.zeros((48, 64, 3), dtype=np.uint8))
    records = records_from_result(result, 1, 0.0)
    accumulator.add(records)
    accumulator.add(records)

    assert accumulator.total == 2
    assert accumulator.average_confidence == 0.75
    assert accumulator.maximum_confidence == 0.75
    assert accumulator.sorted_class_counts() == [["military_tank", 2]]
