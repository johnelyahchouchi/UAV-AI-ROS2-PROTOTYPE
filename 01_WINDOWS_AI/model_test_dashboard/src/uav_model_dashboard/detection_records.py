"""Detection records, CSV serialization, and run summary aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


CSV_FIELDS = (
    "frame_number",
    "timestamp_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


@dataclass(frozen=True)
class DetectionRecord:
    """One model detection in one video frame."""

    frame_number: int
    timestamp_seconds: float
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_csv_row(self) -> dict[str, object]:
        """Return a stable, Windows-friendly CSV row."""
        row = asdict(self)
        row["timestamp_seconds"] = round(self.timestamp_seconds, 6)
        row["confidence"] = round(self.confidence, 6)
        for key in ("x1", "y1", "x2", "y2"):
            row[key] = round(float(row[key]), 2)
        row["track_id"] = "" if self.track_id is None else self.track_id
        return row

    def to_table_row(self) -> list[object]:
        """Return values in the same order as CSV_FIELDS."""
        row = self.to_csv_row()
        return [row[field] for field in CSV_FIELDS]


def class_name_for(names: Mapping[int, Any] | list[Any] | tuple[Any, ...], class_id: int) -> str:
    """Resolve a class name from either Ultralytics names representation."""
    try:
        if isinstance(names, Mapping):
            return str(names.get(class_id, f"class_{class_id}"))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return f"class_{class_id}"


def records_from_result(
    result: Any,
    frame_number: int,
    timestamp_seconds: float,
) -> list[DetectionRecord]:
    """Convert an Ultralytics Results object into serializable records."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = getattr(result, "names", {})
    records: list[DetectionRecord] = []

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        coordinates = [float(value) for value in box.xyxy[0].tolist()]
        if len(coordinates) != 4:
            continue
        track_id = None
        if getattr(box, "id", None) is not None:
            track_id = int(box.id[0].item())
        x1, y1, x2, y2 = coordinates
        records.append(
            DetectionRecord(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                track_id=track_id,
                class_id=class_id,
                class_name=class_name_for(names, class_id),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return records


class DetectionAccumulator:
    """Collect summary values without retaining every detection in memory."""

    def __init__(self) -> None:
        self.total = 0
        self.confidence_sum = 0.0
        self.maximum_confidence: float | None = None
        self.class_counts: Counter[str] = Counter()

    def add(self, records: Iterable[DetectionRecord]) -> None:
        """Add a sequence of records to the running summary."""
        for record in records:
            self.total += 1
            self.confidence_sum += record.confidence
            self.class_counts[record.class_name] += 1
            if self.maximum_confidence is None:
                self.maximum_confidence = record.confidence
            else:
                self.maximum_confidence = max(
                    self.maximum_confidence,
                    record.confidence,
                )

    @property
    def average_confidence(self) -> float | None:
        """Return average confidence, or None when there were no detections."""
        if self.total == 0:
            return None
        return self.confidence_sum / self.total

    def sorted_class_counts(self) -> list[list[object]]:
        """Return class counts sorted by count descending then class name."""
        return [
            [class_name, count]
            for class_name, count in sorted(
                self.class_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
