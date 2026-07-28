from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("The OpenCV test environment cannot create an MJPG video.")
    for index in range(4):
        frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class FakeScalar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def item(self) -> float | int:
        return self.value


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeBox:
    def __init__(self, *, track_id: int | None = None) -> None:
        self.cls = [FakeScalar(2)]
        self.conf = [FakeScalar(0.75)]
        self.xyxy = [FakeVector([1.0, 2.0, 30.0, 40.0])]
        self.id = None if track_id is None else [FakeScalar(track_id)]


class FakeResult:
    def __init__(self, frame: np.ndarray, *, track_id: int | None = None) -> None:
        self._frame = frame
        self.names = {2: "military_tank"}
        self.boxes = [FakeBox(track_id=track_id)]

    def plot(self) -> np.ndarray:
        return self._frame.copy()


class FakeModel:
    task = "detect"

    def __init__(self) -> None:
        self.predictor = None
        self.predict_calls: list[dict[str, object]] = []
        self.track_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.predict_calls.append(kwargs)
        return [FakeResult(kwargs["source"])]

    def track(self, **kwargs: object) -> list[FakeResult]:
        self.track_calls.append(kwargs)
        return [FakeResult(kwargs["source"], track_id=7)]
