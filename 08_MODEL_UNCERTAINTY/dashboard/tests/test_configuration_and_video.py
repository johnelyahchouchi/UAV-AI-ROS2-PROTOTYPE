from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


DASHBOARD_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path[:0] = [str(DASHBOARD_SRC), str(CORE_SRC)]

from uav_uncertainty_dashboard.comparison import parse_sample_counts  # noqa: E402
from uav_uncertainty_dashboard.configuration import (  # noqa: E402
    DeviceChoice,
    ExperimentSettings,
    METHOD_INPUT_PERTURBATION_V1,
)
from uav_uncertainty_dashboard.errors import DashboardError  # noqa: E402
from uav_uncertainty_dashboard.video_sampling import (  # noqa: E402
    interval_timestamps,
    manual_timestamps,
    probe_video,
    sample_video_frames,
    total_inference_calls,
)


def synthetic_video(path: Path, frame_count: int = 10, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("Synthetic video writer unavailable.")
    try:
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()


class ConfigurationAndVideoTests(unittest.TestCase):
    def test_custom_sample_count_and_total_inference_semantics(self) -> None:
        settings = ExperimentSettings.from_values(
            METHOD_INPUT_PERTURBATION_V1,
            17,
            42,
            960,
            0.25,
            0.45,
            0.50,
            DeviceChoice.CPU.value,
        )
        self.assertEqual(settings.sample_count, 17)
        self.assertEqual(settings.to_dict()["total_inference_samples"], 18)
        self.assertEqual(total_inference_calls(6, 10), 66)

    def test_sample_count_comparison_parser_is_ordered_and_bounded(self) -> None:
        self.assertEqual(parse_sample_counts("5, 10, 20, 30, 10"), [5, 10, 20, 30])
        with self.assertRaises(DashboardError):
            parse_sample_counts("5, 10, 20, 30, 40")

    def test_interval_and_manual_timestamp_selection(self) -> None:
        self.assertEqual(interval_timestamps(21.0, 5.0, 20), [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(manual_timestamps("5, 10, 5, 15", 20.0, 20), [5.0, 10.0, 15.0])
        with self.assertRaises(DashboardError):
            manual_timestamps("21", 20.0, 20)

    def test_video_probe_and_sampling_decode_only_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            synthetic_video(path)
            info = probe_video(path)
            self.assertAlmostEqual(info.duration_seconds, 1.0, places=1)
            frames = sample_video_frames(path, [0.0, 0.5])
            self.assertEqual(len(frames), 2)
            self.assertEqual([frame.selection_index for frame in frames], [1, 2])
            self.assertEqual(frames[0].image.shape, (48, 64, 3))


if __name__ == "__main__":
    unittest.main()
