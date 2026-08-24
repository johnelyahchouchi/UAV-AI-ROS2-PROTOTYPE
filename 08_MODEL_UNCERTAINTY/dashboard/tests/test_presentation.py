from __future__ import annotations

import json
from math import isnan
from pathlib import Path
import sys
import unittest

import matplotlib.pyplot as plt


DASHBOARD_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path[:0] = [str(DASHBOARD_SRC), str(CORE_SRC)]

from uav_uncertainty_dashboard.app import _video_table, clear_results  # noqa: E402
from uav_uncertainty_dashboard.diagnostics import instability_events  # noqa: E402
from uav_uncertainty_dashboard.presentation import (  # noqa: E402
    display_comparison_rows,
    entropy_status_markdown,
    review_flags_markdown,
    target_id_scope_note,
)
from uav_uncertainty_dashboard.result_loader import (  # noqa: E402
    family_rows,
    overview_markdown,
    target_rows,
)
from uav_uncertainty_dashboard.result_models import LoadedExperiment  # noqa: E402
from uav_uncertainty_dashboard.visualizations import (  # noqa: E402
    target_metrics_figure,
    video_timeline_figure,
)


def target(entropy: float = 0.0) -> dict[str, object]:
    return {
        "target_id": "target_1",
        "dominant_class": "military_tank",
        "detection_count": 9,
        "sample_count": 11,
        "detected_sample_indices": list(range(9)),
        "missing_sample_indices": [9, 10],
        "detection_persistence": 0.8181818181818182,
        "confidence_mean": 0.2844552960660722,
        "confidence_std": 0.005786497696478189,
        "class_agreement": 1.0,
        "class_entropy_bits": entropy,
        "mean_iou_to_reference": 0.9345678912345,
        "bbox_center_std_pixels": {"x": 1.23456, "y": 2.34567},
        "bbox_size_std_pixels": {"x": 3.45678, "y": 4.56789},
        "reference_bbox_xyxy": [1.0, 2.0, 20.0, 21.0],
        "reference_box_source": "clean_baseline",
    }


def metadata(input_kind: str = "Image") -> dict[str, object]:
    return {
        "dashboard_schema_version": "1.0",
        "input_kind": input_kind,
        "input_name": "input.mp4" if input_kind == "Video" else "input.jpg",
        "model_name": "model.pt",
        "method_name": "Input Perturbation V1",
        "method_version": "1.0",
        "configuration": {
            "seed": 42,
            "sample_count": 10,
            "total_inference_samples": 11,
            "image_size": 960,
            "confidence": 0.254567,
            "nms_iou": 0.456789,
            "match_iou": 0.501234,
            "overlap_iou": 0.8,
            "device": "CPU",
        },
    }


class PresentationTests(unittest.TestCase):
    def test_target_table_rounds_to_three_decimals_without_mutating_raw_values(self) -> None:
        raw_target = target()
        summary = {"targets": [raw_target]}

        row = target_rows(summary)[0]

        self.assertEqual(row[3:9], ["0.818", "0.284", "0.006", "1.000", "0.000", "0.935"])
        self.assertEqual(row[9:13], ["1.235", "2.346", "3.457", "4.568"])
        self.assertEqual(raw_target["detection_persistence"], 0.8181818181818182)
        self.assertEqual(raw_target["confidence_mean"], 0.2844552960660722)
        self.assertIn("0.8181818181818182", json.dumps(summary))
        self.assertIn("0.005786497696478189", json.dumps(summary))

    def test_perturbation_and_comparison_display_values_are_rounded_only_for_ui(self) -> None:
        samples = [
            {
                "sample_index": 0,
                "family": "clean_baseline",
                "detection_count": 1,
                "target_ids_present": ["target_1"],
                "target_ids_missing": [],
                "detections": [{"target_id": "target_1", "confidence": 0.7}],
            },
            {
                "sample_index": 1,
                "family": "brightness",
                "detection_count": 1,
                "target_ids_present": ["target_1"],
                "target_ids_missing": [],
                "detections": [{"target_id": "target_1", "confidence": 0.9844552960660722}],
            },
        ]
        self.assertEqual(family_rows(samples)[0][2], "1.000")
        self.assertEqual(family_rows(samples)[0][5], "0.284")

        raw_row = [
            "run", "method", "1.0", "model", "input", 10, 42, 960,
            0.254567, 0.501234, 1, 0.8181818181818182,
            0.005786497696478189, 1.0, 0.0, 0.9345678912345,
        ]
        displayed = display_comparison_rows([raw_row])[0]
        self.assertEqual(displayed[8:10], ["0.255", "0.501"])
        self.assertEqual(displayed[11:16], ["0.818", "0.006", "1.000", "0.000", "0.935"])
        self.assertEqual(raw_row[11], 0.8181818181818182)

    def test_video_no_detection_row_uses_na_and_adds_review_flag(self) -> None:
        record = {
            "selection_index": 11,
            "timestamp_seconds": 50.0,
            "frame_index": 1500,
            "target_count": 0,
            "dominant_classes": [],
            "mean_persistence": 0.0,
            "mean_confidence": 0.0,
            "mean_class_agreement": 0.0,
            "mean_entropy": 0.0,
            "mean_iou": 0.0,
            "instability_events": [],
            "directory": "frames/frame_011",
        }
        run = LoadedExperiment(Path("run"), metadata("Video"), {"targets": []}, [], {"frames": [record]})

        row = _video_table(run)[0]

        self.assertEqual(row[:5], [11, "50.000", 1500, 0, "No detections"])
        self.assertEqual(row[5:10], ["N/A"] * 5)
        self.assertIn("No detections in this sampled frame", row[10])
        self.assertEqual(record["mean_entropy"], 0.0)

    def test_entropy_messages_cover_no_targets_zero_and_nonzero(self) -> None:
        no_targets = {"targets": []}
        all_zero = {"targets": [target(0.0), {**target(0.0), "target_id": "target_2"}]}
        nonzero = {"targets": [target(0.25)]}

        self.assertIn("Class entropy: N/A", entropy_status_markdown(no_targets))
        self.assertIn("0.000 for all targets", entropy_status_markdown(all_zero))
        self.assertIn("this image", entropy_status_markdown(all_zero))
        self.assertNotIn("frame", entropy_status_markdown(all_zero))
        self.assertIn("sampled frame", entropy_status_markdown(all_zero, is_video=True))
        self.assertEqual(entropy_status_markdown(nonzero), "")

        no_target_figure = target_metrics_figure(no_targets)
        zero_figure = target_metrics_figure(all_zero)
        nonzero_figure = target_metrics_figure(nonzero)
        try:
            self.assertIn("Class entropy: N/A", no_target_figure.axes[3].texts[0].get_text())
            self.assertIn("0.000 for all targets", zero_figure.axes[3].texts[0].get_text())
            self.assertGreater(len(nonzero_figure.axes[3].patches), 0)
        finally:
            plt.close(no_target_figure)
            plt.close(zero_figure)
            plt.close(nonzero_figure)

    def test_video_timeline_uses_gap_for_no_detection_persistence(self) -> None:
        figure = video_timeline_figure(
            [
                {"timestamp_seconds": 0.0, "target_count": 2, "mean_persistence": 0.8},
                {"timestamp_seconds": 5.0, "target_count": 0, "mean_persistence": 0.0},
                {"timestamp_seconds": 10.0, "target_count": 1, "mean_persistence": 1.0},
            ]
        )
        try:
            persistence_values = figure.axes[1].lines[0].get_ydata()
            self.assertTrue(isnan(persistence_values[1]))
        finally:
            plt.close(figure)

    def test_video_overview_identifies_selected_frame_and_image_omits_video_wording(self) -> None:
        summary = {"targets": [target()]}
        frames = [
            {"selection_index": 1},
            {"selection_index": 2},
            {"selection_index": 3, "timestamp_seconds": 10.0, "frame_index": 300},
        ]
        video_run = LoadedExperiment(Path("run"), metadata("Video"), summary, [], {"frames": frames})
        video_text = overview_markdown(video_run, video_frame=frames[2], review_flag_count=2)
        image_run = LoadedExperiment(Path("run"), metadata("Image"), summary, [])
        image_text = overview_markdown(image_run, review_flag_count=2)

        self.assertIn("Currently displaying sampled frame", video_text)
        self.assertIn("Timestamp:** 10.000 s", video_text)
        self.assertIn("Frame index:** 300", video_text)
        self.assertIn("Selection:** 3 of 3", video_text)
        self.assertIn("Use Video Analysis", video_text)
        self.assertIn("frame-local", target_id_scope_note(True).lower())
        self.assertNotIn("sampled frame", image_text.lower())
        self.assertNotIn("Video Analysis", image_text)
        self.assertEqual(target_id_scope_note(False), "")

    def test_review_flags_are_prominent_and_categorized(self) -> None:
        no_detection_events = instability_events([], [])
        text = review_flags_markdown(
            no_detection_events
            + [
                "target_1: appears only under perturbation in this run.",
                "target_1: missing in 8/11 samples.",
                "target_2: class disagreement observed; agreement=0.889.",
                "target_1 and target_2: possible overlapping/alternative detection (reference IoU=0.952).",
                "target_4: box variation exceeds the dashboard diagnostic rule.",
            ]
        )
        for category in (
            "No detections",
            "Perturbation-only detection",
            "Missing detections",
            "Class disagreement",
            "Overlap diagnostic",
            "Bounding-box variation",
        ):
            self.assertIn(f"**{category}**", text)
        self.assertIn("not automatic failures", text)

    def test_clear_result_shape_matches_render_outputs(self) -> None:
        self.assertEqual(len(clear_results()), 25)


if __name__ == "__main__":
    unittest.main()
