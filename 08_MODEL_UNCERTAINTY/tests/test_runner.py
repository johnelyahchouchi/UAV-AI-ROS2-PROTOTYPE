from __future__ import annotations

from argparse import Namespace
import csv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uav_uncertainty import mc_stability_runner as runner  # noqa: E402
from uav_uncertainty.detection_types import Detection  # noqa: E402


class FakeDetector:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls = 0

    def detect(self, _image: object) -> list[Detection]:
        self.calls += 1
        return [Detection(0, "tank", 0.8, 5.0, 5.0, 20.0, 20.0)]


class LastSampleMissingDetector(FakeDetector):
    def detect(self, image: object) -> list[Detection]:
        detections = super().detect(image)
        return [] if self.calls == 11 else detections


class UnexpectedDetector:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("Detector must not be constructed after a collision.")


class RunnerTests(unittest.TestCase):
    def _arguments(
        self,
        root: Path,
        *,
        samples: int = 1,
        overwrite: bool = False,
        run_name: str = "test_run",
    ) -> Namespace:
        model_path = root / "model.pt"
        model_path.write_bytes(b"fake test checkpoint")
        image_path = root / "frame.jpg"
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        return Namespace(
            model=str(model_path),
            image=str(image_path),
            samples=samples,
            seed=42,
            imgsz=64,
            conf=0.25,
            iou=0.45,
            match_iou=0.50,
            device="cpu",
            output_dir=str(root / "outputs"),
            run_name=run_name,
            overwrite=overwrite,
        )

    def _run_quietly(
        self,
        args: Namespace,
        detector: type[FakeDetector],
    ) -> tuple[Path, Path]:
        with patch.object(runner, "UltralyticsDetector", detector), redirect_stdout(
            StringIO()
        ):
            return runner.run_analysis(args)

    def test_samples_ten_produces_eleven_total_inferences_and_metric_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = self._arguments(Path(temporary_directory), samples=10)
            json_path, csv_path = self._run_quietly(args, LastSampleMissingDetector)

            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["sampling"]["perturbation_count"], 10)
            self.assertEqual(summary["sampling"]["total_inference_sample_count"], 11)
            target = summary["targets"][0]
            self.assertEqual(target["sample_count"], 11)
            self.assertEqual(target["detection_count"], 10)
            self.assertEqual(target["detection_persistence"], 10 / 11)
            with csv_path.open(encoding="utf-8", newline="") as input_file:
                csv_target = next(csv.DictReader(input_file))
            self.assertEqual(csv_target["sample_count"], "11")
            self.assertEqual(float(csv_target["detection_persistence"]), 10 / 11)

    def test_existing_results_are_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            run_directory = root / "outputs" / args.run_name
            run_directory.mkdir(parents=True)
            (run_directory / "summary.json").write_text("existing", encoding="utf-8")
            (run_directory / "targets.csv").write_text("existing", encoding="utf-8")

            with patch.object(runner, "UltralyticsDetector", UnexpectedDetector):
                with self.assertRaisesRegex(FileExistsError, "Use --overwrite"):
                    runner.run_analysis(args)
            self.assertEqual(
                (run_directory / "summary.json").read_text(encoding="utf-8"),
                "existing",
            )
            self.assertEqual(
                (run_directory / "targets.csv").read_text(encoding="utf-8"),
                "existing",
            )

    def test_overwrite_replaces_owned_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root, overwrite=True)
            run_directory = root / "outputs" / args.run_name
            run_directory.mkdir(parents=True)
            (run_directory / "summary.json").write_text("old json", encoding="utf-8")
            (run_directory / "targets.csv").write_text("old csv", encoding="utf-8")

            json_path, csv_path = self._run_quietly(args, FakeDetector)

            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["target_count"], 1)
            self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("target_id,"))
            self.assertEqual(list(run_directory.glob(".*.tmp")), [])

    def test_overwrite_preserves_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root, overwrite=True)
            run_directory = root / "outputs" / args.run_name
            run_directory.mkdir(parents=True)
            (run_directory / "summary.json").write_text("old json", encoding="utf-8")
            unrelated = run_directory / "operator_notes.txt"
            unrelated.write_text("keep me", encoding="utf-8")

            self._run_quietly(args, FakeDetector)

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_repository_local_input_warning_does_not_reject_paths(self) -> None:
        warning = StringIO()
        local_path = Path(__file__)
        with redirect_stderr(warning):
            runner._warn_if_repository_local_inputs(local_path, local_path)
        self.assertIn("inside the Git repository", warning.getvalue())


if __name__ == "__main__":
    unittest.main()
