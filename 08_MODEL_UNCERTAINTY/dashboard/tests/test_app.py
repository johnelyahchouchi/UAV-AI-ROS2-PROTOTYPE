from __future__ import annotations

from pathlib import Path
import sys
import unittest


DASHBOARD_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path[:0] = [str(DASHBOARD_SRC), str(CORE_SRC)]

from uav_uncertainty_dashboard.app import build_app  # noqa: E402


class AppTests(unittest.TestCase):
    def test_dashboard_builds_with_required_tabs_and_only_real_method(self) -> None:
        demo = build_app()
        config = str(demo.config)
        for title in (
            "Run Experiment",
            "Overview",
            "Target Analysis",
            "Perturbation Analysis",
            "Video Analysis",
            "Compare Experiments",
            "Raw Results",
            "Exports",
        ):
            self.assertIn(title, config)
        self.assertIn("Input Perturbation V1", config)
        self.assertNotIn("Monte Carlo Dropout V2", config)
        self.assertIn("Cancel", config)
        self.assertIn("Manual timestamps", config)


if __name__ == "__main__":
    unittest.main()
