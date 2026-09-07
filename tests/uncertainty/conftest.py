from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_SRC = PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src"
sys.path.insert(0, str(UNCERTAINTY_SRC))
