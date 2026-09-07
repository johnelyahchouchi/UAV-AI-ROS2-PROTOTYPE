#!/usr/bin/env python3
"""Compatibility entry point for the reusable live tester package."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_AI_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT, WINDOWS_AI_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from live_tester import *  # noqa: E402,F401,F403
from live_tester import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
