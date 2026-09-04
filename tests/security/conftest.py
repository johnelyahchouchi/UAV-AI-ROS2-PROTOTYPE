from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@pytest.fixture
def valid_detection() -> dict[str, object]:
    return {
        "class_id": 0,
        "class_name": "BTR",
        "confidence": 0.91,
        "target_id": "Target_1",
        "threat_score": 82.5,
        "bbox": [10, 20, 100, 120],
        "unknown_network_field": "discard me",
    }


@pytest.fixture
def valid_header(valid_detection) -> dict[str, object]:
    return {
        "protocol_version": 2,
        "session_id": "a" * 64,
        "seq": 1,
        "timestamp": 1_700_000_000.0,
        "source_width": 640,
        "source_height": 480,
        "jpeg_size": 4,
        "detections": [copy.deepcopy(valid_detection)],
    }
