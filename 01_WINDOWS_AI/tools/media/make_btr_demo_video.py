"""Build a local MP4 demo from an external BTR image directory."""

from __future__ import annotations

import os
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data" / "BTR" / "test" / "images"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "08_OUTPUTS" / "btr_demo.mp4"


def main() -> int:
    """Create the demo video using environment-configured external paths."""

    input_directory = Path(os.environ.get("UAV_BTR_IMAGE_DIR", DEFAULT_INPUT)).expanduser()
    output_video = Path(os.environ.get("UAV_DEMO_VIDEO_OUT", DEFAULT_OUTPUT)).expanduser()
    images = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in input_directory.glob(pattern)
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_directory}")

    width, height = 960, 540
    fps = 10
    frames_per_image = fps * 2
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {output_video}")
    try:
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            resized = cv2.resize(image, (width, height))
            for _ in range(frames_per_image):
                writer.write(resized)
    finally:
        writer.release()

    print(f"BTR demo video created: {output_video}")
    print(f"Images considered: {len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
