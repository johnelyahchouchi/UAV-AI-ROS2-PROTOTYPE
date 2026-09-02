import shutil
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import TANK_RECOGNITION_DATASET_DIR

RAW_DIR = TANK_RECOGNITION_DATASET_DIR / "00_raw_by_class"

SOURCE_DIR = RAW_DIR / "artillery"

DESTINATIONS = {
    ord("c"): RAW_DIR / "artillery_cannon",
    ord("g"): RAW_DIR / "rocket_launcher_grad",
    ord("s"): RAW_DIR / "rocket_launcher_smerch",
    ord("m"): RAW_DIR / "mlrs_unknown",
    ord("u"): RAW_DIR / "unknown_artillery",
}

KEY_NAMES = {
    ord("c"): "artillery_cannon",
    ord("g"): "rocket_launcher_grad",
    ord("s"): "rocket_launcher_smerch",
    ord("m"): "mlrs_unknown",
    ord("u"): "unknown_artillery",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def unique_destination(dest_dir, src):
    dest = dest_dir / src.name

    if not dest.exists():
        return dest

    counter = 1

    while True:
        candidate = dest_dir / f"{src.stem}_{counter:03d}{src.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def resize_for_screen(img, max_width=1000, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


def main():
    if not SOURCE_DIR.exists():
        print(f"Source folder not found: {SOURCE_DIR}")
        return

    for dest in DESTINATIONS.values():
        dest.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images = sorted(images)

    print("\nARTILLERY / LAUNCHER MANUAL SORTER V1")
    print("=" * 80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Images to sort: {len(images)}")
    print("=" * 80)
    print("Controls:")
    print("  C = artillery_cannon")
    print("  G = rocket_launcher_grad")
    print("  S = rocket_launcher_smerch")
    print("  M = mlrs_unknown")
    print("  U = unknown_artillery")
    print("  K = skip / keep for later")
    print("  Q = quit")
    print("=" * 80)

    if len(images) == 0:
        print("No images found in artillery folder.")
        return

    index = 0

    while index < len(images):
        src = images[index]

        if not src.exists():
            index += 1
            continue

        img = cv2.imread(str(src))

        if img is None:
            print(f"[BAD IMAGE] {src.name}")
            index += 1
            continue

        display = resize_for_screen(img)

        title = f"Artillery Sorter [{index + 1}/{len(images)}] - {src.name}"
        cv2.imshow(title, display)

        print(f"\n[{index + 1}/{len(images)}] {src.name}")
        print("Press C/G/S/M/U, K skip, Q quit.")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)

        if key == ord("q"):
            print("Stopped by user.")
            break

        if key == ord("k"):
            print("[SKIP] kept for later")
            index += 1
            continue

        if key in DESTINATIONS:
            dest_dir = DESTINATIONS[key]
            dest = unique_destination(dest_dir, src)

            shutil.move(str(src), str(dest))

            print(f"[MOVED] {src.name} -> {KEY_NAMES[key]}")
            index += 1
            continue

        print("[NO ACTION] unknown key, image skipped for now")
        index += 1

    cv2.destroyAllWindows()

    print("\nSORTING SESSION FINISHED")
    print("=" * 80)

    for name in [
        "artillery",
        "artillery_cannon",
        "rocket_launcher_grad",
        "rocket_launcher_smerch",
        "mlrs_unknown",
        "unknown_artillery",
    ]:
        folder = RAW_DIR / name
        count = len([
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]) if folder.exists() else 0

        print(f"{name:25s}: {count:4d} images")


if __name__ == "__main__":
    main()
