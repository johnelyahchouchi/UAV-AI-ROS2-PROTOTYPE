from pathlib import Path
import shutil
import cv2
import numpy as np


DATASET_DIR = Path(r"C:\uav_datasets_master\07_tank_platform_recognition")
RAW_DIR = DATASET_DIR / "00_raw_by_class"
REVIEW_DIR = RAW_DIR / "99_uncertain_review"
SKIPPED_DIR = RAW_DIR / "98_skipped_review_later"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEY_TO_CLASS = {
    ord("0"): "tank_unknown",
    ord("1"): "tank_t72",
    ord("2"): "tank_t80",
    ord("3"): "tank_t90",
    ord("4"): "tank_m1_abrams",
    ord("5"): "tank_leopard2",
    ord("6"): "tank_merkava",
    ord("7"): "tank_challenger2",
    ord("8"): "tank_leclerc",
    ord("9"): "ifv_bmp",

    ord("a"): "apc_btr",
    ord("A"): "apc_btr",

    ord("b"): "armored_truck",
    ord("B"): "armored_truck",

    ord("c"): "military_truck",
    ord("C"): "military_truck",

    ord("d"): "artillery",
    ord("D"): "artillery",
}

WINDOW_NAME = "Tank Platform Crop Sorter V2"


def get_images():
    images = []

    if not REVIEW_DIR.exists():
        return images

    for file in REVIEW_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return sorted(images)


def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_display(image, filename, remaining):
    h, w = image.shape[:2]

    max_w = 780
    max_h = 520

    scale = min(max_w / max(1, w), max_h / max(1, h), 5.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    x0 = 30
    y0 = 70

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + new_w, y0 + new_h),
        (0, 255, 255),
        2,
    )

    draw_text(canvas, "TANK PLATFORM CROP SORTER V2", 30, 35, (0, 255, 255), 0.8, 2)
    draw_text(canvas, f"Remaining in review: {remaining}", 850, 35, (255, 255, 255), 0.55, 1)
    draw_text(canvas, f"File: {filename}", 30, 670, (180, 180, 180), 0.45, 1)

    x = 850
    y = 85

    draw_text(canvas, "Press key to move image:", x, y, (255, 255, 255), 0.55, 2)
    y += 35

    instructions = [
        ("0", "tank_unknown"),
        ("1", "tank_t72"),
        ("2", "tank_t80"),
        ("3", "tank_t90"),
        ("4", "tank_m1_abrams"),
        ("5", "tank_leopard2"),
        ("6", "tank_merkava"),
        ("7", "tank_challenger2"),
        ("8", "tank_leclerc"),
        ("9", "ifv_bmp"),
        ("A", "apc_btr"),
        ("B", "armored_truck"),
        ("C", "military_truck"),
        ("D", "artillery"),
        ("S", "move to skipped review later"),
        ("X", "delete bad crop"),
        ("Q", "quit"),
    ]

    for key, label in instructions:
        color = (255, 255, 255)

        if key in ["0", "S"]:
            color = (0, 255, 255)

        if key in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            color = (0, 180, 255)

        if key in ["B", "C"]:
            color = (255, 180, 0)

        if key == "D":
            color = (0, 0, 255)

        if key == "X":
            color = (0, 0, 255)

        draw_text(canvas, f"{key}  ->  {label}", x, y, color, 0.48, 1)
        y += 28

    draw_text(
        canvas,
        "Important: S no longer loops. It moves image to 98_skipped_review_later.",
        800,
        650,
        (0, 255, 255),
        0.42,
        1,
    )

    draw_text(
        canvas,
        "If unsure, press S or 0. Do not guess exact tank type.",
        800,
        680,
        (0, 255, 255),
        0.42,
        1,
    )

    return canvas


def safe_move(src, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    dest = target_dir / src.name

    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1

        while True:
            candidate = target_dir / f"{stem}_{counter:03d}{suffix}"

            if not candidate.exists():
                dest = candidate
                break

            counter += 1

    shutil.move(str(src), str(dest))


def main():
    if not REVIEW_DIR.exists():
        print(f"[ERROR] Review folder does not exist: {REVIEW_DIR}")
        return

    SKIPPED_DIR.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Tank Platform Crop Sorter V2 started.")
    print("S now moves skipped images to:")
    print(SKIPPED_DIR)

    while True:
        images = get_images()

        if not images:
            print("No more images in 99_uncertain_review.")
            break

        img_path = images[0]
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"[WARN] Broken image deleted: {img_path}")
            img_path.unlink(missing_ok=True)
            continue

        display = make_display(image, img_path.name, len(images))
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("Quit requested.")
            break

        if key == ord("s") or key == ord("S"):
            safe_move(img_path, SKIPPED_DIR)
            print(f"Skipped for later review: {img_path.name}")
            continue

        if key == ord("x") or key == ord("X"):
            print(f"Deleted bad crop: {img_path.name}")
            img_path.unlink(missing_ok=True)
            continue

        if key in KEY_TO_CLASS:
            target_class = KEY_TO_CLASS[key]
            target_dir = RAW_DIR / target_class
            safe_move(img_path, target_dir)
            print(f"Moved {img_path.name} -> {target_class}")
            continue

        print("Unknown key. Use only the keys shown on the screen.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()