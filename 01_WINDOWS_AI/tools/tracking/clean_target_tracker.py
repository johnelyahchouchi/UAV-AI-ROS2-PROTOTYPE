from ultralytics import YOLO
import cv2
from pathlib import Path
import csv
import argparse
import time


# ============================================================
# High-contrast colors
# OpenCV uses BGR, not RGB
# ============================================================

TARGET_COLORS = [
    (255, 255, 0),    # Target_1 = cyan
    (255, 0, 255),    # Target_2 = magenta
    (0, 165, 255),    # Target_3 = orange
    (0, 0, 255),      # Target_4 = red
    (255, 0, 0),      # Target_5 = blue
    (0, 255, 255),    # Target_6 = yellow
]

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)


def target_color(target_id):
    """
    Give each Target_X a clear different color.
    """
    try:
        n = int(str(target_id).split("_")[-1])
    except Exception:
        n = 1

    return TARGET_COLORS[(n - 1) % len(TARGET_COLORS)]


def draw_text_box(frame, text, x, y, color, scale=0.70, thickness=2):
    """
    Draw readable white text on black background with colored border.
    Much clearer than raw green text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = int(max(5, x))
    y = int(max(25, y))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x2 = min(frame.shape[1] - 5, x + tw + 16)
    y1 = max(5, y - th - baseline - 10)
    y2 = min(frame.shape[0] - 5, y + baseline + 8)

    cv2.rectangle(frame, (x, y1), (x2, y2), BLACK, -1)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x + 8, y), font, scale, WHITE, thickness)


def center_of_box(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(p1, p2):
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def get_class_name(model, cls_id):
    """
    Works whether model.names is a dict or a list.
    """
    cls_id = int(cls_id)

    if isinstance(model.names, dict):
        return model.names.get(cls_id, str(cls_id))

    return model.names[cls_id]


def parse_allowed_classes(text):
    """
    Convert class string into a clean set.
    Example:
    military_tank,military_vehicle,military_truck
    """
    return set([x.strip() for x in text.split(",") if x.strip()])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="military_kaggle_v1.pt")
    parser.add_argument("--source", default="1minutesVIEWdroneVIDEOTANKS.mp4")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tracker", default="botsort.yaml")

    # Tracking memory settings
    parser.add_argument("--reid_distance", type=float, default=180.0)
    parser.add_argument("--max_missing_frames", type=int, default=60)

    # This prevents fake short detections from becoming Target_2 / Target_3
    parser.add_argument("--confirm_frames", type=int, default=3)

    # Ignore tiny bad boxes
    parser.add_argument("--min_area", type=int, default=250)

    # Only track military vehicle classes
    parser.add_argument(
        "--allowed_classes",
        default="military_tank,military_vehicle,military_truck,tank,armored_vehicle"
    )

    args = parser.parse_args()

    allowed_classes = parse_allowed_classes(args.allowed_classes)

    source_path = Path(args.source)
    out_dir = Path("tracking_tests") / f"clean_targets_{source_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video = out_dir / "clean_target_tracking.mp4"
    out_csv = out_dir / "clean_target_history.csv"

    print("=" * 70)
    print("CLEAN TARGET TRACKER")
    print("=" * 70)
    print(f"Model:   {args.model}")
    print(f"Source:  {args.source}")
    print(f"Tracker: {args.tracker}")
    print(f"Output:  {out_video}")
    print(f"CSV:     {out_csv}")
    print(f"Allowed classes: {sorted(list(allowed_classes))}")
    print("=" * 70)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()

    writer = None

    # raw tracker id -> clean Target_X
    raw_to_target = {}

    # target memory
    target_memory = {}

    # pending raw tracks before becoming official Target_X
    pending_raw_tracks = {}

    next_target_number = 1

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame",
        "time_sec",
        "target_id",
        "raw_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "status"
    ])

    def create_new_target(raw_id, center, frame_idx, class_name):
        nonlocal next_target_number

        target_id = f"Target_{next_target_number}"
        next_target_number += 1

        raw_to_target[raw_id] = target_id

        target_memory[target_id] = {
            "raw_ids": {raw_id},
            "last_center": center,
            "last_frame": frame_idx,
            "class_name": class_name,
            "last_conf": 0.0,
            "trail": [],
            "created_frame": frame_idx,
        }

        return target_id

    def find_existing_target(center, frame_idx):
        """
        If tracker changes raw ID, reconnect it to the old Target_X
        using last known position.
        """
        best_target = None
        best_dist = 999999.0

        for target_id, mem in target_memory.items():
            age = frame_idx - mem["last_frame"]

            if age > args.max_missing_frames:
                continue

            d = distance(center, mem["last_center"])

            if d < best_dist and d <= args.reid_distance:
                best_dist = d
                best_target = target_id

        return best_target

    def assign_target_id(raw_id, center, frame_idx, class_name):
        """
        Convert raw tracker ID into clean Target_X.
        Also prevents one-frame false positives from creating new target IDs.
        """
        raw_id = int(raw_id)

        # Existing raw ID
        if raw_id in raw_to_target:
            return raw_to_target[raw_id]

        # Try to reconnect to old target if tracker changed ID
        existing_target = find_existing_target(center, frame_idx)

        if existing_target is not None:
            raw_to_target[raw_id] = existing_target
            target_memory[existing_target]["raw_ids"].add(raw_id)
            return existing_target

        # New raw ID: wait until seen for several frames
        if raw_id not in pending_raw_tracks:
            pending_raw_tracks[raw_id] = {
                "count": 1,
                "first_frame": frame_idx,
                "last_center": center,
                "class_name": class_name,
            }
        else:
            pending_raw_tracks[raw_id]["count"] += 1
            pending_raw_tracks[raw_id]["last_center"] = center
            pending_raw_tracks[raw_id]["class_name"] = class_name

        if pending_raw_tracks[raw_id]["count"] < args.confirm_frames:
            return None

        return create_new_target(raw_id, center, frame_idx, class_name)

    frame_idx = 0

    results = model.track(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        persist=True,
        stream=True,
        verbose=False,
    )

    cv2.namedWindow("Clean Target Tracker", cv2.WINDOW_NORMAL)

    for result in results:
        frame_idx += 1
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

        active_this_frame = set()

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, raw_id, conf, cls_id in zip(xyxy, ids, confs, clss):
                x1, y1, x2, y2 = box.astype(int)

                box_area = max(0, x2 - x1) * max(0, y2 - y1)

                if box_area < args.min_area:
                    continue

                class_name = get_class_name(model, cls_id)

                if class_name not in allowed_classes:
                    continue

                cx, cy = center_of_box(x1, y1, x2, y2)

                target_id = assign_target_id(
                    raw_id=int(raw_id),
                    center=(cx, cy),
                    frame_idx=frame_idx,
                    class_name=class_name,
                )

                # If not confirmed yet, show small pending marker only
                if target_id is None:
                    pending_text = f"pending raw:{raw_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, 1)
                    draw_text_box(frame, pending_text, x1, y1 - 10, GRAY, scale=0.45, thickness=1)
                    continue

                active_this_frame.add(target_id)

                mem = target_memory[target_id]
                mem["last_center"] = (cx, cy)
                mem["last_frame"] = frame_idx
                mem["class_name"] = class_name
                mem["last_conf"] = float(conf)
                mem["trail"].append((int(cx), int(cy)))

                if len(mem["trail"]) > 80:
                    mem["trail"] = mem["trail"][-80:]

                color = target_color(target_id)

                # Main bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Main label
                label = f"{target_id}  {class_name}  {conf:.2f}"
                draw_text_box(frame, label, x1, y1 - 12, color, scale=0.72, thickness=2)

                # Raw tracker ID, smaller and less important
                raw_label = f"raw:{raw_id}"
                draw_text_box(frame, raw_label, x1, y2 + 24, (80, 80, 80), scale=0.48, thickness=1)

                # Trajectory line
                trail = mem["trail"]

                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], color, 3)

                if trail:
                    cv2.circle(frame, trail[-1], 6, color, -1)

                csv_writer.writerow([
                    frame_idx,
                    round(frame_idx / fps, 3),
                    target_id,
                    int(raw_id),
                    class_name,
                    round(float(conf), 4),
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    round(cx, 2),
                    round(cy, 2),
                    "active",
                ])

        # ============================================================
        # Small dashboard overlay
        # ============================================================

        active_count = len(active_this_frame)
        total_targets = len(target_memory)

        cv2.rectangle(frame, (10, 10), (650, 115), BLACK, -1)
        cv2.rectangle(frame, (10, 10), (650, 115), (255, 255, 255), 1)

        cv2.putText(
            frame,
            "UAV CLEAN TARGET TRACKING",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            WHITE,
            2,
        )

        cv2.putText(
            frame,
            f"Active targets: {active_count}   Saved targets: {total_targets}   Frame: {frame_idx}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "Clean IDs: Target_1 / Target_2 / Target_3   |   Raw tracker IDs shown small in gray",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            GRAY,
            1,
        )

        writer.write(frame)
        cv2.imshow("Clean Target Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if writer is not None:
        writer.release()

    csv_file.close()
    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Saved video: {out_video}")
    print(f"Saved CSV:   {out_csv}")


if __name__ == "__main__":
    main()