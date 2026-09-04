import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.environ.get("UAV_DATASET_ROOT", PROJECT_ROOT / "04_DATASET_ENGINEERING" / "local_data"))
RAW_DIR = DATASET_DIR / "00_raw_by_class"

NEW_CLASSES = [
    "artillery_cannon",
    "rocket_launcher_grad",
    "rocket_launcher_smerch",
    "mlrs_unknown",
    "unknown_artillery",
]


def main():
    print("\nCREATING ARTILLERY / LAUNCHER RAW CLASS FOLDERS")
    print("=" * 80)

    for class_name in NEW_CLASSES:
        folder = RAW_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {folder}")

    print("=" * 80)
    print("Done. New artillery folders are inside:")
    print(RAW_DIR)


if __name__ == "__main__":
    main()
