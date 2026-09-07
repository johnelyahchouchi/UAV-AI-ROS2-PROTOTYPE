# Training Recipes and Provenance

This directory keeps reproducible configuration, scripts, and lightweight experiment
metadata. Trained checkpoints, datasets, plots, batch previews, and complete run folders
are local/external artifacts and do not belong in source Git.

## Layout

- `scripts/`: explicit historical detector training recipes.
- `configs/`: original records plus portable dataset configurations.
- `detection_runs/`: retained `args.yaml` and `results.csv` for detector experiments.
- `classification_runs/`: retained `args.yaml` and `results.csv` for classifier experiments.

Ultralytics recreates `results.png`, confusion matrices, metric curves, labels previews,
training batches, validation batches, and `weights/`. Root `.gitignore` excludes those
generated artifacts while allowing lightweight provenance already tracked here.

## Environment

Use the controlled Python environment and provide external inputs explicitly:

```powershell
$env:UAV_YOLO_PYTHON = "D:\path\to\UAV_YOLO_ENV\Scripts\python.exe"
$env:UAV_BASE_MODEL_PATH = "D:\models\trusted-base.pt"
$env:UAV_TRAINING_DATA_YAML = "D:\datasets\experiment\data.yaml"
$env:UAV_TRAINING_OUTPUT_DIR = "D:\uav-training-runs"
& $env:UAV_YOLO_PYTHON .\05_TRAINING\scripts\train_kaggle_military_v1.py
```

The base checkpoint must be SHA-256 allowlisted before the shared trusted loader will
deserialize it. Dataset lineage and deployment decisions are documented in
`04_DATASET_ENGINEERING/inventory/DATASET_TO_MODEL_LINEAGE.md` and
`00_PROJECT_GUIDE/MODEL_REGISTRY.md`.

Committed dataset YAMLs are templates containing `<external-dataset-root>`. Copy the
selected YAML outside the repository, replace that placeholder, and pass the copy via
`UAV_TRAINING_DATA_YAML`. Dataset cleaners write runnable `data.yaml` files alongside
their generated external datasets instead of writing local paths into source files.

The similarly named BTR and tank-classifier run folders remain because their
`args.yaml` and `results.csv` files differ and preserve distinct historical
hyperparameters/results. They are research provenance, not alternative launch targets.
