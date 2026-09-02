# Portable project paths

Application code derives the repository root from source-file locations. Run commands
from the repository root unless a subsystem README says otherwise; the repository does
not need to live on the Desktop or under a particular username.

Repository defaults are defined in `project_paths.py`. Large or machine-specific
resources can remain outside Git and be selected with these environment variables:

| Variable | Purpose |
|---|---|
| `UAV_MODELS_DIR` | Root for repository model-layout defaults |
| `UAV_MODEL_PATH` | Active detector `.pt` file |
| `UAV_BASE_MODEL_PATH` | Base YOLO `.pt` file used by training/general modes |
| `UAV_KAGGLE_BASE_MODEL_PATH` | YOLOv8s base model used by Kaggle training |
| `UAV_BTR_MODEL_PATH` | BTR detector `.pt` file |
| `UAV_TANK_CLASSIFIER_PATH` | Active tank classifier `.pt` file |
| `UAV_ARMORED_CLASSIFIER_PATH` | Active armored-vehicle classifier `.pt` file |
| `UAV_ARTILLERY_CLASSIFIER_PATH` | Active artillery classifier `.pt` file |
| `UAV_DATASETS_ROOT` | Parent of the default dataset layout |
| `UAV_TANK_RECOGNITION_DATASET_DIR` | Tank-recognition master dataset |
| `UAV_BTR_DATASET_DIR` | BTR YOLO dataset |
| `UAV_KAGGLE_DATASET_DIR` | Kaggle military YOLO dataset |
| `UAV_AMAD5_DATASET_DIR` | Original AMAD5 dataset |
| `UAV_AMAD5_CLEAN_DATASET_DIR` | Clean AMAD5 dataset |
| `UAV_ROBOFLOW_MILITARY_DATASET_DIR` | Roboflow military-footage dataset |
| `UAV_ROBOFLOW_TANK_DATASET_DIR` | Original Roboflow tank dataset |
| `UAV_ROBOFLOW_TANK_CLEAN_DATASET_DIR` | Clean Roboflow tank dataset |
| `UAV_MULTICLASS_DATASET_DIR` | Frame-extraction/selection workspace |
| `UAV_DATASET_EXTRACT_DIR` | Temporary extracted-dataset cache |
| `UAV_TEST_MEDIA_DIR` | External or repository-local test media root |
| `UAV_DEFAULT_VIDEO_PATH` | General-surveillance video |
| `UAV_TANK_VIDEO_PATH` | Tank/BTR test video |
| `UAV_BTR_DEMO_IMAGES_DIR` | Source images for the BTR demo-video builder |
| `UAV_BTR_DEMO_VIDEO_PATH` | Generated BTR demo video |
| `UAV_EQUIPMENT_ZIP_PATH` | Equipment dataset ZIP inspected by the ZIP tool |
| `UAV_ARTILLERY_ZIP_PATH` | Specific artillery dataset ZIP |
| `UAV_ARTILLERY_ZIPS_DIR` | Directory scanned for artillery dataset ZIPs |
| `UAV_OUTPUTS_DIR` | Common output root |
| `UAV_AGENTIC_OUTPUT_DIR` | Mission-state adapter output directory |
| `UAV_TRAINING_OUTPUT_DIR` | Ultralytics training-run output root |
| `UAV_YOLO_PYTHON` | Python executable for the protected YOLO environment |

Relative override values are anchored to the repository root. Absolute Windows and
Linux values are accepted. Required resources fail with a message naming the relevant
environment variable; scripts do not guess a username-specific fallback.

Checked-in dataset YAMLs use paths relative to their own configuration directory and
expect the default dataset layout below `datasets/`. Training scripts use the variables
above when datasets live elsewhere.

Historical inventory CSVs, original-path audit reports, and generated Ultralytics
`args.yaml` files intentionally retain the paths recorded at the time they were
produced. They are evidence, not runtime configuration.
