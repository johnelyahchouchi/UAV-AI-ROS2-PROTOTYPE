# Detection Dataset Configuration Registry

All committed YAML files are portable templates. Replace
`<external-dataset-root>` with an approved local dataset root in an external copy
before training; do not commit the machine-specific copy.

## AMAD5 aerial detector

Historical configuration:

`original_training_configs/amad5_aerial_original.yaml`

Status:

The original class mapping and split layout are preserved, while the former
machine-specific root has been sanitized.

## BTR detector

Historical configuration:

`original_training_configs/btr_original.yaml`

Original dataset location:

`<legacy-project-root>\BTR.v1i.yolov8`

Status:

Dataset currently exists, but it remains tied to the old project directory.

## Military Kaggle detector

Historical configuration:

`original_training_configs/military_kaggle_original.yaml`

Original dataset location:

`<legacy-project-root>\big_datasets\01_kaggle_military_assets\military_object_dataset`

Status:

Dataset currently exists, but it remains tied to the old project directory.

## Rule

Original YAML files preserve their class mappings and split layouts for traceability;
machine-specific root paths are represented by placeholders.

Future reproducible configurations must use documented dataset-root variables or clean dataset-master locations instead of depending on the old `uav_ai_company` folder.
