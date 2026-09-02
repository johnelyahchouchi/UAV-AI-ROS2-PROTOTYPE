# Detection Dataset Configuration Registry

## AMAD5 aerial detector

Historical configuration:

`original_training_configs/amad5_aerial_original.yaml`

Status:

The original training run referenced this configuration through a relative path. The configuration has now been preserved inside the clean project.

## BTR detector

Historical configuration:

`original_training_configs/btr_original.yaml`

Status:

The normalized configuration uses the repository-local `datasets/01_detection/BTR_v1`
layout. Training scripts can instead use `UAV_BTR_DATASET_DIR`.

## Military Kaggle detector

Historical configuration:

`original_training_configs/military_kaggle_original.yaml`

Status:

The normalized configuration uses the repository-local
`datasets/01_detection/military_kaggle_v1` layout. Training scripts can instead use
`UAV_KAGGLE_DATASET_DIR`.

## Rule

The original class/split metadata is preserved, while machine-local dataset roots are
normalized to repository-relative split paths.

Future reproducible configurations must use documented dataset-root variables or the
repository-local `datasets/` layout.
