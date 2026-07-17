# Detection Dataset Configuration Registry

## AMAD5 aerial detector

Historical configuration:

`original_training_configs/amad5_aerial_original.yaml`

Status:

The original training run referenced this configuration through a relative path. The configuration has now been preserved inside the clean project.

## BTR detector

Historical configuration:

`original_training_configs/btr_original.yaml`

Original dataset location:

`C:\Users\UAVlab\Desktop\uav_ai_company\BTR.v1i.yolov8`

Status:

Dataset currently exists, but it remains tied to the old project directory.

## Military Kaggle detector

Historical configuration:

`original_training_configs/military_kaggle_original.yaml`

Original dataset location:

`C:\Users\UAVlab\Desktop\uav_ai_company\big_datasets\01_kaggle_military_assets\military_object_dataset`

Status:

Dataset currently exists, but it remains tied to the old project directory.

## Rule

Original YAML files are preserved unchanged for traceability.

Future reproducible configurations must use documented dataset-root variables or clean dataset-master locations instead of depending on the old `uav_ai_company` folder.
