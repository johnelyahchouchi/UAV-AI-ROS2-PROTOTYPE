# Dataset-to-Model Lineage

This document records how each deployed model was produced.

---

## 1. General Military Detector

### Deployed model

`military_kaggle_v1.pt`

### Training run

`05_TRAINING/detection_runs/military_kaggle_yolov8s_v1`

### Base model

`yolov8s.pt`

### Task

Object detection.

### Purpose

Detects broad military target categories in complete video frames before tracking and specialized classification.

### Verified relationship

The deployed model is an exact SHA-256 match with:

`training_runs/military_kaggle_yolov8s_v1/weights/best.pt`

### Dataset source

Requires final documentation and confirmation from the original Kaggle dataset and training configuration.

---

## 2. Tank-Type Classifier V3

### Deployed model

`tank_type_classifier_v3_only_tanks.pt`

### Dataset builder

`builders/build_tank_type_classifier_dataset_v3.py`

### Final dataset

`01_tank_type_classifier_dataset_v3`

### Training run

`05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks-2`

### Base model

`yolov8s-cls.pt`

### Task

Image classification.

### Classes

- tank_t72
- tank_t80
- tank_t90
- tank_m1_abrams
- tank_leopard2
- tank_merkava

### Verified relationship

The deployed model is an exact SHA-256 match with:

`runs/classify/tank_type_classifier_v3_only_tanks-2/weights/best.pt`

### Deployment decision

V3 was retained because it behaved more reliably on realistic video than V4.

---

## 3. Armored-Vehicle Classifier V1

### Deployed model

`armored_vehicle_classifier_v1.pt`

### Dataset builder

`builders/build_armored_vehicle_classifier_dataset_v1.py`

### Final dataset

`01_armored_vehicle_classifier_dataset_v1`

### Training run

`05_TRAINING/classification_runs/armored_vehicle_classifier_v1`

### Base model

`yolov8s-cls.pt`

### Task

Image classification.

### Classes

- ifv_bmp
- military_truck
- armored_truck

### Verified relationship

The deployed model is an exact SHA-256 match with:

`runs/classify/armored_vehicle_classifier_v1/weights/best.pt`

---

## 4. Artillery and Launcher Classifier V1

### Deployed model

`artillery_launcher_classifier_v1.pt`

### Dataset inspection scripts

- `inspectors/inspect_yolo_zip_classes.py`
- `inspectors/inspect_artillery_keywords_in_zip.py`
- `inspectors/inspect_all_artillery_zips.py`

### Import script

`importers/import_artillery_launcher_labeled_zips_v1.py`

### Dataset builder

`builders/build_artillery_launcher_classifier_dataset_v1.py`

### Final dataset

`01_artillery_launcher_classifier_dataset_v1`

### Training run

`05_TRAINING/classification_runs/artillery_launcher_classifier_v1`

### Base model

`yolov8s-cls.pt`

### Task

Image classification.

### Classes

- rocket_launcher_grad
- mlrs_unknown
- self_propelled_artillery
- unknown_artillery

### Verified relationship

The deployed model is an exact SHA-256 match with:

`runs/classify/artillery_launcher_classifier_v1/weights/best.pt`

---

## 5. Tank Classifier V4 — Experimental

### Model

`tank_type_classifier_v4_safe_unknown.pt`

### Dataset builder

`builders/build_tank_type_classifier_dataset_v4_safe_unknown.py`

### Training run

`05_TRAINING/classification_runs/tank_type_classifier_v4_safe_unknown`

### Purpose

Added an explicit unknown-tank class.

### Status

Experimental and not approved for deployment.

### Reason

The model produced too many unknown predictions on realistic video.

---

## 6. Older Tank-Platform Experiments

### Models

- `tank_platform_classifier_v0.pt`
- `tank_platform_classifier_v1_exact_types.pt`
- `tank_platform_classifier_v2_exact_focus.pt`

### Training runs

- `tank_platform_classifier_v0`
- `tank_platform_classifier_v1_exact_types`
- `tank_platform_classifier_v2_exact_focus`

### Status

Historical experiments preserved for comparison.

They are not current deployment models.

---

## 7. BTR Experiments

### Models

- `btr_best.pt`
- `btr_best_v2.pt`

### Training runs

- `btr_yolov8n_local_test`
- `btr_yolov8n_v2_50epochs`

### Status

Experimental detector models.

### Documentation still required

- exact source dataset;
- final class definitions;
- comparison between V1 and V2;
- reason neither model became the main deployment detector.

---

## Rule for Future Models

Every new model must document:

1. Source dataset.
2. Dataset version.
3. Class definitions.
4. Inspection and cleaning scripts.
5. Builder script.
6. Training command.
7. Base pretrained model.
8. Training run directory.
9. Evaluation metrics.
10. Real-video results.
11. Deployment or rejection decision.
12. SHA-256 file hash.
