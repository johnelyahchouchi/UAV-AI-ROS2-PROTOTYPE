# UAV Military Detection Model v4 Plan

## Current Goal
Build a stronger UAV-compatible military target detection model using large public datasets and our own custom drone/demo frames.

## Current Models
- btr_best_v2.pt
  - Small custom BTR detector
  - Good for BTR-style demo
  - Limited generalization

- military_kaggle_v1.pt
  - Training from Kaggle 12-class military dataset
  - First broad military object detector

## Dataset Sources

### 01_kaggle_military_assets
Broad military dataset.
Classes include:
- military_tank
- military_truck
- military_vehicle
- civilian_vehicle
- soldier
- weapon
- military_aircraft
- military_warship

### 02_roboflow_military_vehicle
Military footage recognition dataset.
Classes include:
- artillery
- car
- explosion
- military_truck
- military_vehicle
- person
- tank
- truck

### 03_roboflow_tank
Tank/army vehicle dataset.
Cleaned into:
- military_vehicle

### 04_aerial_vehicle_datasets
AMAD-5 aerial military dataset.
Important for UAV/drone view.
Classes:
- military_tank
- military_vehicle
- civilian
- soldier
- civilian_vehicle

### 05_our_custom_drone_frames
Our own selected frames from demo videos.
Status:
- 206 selected images
- Needs auto-labeling or pseudo-labeling

## Recommended Final Class Mapping

For main detector v4:

- military_vehicle
  - military_tank
  - tank
  - military_vehicle
  - military_truck
  - artillery when vehicle-mounted

- civilian_vehicle
  - civilian_vehicle
  - car
  - truck

- person
  - soldier
  - person
  - civilian

Optional ignored classes:
- weapon
- explosion
- trench
- military_aircraft
- military_warship

## Final Architecture

Main detector:
- detects military_vehicle, civilian_vehicle, person

Tracker:
- assigns IDs such as Target_1, Target_2, Target_3

Dashboard:
- shows target ID, confidence, trajectory, target memory

Submodel later:
- classifies cropped military target into:
  - tank
  - APC/BTR
  - IFV
  - MRAP
  - military_truck
  - unknown_military_vehicle

## Next Steps
1. Finish current Kaggle 15-epoch training.
2. Test military_kaggle_v1.pt on multiTankVideo.mp4.
3. Compare with btr_best_v2.pt.
4. Build merged v4 dataset.
5. Train military_detector_v4_combined.pt.
6. Add ByteTrack IDs to Windows sender.
7. Add pinned target mode.
8. Add submodel classifier later.