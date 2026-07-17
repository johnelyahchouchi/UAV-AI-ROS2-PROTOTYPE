# Tracking Data Export Tool

## File

win_yolo_data_extractor.py

## Purpose

Runs an Ultralytics YOLO tracking model on a video and exports structured mission information.

## Main functions

- loads a YOLO detector;
- tracks detected objects;
- creates simplified persistent target IDs;
- filters military-related classes;
- calculates movement direction;
- assigns a basic threat level;
- exports structured CSV reports.

## Outputs

- frame_by_frame_detections.csv
- target_summary.csv
- mission_report.csv

## Classification

This is a Windows AI analysis/export utility.

It is not:

- a training script;
- a dataset importer;
- a ROS2 node;
- the live TCP sender.

## Future improvement

The hard-coded threat-level function should eventually be replaced by a documented and validated scoring method.
