# 07 - Dataset Engineering

## Overview

The quality of an object detection model depends primarily on the quality of the training dataset.

Before any model training takes place, images must be collected, organized and annotated to accurately represent the objects and environments that the detector is expected to recognize.

For this project, multiple datasets were combined and refined to improve the detection of surveillance targets and military vehicles.

---

## Dataset Collection

The training data was gathered from multiple publicly available datasets covering different object categories.

Using several datasets increased the diversity of training examples and improved the model's ability to generalize to new environments.

Specialized datasets were also incorporated to improve the recognition of military vehicles that are not commonly represented in general-purpose detection datasets.

---

## Data Annotation

Every image used for training requires annotations describing the location and category of each object.

Bounding boxes were created using the YOLO annotation format, where each object is represented by its class identifier and normalized bounding box coordinates.

These annotations serve as the ground truth during training, allowing the model to compare its predictions with the expected results and update its parameters accordingly.

---

## Dataset Organization

The dataset was organized following the standard Ultralytics directory structure, separating images and labels into training, validation and testing subsets.

This separation ensures that model evaluation is performed using images that were not seen during training, providing a more reliable estimate of real-world performance.

---

## Dataset Quality

Dataset quality has a direct impact on model performance.

Incorrect annotations, missing labels, duplicated images or class imbalance can significantly reduce detection accuracy and increase false predictions.

Careful dataset preparation is therefore considered one of the most important stages of the entire machine learning pipeline.
