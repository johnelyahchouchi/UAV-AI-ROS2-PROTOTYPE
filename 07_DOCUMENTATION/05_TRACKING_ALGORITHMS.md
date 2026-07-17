# 05 - Object Detection and Tracking

## Overview

The primary objective of the perception system is to identify objects present in the incoming video stream and monitor them as they move through the scene.

This process consists of two complementary stages. First, an object detector identifies and localizes every object of interest within a single frame. Next, a tracking algorithm associates these detections across consecutive frames, allowing each object to maintain a persistent identity over time.

Together, these stages transform independent image detections into continuous target observations.

---

## Object Detection

Object detection is performed using the Ultralytics implementation of the YOLO (You Only Look Once) algorithm.

For every incoming frame, the model predicts the location of each detected object by generating bounding boxes, assigning a class label and estimating a confidence score that reflects the probability of a correct prediction.

Each frame is processed independently, meaning the detector has no knowledge of objects detected in previous frames.

---

## Object Tracking

To maintain object identities over time, the project uses the BoT-SORT multi-object tracking algorithm.

The tracker associates detections between consecutive frames and assigns a persistent tracking identifier to each target.

This allows the system to distinguish between multiple objects of the same class while maintaining temporal consistency throughout the video sequence.

---

## Detection Output

Each detected object is represented by a collection of attributes describing its current state.

These attributes include the object class, confidence score, bounding box coordinates and tracking identifier.

The resulting detection data forms the foundation for every subsequent stage of the perception pipeline, including threat evaluation, communication and visualization.
