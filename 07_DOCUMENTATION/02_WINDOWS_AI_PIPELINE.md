# 02 - Windows AI Pipeline

## Overview

The Windows AI application is responsible for transforming a live video stream into structured detection data that can be shared with the ROS 2 environment.

It performs the complete perception pipeline, including image acquisition, object detection, object tracking, threat evaluation and data transmission.

By the end of this process, each frame has been converted from raw pixel data into meaningful information describing the observed scene.

---

## Image Acquisition

The perception pipeline begins by receiving a continuous stream of images from the UAV camera.

Each frame is captured using OpenCV and immediately forwarded to the AI inference stage. At this point, the application has no understanding of the image content; it simply receives a matrix of pixel values representing the current scene.

The quality, resolution and frame rate of the incoming video directly influence the performance and accuracy of the entire perception pipeline.

---

## Object Detection

Each captured frame is analysed using a YOLO object detection model.

The detector identifies objects of interest and predicts their class, confidence score and bounding box coordinates.

Unlike traditional computer vision techniques based on manually designed features, YOLO learns visual patterns directly from training data, allowing it to recognise complex objects in real time.

---

## Object Tracking

Object detection processes every frame independently and does not preserve object identity over time.

To maintain persistent target identities, the project uses the BoT-SORT tracking algorithm.

The tracker associates detections across consecutive frames, assigns unique IDs and estimates object motion, allowing the system to follow individual targets even while they are moving.

---

## Threat Evaluation

After tracking, each detected object is evaluated using a simple threat estimation strategy.

The current implementation combines detection information with application-specific heuristics to assign a threat level that can assist the operator during surveillance.

Although intentionally simple, this stage demonstrates how additional decision-making logic can be integrated into the perception pipeline.

---

## Data Serialization

Once processing is complete, the detection results are converted into a structured JSON representation.

Each message contains the information required by the ROS 2 environment, including object identifiers, classes, confidence scores, tracking IDs and threat information.

Using a structured format simplifies communication between independent software components while allowing the protocol to evolve over time.

---

## TCP Transmission

The serialized detection data is transmitted to the Ubuntu ROS 2 computer through a TCP socket.

TCP provides reliable, ordered communication, ensuring that complete detection messages arrive without corruption before being published inside the ROS 2 ecosystem.

At this stage, the perception pipeline has completed its task and hands the processed information to the robotics layer.
