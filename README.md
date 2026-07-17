JOHN EL YAHCHOUCHI 
ADDITESS INTERNSHIP
The current structure includes:

00_PROJECT_GUIDE
01_WINDOWS_AI
02_ROS2_WINDOWS_MIRROR
04_DATASET_ENGINEERING
05_TRAINING
07_DOCUMENTATION

This is the base structure as first prototype that the project will continue to follow.


# UAV AI ROS2 Prototype

## Project Overview

This project implements a modular UAV perception pipeline designed to detect, track, prioritize, and transmit targets from an onboard camera to a ROS 2 environment.

The system is divided into two independent parts.

The Windows side is responsible for computer vision. It captures video frames, performs object detection using YOLO, tracks detected objects with BoT-SORT, evaluates target priority, and packages the results into a TCP message.

The Ubuntu side runs ROS 2. It receives the TCP packets, converts them into ROS messages, republishes them as ROS topics, and makes the information available to dashboards or other robotic components.

Separating the AI pipeline from the ROS ecosystem keeps both parts independent while allowing each machine to use the software stack best suited to its role.
