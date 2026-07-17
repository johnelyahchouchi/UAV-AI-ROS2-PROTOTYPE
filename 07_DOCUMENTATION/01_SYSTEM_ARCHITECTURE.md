# 01 - System Architecture

## Overview

The project is divided into two main environments:

- Windows AI Computer
- Ubuntu ROS 2 Computer

This separation allows GPU-intensive AI inference to run on the Windows machine while the robotic middleware and visualization tools execute on the Ubuntu ROS 2 environment.

The two systems communicate through a TCP connection, allowing detection results to be transferred in real time.

Each environment is responsible for a specific part of the perception pipeline, creating a modular architecture that can be extended or modified independently.

## System Components

The perception pipeline is composed of two independent software environments connected through a TCP communication channel.

The Windows environment is responsible for image processing and AI inference, while the Ubuntu environment manages robotics communication through ROS 2 and provides the interface used by the operator.

This separation allows each system to focus on a specific task, improving modularity, maintainability and future scalability.

### Windows AI Environment

The Windows application receives the live video stream, performs object detection using YOLO, tracks objects with BoT-SORT and evaluates a basic threat level for every detected target.

Once processing is complete, the detection results are serialized into a JSON structure and transmitted to the ROS 2 computer through a TCP socket.

### Ubuntu ROS 2 Environment

The Ubuntu system receives the TCP packets, reconstructs the detection data and publishes the information as ROS 2 topics.

These topics can then be consumed by dashboards, visualization tools or any additional ROS 2 node without modifying the AI pipeline itself.
