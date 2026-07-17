# 01 - System Architecture

## Overview

The project is divided into two main environments:

- **Windows AI Computer**
- **Ubuntu ROS 2 Computer**

This separation allows GPU-intensive AI inference to run on the Windows machine while the robotic middleware and visualization tools execute on the Ubuntu ROS 2 environment.

The two systems communicate through a TCP connection, allowing detection results to be transferred in real time.

Each environment is responsible for a specific part of the perception pipeline, creating a modular architecture that can be extended or modified independently.
