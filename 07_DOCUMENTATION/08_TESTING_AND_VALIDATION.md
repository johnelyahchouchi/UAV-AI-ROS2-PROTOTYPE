# 08 - Testing & Validation

## Overview

Testing was performed throughout the development process to verify the functionality, reliability and integration of each component within the perception pipeline.

Rather than evaluating only the object detection model, the validation process covered the complete system, including AI inference, object tracking, TCP communication, ROS 2 integration and operator visualization.

This approach ensured that each subsystem functioned correctly both individually and as part of the overall architecture.

---

## Model Validation

Before deployment, each trained model was evaluated using the validation tools provided by the Ultralytics framework.

Performance metrics such as precision, recall and mean Average Precision (mAP) were monitored throughout training to assess detection accuracy and identify opportunities for improvement.

Models that demonstrated stable performance were exported and integrated into the operational pipeline.

---

## Integration Testing

After deployment, the interaction between the Windows AI application and the Ubuntu ROS 2 environment was verified.

Testing confirmed that detection results were correctly transmitted over TCP, reconstructed by the ROS 2 bridge and successfully published as ROS 2 topics for downstream applications.

This stage ensured that communication between software components remained reliable under normal operating conditions.

---

## System Validation

The complete perception pipeline was tested using recorded video streams and live camera feeds.

Validation focused on confirming that objects were detected, tracked consistently, assigned threat information and displayed correctly on the operator dashboards.

These tests verified the correct operation of the complete perception workflow from image acquisition to visualization.

---

## Continuous Testing

Testing was performed continuously throughout development.

Every significant modification to the AI models, communication protocol or ROS 2 integration was verified before being incorporated into the final prototype.

This iterative validation process reduced integration issues and improved the overall stability of the system.
