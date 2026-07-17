# 10 - Development Roadmap

## Overview

The current implementation establishes a complete AI perception pipeline capable of detecting, tracking and distributing object information within a ROS 2 environment.

The modular architecture was intentionally designed to simplify future improvements. As new algorithms, sensors or robotic capabilities become available, they can be integrated with minimal impact on the existing software components.

The following roadmap outlines potential directions for future development.

---

## Improved AI Models

Future work may focus on training larger and more specialized detection models using expanded datasets containing additional vehicle categories, environmental conditions and imaging perspectives.

Model optimization techniques such as TensorRT or ONNX Runtime acceleration could also be incorporated to improve inference performance on embedded hardware.

---

## Enhanced Threat Assessment

The current rule-based threat evaluation module could be replaced by more advanced decision-making algorithms.

Future implementations may consider object motion, trajectory prediction, mission objectives and information from multiple sensors to generate more accurate threat assessments.

---

## Multi-UAV Support

Although the current prototype demonstrates the perception pipeline using a limited number of UAVs, the architecture can be extended to support larger UAV fleets.

Future developments may include coordinated perception, shared target tracking and collaborative decision-making between multiple aerial platforms.

---

## Sensor Fusion

Future versions of the system may combine camera data with additional sensors such as LiDAR, radar, thermal cameras or GNSS information.

Combining multiple sensing modalities would improve perception robustness in challenging environments where a single sensor may not provide sufficient information.

---

## Autonomous Behaviors

The perception pipeline can serve as the foundation for higher-level autonomous capabilities.

Future robotic modules could use the published perception data for autonomous navigation, target following, obstacle avoidance or mission planning without requiring modifications to the perception system itself.

---

## Conclusion

The project demonstrates a modular architecture that separates perception, communication and robotic integration into independent software components.

This design allows future improvements to be incorporated incrementally while preserving the overall system architecture, providing a flexible foundation for continued research and development.
