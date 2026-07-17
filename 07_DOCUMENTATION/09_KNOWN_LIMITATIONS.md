# 09 - Known Limitations

## Overview

The current implementation demonstrates the feasibility of integrating artificial intelligence with a ROS 2-based robotic system for real-time perception.

However, the project represents an engineering prototype and several components can be further improved to increase robustness, scalability and operational performance.

The following limitations identify areas for future development rather than implementation errors.

---

## Threat Evaluation

The current threat evaluation module is based on predefined heuristic rules.

While suitable for demonstrating the architecture, the assigned threat levels have not been validated using operational data or quantitative performance metrics.

Future work could incorporate learned decision models or multi-sensor information to improve the reliability of threat assessment.

---

## Object Tracking

The system relies on BoT-SORT to maintain object identities across consecutive frames.

Although tracking performs well under normal operating conditions, temporary occlusions, rapid object motion or crowded scenes may still result in identity switches or lost tracks.

Future improvements may include stronger appearance models or sensor fusion techniques.

---

## Dataset Coverage

The performance of the detector is directly influenced by the diversity of the training dataset.

Objects, viewing angles or environmental conditions that are insufficiently represented during training may reduce detection accuracy in real-world deployments.

Expanding the dataset with additional environments and target variations would improve model generalization.

---

## System Scalability

The current architecture has been designed and validated as a prototype for a limited number of video streams.

Deployments involving multiple UAVs or higher data rates may require distributed processing, optimized communication protocols and additional computational resources to maintain real-time performance.

---

## Future Improvements

The modular architecture of the project allows individual components to be upgraded independently.

Future developments may include improved detection models, advanced threat assessment, additional sensors, optimized communication mechanisms and tighter integration with autonomous robotic behaviors.
