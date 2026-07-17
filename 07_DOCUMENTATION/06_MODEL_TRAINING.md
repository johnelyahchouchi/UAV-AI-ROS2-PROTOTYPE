# 06 - Threat Evaluation

## Overview

Once objects have been detected and tracked, the perception pipeline performs an additional analysis to estimate their relative importance.

This stage introduces application-specific decision logic that assigns a threat level to each detected target based on predefined evaluation criteria.

Rather than replacing human decision-making, the objective is to prioritize information and help operators identify potentially important targets more efficiently.

---

## Threat Assessment

The current implementation evaluates each tracked object using a simple heuristic approach.

Information produced during the perception stage, such as the detected class and the apparent size of the object, is combined to estimate a threat level.

The resulting value is attached to every tracked object before transmission to the ROS 2 environment.

---

## Design Considerations

The threat evaluation module is intentionally separated from the object detection stage.

This modular design allows the perception system to remain independent of application-specific decision logic, making it possible to replace or improve the threat evaluation strategy without modifying the detection or tracking algorithms.

Future versions may incorporate additional information such as object velocity, trajectory, distance estimation or external sensor data to produce more advanced threat assessments.
