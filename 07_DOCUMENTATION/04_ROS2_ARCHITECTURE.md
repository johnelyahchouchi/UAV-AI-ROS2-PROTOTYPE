# 04 - ROS 2 Architecture

## Overview

ROS 2 serves as the communication backbone of the robotic system. Rather than connecting software components through direct function calls, ROS 2 allows independent nodes to exchange information using a publish-subscribe architecture.

In this project, ROS 2 receives processed detection data from the Windows AI application and distributes it to visualization tools and other robotic modules.

This design enables multiple applications to access the same information simultaneously without modifying the perception pipeline.

---

## ROS 2 Nodes

Each major software component is implemented as an independent ROS 2 node.

Every node performs a specific task, such as receiving TCP messages, publishing detections, displaying the operator dashboard or processing additional robotic information.

Because each node operates independently, components can be started, stopped or updated without affecting the rest of the system.

---

## Topics

Communication between nodes takes place through ROS 2 topics.

A node publishes information to a topic, while one or more nodes subscribe to that topic to receive the published messages.

This publish-subscribe model removes direct dependencies between software components and improves the modularity of the overall architecture.

---

## ROS 2 Bridge

The ROS 2 bridge acts as the interface between the TCP communication layer and the ROS 2 ecosystem.

After receiving serialized detection messages from the Windows AI application, the bridge reconstructs the data and publishes it as ROS 2 messages.

From this point onward, every ROS 2 node receives the same synchronized perception data regardless of its internal implementation.

---

## Visualization

Operator dashboards subscribe to the published detection topics and visualize the current perception results.

Because dashboards obtain their information directly from ROS 2 topics, multiple visualization tools can operate simultaneously without increasing the computational load on the AI pipeline.
