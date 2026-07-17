# 00 - Project Overview

## Introduction

During my internship at ADDITESS Ltd., I worked on the development of an AI perception pipeline for UAV applications.

The main idea behind the project is simple: instead of sending only a live video stream to the operator, the UAV should also understand what it is looking at and provide useful information in real time.

For example, rather than forcing an operator to continuously watch several video feeds, the system should be able to automatically detect objects, keep track of them over time, evaluate their importance, and share that information with the rest of the robotic system.

To achieve this, the project combines several technologies including computer vision, deep learning, object tracking, TCP communication and ROS 2.

Although each component can work independently, the real objective is to integrate them into a single pipeline that transforms raw camera images into meaningful information that can support human operators and future autonomous systems.

---

## Why this project?

A camera alone is not intelligent.

It simply captures images and sends them to the computer. Every frame is nothing more than a collection of millions of pixels containing colour information.

Humans can immediately recognise a vehicle, a pedestrian or another drone because our brains have learned to interpret those pixels. A computer, however, only sees numbers until an algorithm gives those numbers meaning.

This is where artificial intelligence becomes useful.

Instead of asking an operator to constantly analyse live video, the AI system automatically processes every frame, detects the objects that appear, tracks them as they move and sends structured information to the rest of the system.

The goal is not to replace the operator, but to reduce workload, improve situational awareness and make important information immediately available.

---

## Project Goal

The final objective is to build a complete perception pipeline capable of:

- Receiving live video from a UAV camera.
- Detecting objects of interest.
- Tracking each object over time.
- Estimating a basic threat level.
- Transmitting the information over the network.
- Publishing the results inside ROS 2.
- Displaying everything on an operator dashboard.

The result is a system that converts raw video into useful information that other robotic software can understand and use.
