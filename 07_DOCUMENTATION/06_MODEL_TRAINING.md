# 06 - Model Training

## Overview

The object detection models used by the perception pipeline were trained before deployment using custom datasets and the Ultralytics YOLO training framework.

Training is the process during which the neural network learns to recognize different object categories by analysing thousands of labelled images. Once training is complete, the resulting model can be deployed to perform real-time inference on previously unseen data.

The training process was performed separately from the operational system using Google Colab with GPU acceleration.

---

## Training Pipeline

The complete training workflow consists of several sequential stages:

1. Dataset preparation and organization.
2. Image annotation using bounding boxes.
3. Dataset splitting into training, validation and testing subsets.
4. Model training using the Ultralytics framework.
5. Performance evaluation.
6. Exporting the final model for deployment.

Each stage contributes to the final performance of the detector and directly influences its accuracy and robustness.

---

## Model Selection

The project uses YOLO as the primary object detection architecture due to its balance between detection accuracy and real-time performance.

Different models were trained depending on the application, including general surveillance models and specialized military vehicle detectors.

After training, the selected model is exported and integrated into the Windows AI perception pipeline for inference.

---

## Performance Evaluation

During training, multiple performance metrics are monitored to evaluate model quality.

These include training and validation losses, precision, recall and mean Average Precision (mAP).

The evaluation process helps determine whether the model is learning effectively and whether additional training or dataset improvements are required.
