# GitHub reference audit

Audit date: 2026-08-11

This audit records concepts considered for UAV detector uncertainty work. Repository
HEADs were checked before implementation. V1 does not copy or substantially adapt code
from any repository below; it independently implements small generic perturbation, IoU
matching, and descriptive-statistics routines.

## Prakritijain11/mcdo-yolo-uncertainty-estimation

- Repository: https://github.com/Prakritijain11/mcdo-yolo-uncertainty-estimation
- Audited HEAD: 149a2b96d924fe6decf912cc08ae1dff0815c40e
- License: MIT, copyright 2025 Prakriti Jain. The repository notes that its
  imagecorruptions dependency is separately Apache-2.0 licensed.

### Purpose and YOLO generation

This is a YOLOv8/Ultralytics master's-thesis repository studying detection degradation
under 19 corruption types and Monte Carlo Dropout at dropout rates 0.1, 0.5, and 0.7.
Its configuration replaces selected YOLOv8 convolution blocks with ConvMCDO blocks. It
also contains a CustomYOLO wrapper intended to inject dropout into an existing model.

### Framework and uncertainty method

- Framework: Python, PyTorch, Ultralytics YOLOv8.
- Corruption testing: imagecorruptions across multiple families and severity levels.
- Model uncertainty: repeated forward passes with Dropout2d active at inference.
- Evaluation: variance-oriented analysis and Probabilistic Detection Quality concepts.

### Verified implementation findings

The useful ConvMCDO design makes dropout an explicit part of forward(): convolution,
BatchNorm, activation, then Dropout2d. That is the relevant architectural principle for
a future MCDO layer: a module only affects computation when forward() actually invokes
it.

The separate CustomYOLO pattern should not be copied:

- _add_dropout_layers() registers Dropout2d as a child of each nn.Conv2d with
  module.add_module(). PyTorch nn.Conv2d.forward() does not automatically execute newly
  registered child modules, so this alone does not insert dropout into the data path.
- predict() calls self.model.train(), and the inference wrapper also calls train(). That
  changes BatchNorm and all other training-sensitive modules, rather than selectively
  enabling dropout while the network remains in evaluation mode.
- calculate_uncertainty() directly builds an array of boxes and applies np.var along the
  sample axis. It has no target association step and therefore assumes compatible box
  counts and ordering across stochastic samples.
- The configuration and wrapper are tied to custom Ultralytics model parsing and to a
  specific YOLOv8-era architecture. The ConvMCDO constructor also uses ch_out for both
  input and output channels, which is not a generic drop-in convolution signature.

### Useful concepts

- Explicit dropout in the forward path.
- Controlled corruption experiments.
- Multiple stochastic samples rather than treating one confidence as uncertainty.
- Separate localization and classification behavior.
- PDQ as a possible future evaluation concept when probabilistic detections and ground
  truth are available.

### What V1 reuses and rejects

V1 reuses only the general research ideas of controlled corruptions, repeated sampling,
and separate output dimensions. It rejects custom model surgery, model-wide train mode,
direct variance over unassociated boxes, broad corruption severity, SciPy/Seaborn/data
science dependencies, and any training workflow.

### Licensing implications

MIT permits reuse with preservation of its copyright and permission notice in copied or
substantially adapted portions. No source was copied or substantially adapted, so V1 has
no embedded third-party code notice. This document retains conceptual attribution.

## tjiagoM/stochastic-YOLO

- Repository: https://github.com/tjiagoM/stochastic-YOLO
- Audited HEAD: 25775d871026a84e82321777b483bec834086707
- Paper: https://arxiv.org/abs/2009.02967
- License: Apache License 2.0.

### Purpose and YOLO generation

Stochastic-YOLO adapts an old Ultralytics YOLOv3 fork for probabilistic object detection
under dataset shift. Its Darknet-style configurations add dropout at specified layer
IDs. The repository fine-tunes multiple dropout rates and evaluates COCO data under
corruptions and severities.

### Framework and uncertainty method

- Framework: legacy PyTorch Ultralytics YOLOv3 code plus Darknet-style cfg files.
- Model uncertainty: Monte Carlo Dropout with configurable inference sample counts.
- Evaluation: label and spatial uncertainty, PDQ-related evaluation, corruption sweeps,
  ensembles, and sensitivity studies.
- Efficiency: an optional cached MCDrop path reuses computation before the first
  stochastic layer.

### Useful concepts

- Treat classification (what) and localization (where) uncertainty separately.
- Evaluate behavior across controlled dataset shift, not only clean data.
- Compare dropout placement/rate/sample-count sensitivity.
- Cache deterministic early features when stochastic layers are late in the network.
- Evaluate uncertainty quality separately from standard mAP.

### Compatibility issues

The implementation is built around a historical YOLOv3 fork, custom cfg files, COCO
submodules, PDQ tooling, and training/evaluation scripts. Porting it would replace or
fork the current Ultralytics detector architecture, introduce substantial dependencies,
and conflict with V1's no-training/no-weight-change boundary.

### What V1 reuses and rejects

V1 reuses the experimental concepts of repeated samples, corruption testing, separate
class/box metrics, and possible future caching. It rejects the YOLOv3 architecture,
dropout training, cfg layer IDs, bundled COCO/PDQ infrastructure, ensemble workflow, and
direct integration with the active detector.

### Licensing implications

Apache-2.0 requires preservation of license/notice terms and marking changes when code is
distributed in modified form. No Stochastic-YOLO source is included or adapted in V1.
Conceptual attribution and the paper link are recorded here.

## flkraus/bayesian-yolov3

- Repository: https://github.com/flkraus/bayesian-yolov3
- Audited HEAD: f9faa718542c3dd657f5acb23b6642f399c63645
- Paper: https://arxiv.org/abs/1905.10296
- License: MIT.

### Purpose and YOLO generation

This repository accompanies “Uncertainty Estimation in One-Stage Object Detection.” It
implements YOLOv3 with uncertainty estimation in TensorFlow and supplies separate
standard, aleatoric, and epistemic inference scripts plus uncertainty training.

### Framework and uncertainty method

- Framework: TensorFlow, custom YOLOv3 implementation, TFRecord input.
- Aleatoric uncertainty: learned observation/localization uncertainty through modified
  outputs and loss.
- Epistemic uncertainty: Monte Carlo Dropout and repeated inference.
- Evaluation context: automotive pedestrian detection, NMS, localization accuracy, and
  occlusion behavior.

### Useful concepts

- A one-stage detector has distinct classification and localization uncertainty.
- Uncertainty should be checked against accuracy/IoU and occlusion rather than interpreted
  in isolation.
- Epistemic and aleatoric uncertainty are different scientific questions and require
  different model/evaluation designs.

### Compatibility issues

The code is TensorFlow YOLOv3 rather than the repository's current PyTorch Ultralytics
detector. It requires training, custom model outputs/losses, TFRecords, and old YOLO/NMS
assumptions. It cannot be a drop-in adapter around existing .pt weights.

### What V1 reuses and rejects

V1 reuses only the theoretical separation of class and location behavior. It rejects the
TensorFlow implementation, modified loss/output tensors, training workflow, dataset
format, and direct port of the architecture.

### Licensing implications

MIT would permit adaptation with its notice preserved, but no source is copied or
substantially adapted. The repository and paper remain theoretical references.

## Why V1 is independently implemented

The requested V1 asks a narrower question than all three repositories: how stable is the
current unchanged detector when one UAV image receives mild non-geometric quality
variations? It does not claim Bayesian epistemic or aleatoric uncertainty.

Independent implementation is preferable because it:

1. preserves current weights, Ultralytics behavior, sender, tracker, threat logic, ROS 2,
   and dashboards;
2. needs only five small OpenCV/NumPy transforms and transparent statistics;
3. inserts a detector-independent Detection boundary so core tests need no model stack;
4. fixes the essential box-association problem before calculating variance or entropy;
5. avoids old YOLOv3 forks, TensorFlow, SciPy/Hungarian assignment, imagecorruptions,
   Seaborn, PDQ submodules, and training dependencies;
6. keeps future true MCDO as a separately reviewed adapter/model research phase.

## Future V2 design notes — not implemented

For a future true Monte Carlo Dropout investigation:

- keep the entire network in eval mode;
- keep BatchNorm in eval mode;
- activate only selected dropout modules;
- verify dropout is explicitly executed in forward paths;
- investigate late neck/detection-head placement;
- associate detections before any sample variance calculation;
- compare classification and localization uncertainty separately;
- test N = 5, 10, 20, and 30;
- measure latency and investigate caching deterministic features before the first
  stochastic layer.
