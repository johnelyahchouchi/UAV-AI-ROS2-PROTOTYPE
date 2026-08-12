# UAV Model Uncertainty — Input-Perturbation Stability V1

## Problem

An Ultralytics detector can report a class and confidence for one image, but that
single confidence does not show whether the result survives small, realistic changes
in image quality. A high-confidence detection may disappear, change class, or move
substantially after mild noise or compression.

This standalone subsystem measures that robustness. It does not modify model weights,
train a model, control a UAV, publish ROS 2 messages, run BoT-SORT, or feed uncertainty
into threat scoring.

## What V1 means by Monte Carlo input perturbation

For one source image, V1 performs:

1. one clean baseline inference;
2. N seeded, mildly perturbed versions of the same image;
3. ordinary Ultralytics inference on every image;
4. class-agnostic IoU matching of detections across samples;
5. separate persistence, confidence, class, and box-stability metrics per target.

The default N is 10. Therefore the default metric denominator is 11 inference samples:
one clean baseline plus ten perturbed samples. The JSON records both counts explicitly.
In CLI terms, `--samples N` always means N perturbed samples and N + 1 total
inference samples because the clean baseline is always included.

Perturbations are non-geometric, so coordinates remain directly comparable:

- brightness gain and offset;
- contrast scaling around the per-channel mean;
- light Gaussian blur;
- low-sigma Gaussian sensor noise;
- JPEG compression at high quality.

The generator cycles through the five families in a fixed order and draws parameters
from a NumPy random generator seeded by --seed. The same image, seed, sample count, and
configuration reproduce the same variants.

## Architecture

    Source image
        |
        +-- clean baseline
        +-- seeded perturbation generator (N images)
        |
        v
    lazy Ultralytics detector adapter
        |
        v
    internal Detection dataclass
        |
        v
    deterministic class-agnostic IoU matcher
        |
        v
    transparent per-target metrics
        |
        +-- console
        +-- JSON
        +-- CSV

The core types, matcher, and metrics do not depend on Ultralytics Results, PyTorch,
model weights, a GPU, ROS 2, or a camera. Ultralytics is imported only when the live
detector adapter is constructed.

## Matching policy

V1 uses deterministic greedy highest-IoU matching with a default threshold of 0.50.
For each new sample it compares detections with the mean box of each existing target
cluster. Candidate pairs are sorted by descending IoU with stable geometric tie-breaks.
Each detection and each cluster can be assigned at most once in one sample. Unmatched
detections create new clusters.

Class equality is deliberately not required. A spatially corresponding box that changes
from tank to military_vehicle remains in the same cluster; that disagreement is measured
as classification instability.

Greedy matching is small and explainable, and is adequate for this V1. It can be less
reliable in severe crowding, large target motion, or when two objects repeatedly overlap.
Those cases should be benchmarked before considering Hungarian assignment or appearance
features.

## Metrics

Each target exposes raw metrics rather than a combined uncertainty score:

| Metric | Definition |
|---|---|
| sample_count | Clean baseline plus all perturbed inference samples |
| detection_count | Samples in which the target cluster was observed |
| detection_persistence | detection_count / sample_count |
| confidence_mean | Mean confidence over detected observations |
| confidence_std | Population standard deviation of confidence |
| class_histogram | Count of each observed class |
| class_distribution | Observed class proportions |
| dominant_class | Most frequent class; lexical tie-break |
| class_agreement | Dominant class count / detected observations |
| class_entropy_bits | Shannon entropy using log base 2 |
| bbox_center_std_pixels | Population std of center x and y |
| bbox_size_std_pixels | Population std of width and height |
| mean_iou_to_reference | Mean IoU to the clean box, or mean box if no clean observation exists |

Higher persistence, agreement, and mean IoU indicate greater observed stability under
this configured test. Lower standard deviations and entropy indicate less variation.
These values are not calibrated probabilities of correctness, do not replace labeled
validation, and are not assigned HIGH/MEDIUM/LOW reliability labels in V1.

## Environment assumptions

Reuse the existing Windows YOLO environment. Do not upgrade or replace its protected
packages. The checked-in constraints are in:

    01_WINDOWS_AI/model_test_dashboard/constraints-yolo-env.txt

The audited versions are NumPy 1.26.4, OpenCV 4.10.0.84, Ultralytics 8.4.107,
PyTorch 2.13.0+cu130, and Torchvision 0.28.0+cu130. Runtime dependencies for this
subsystem are NumPy, OpenCV, and Ultralytics; PyTorch is loaded through Ultralytics.
No new dependency file is introduced.

The pure tests use standard-library unittest and need only NumPy and OpenCV. They do
not import Ultralytics.

## Run

From the repository root in PowerShell, select the existing YOLO Python executable and
put this subsystem's src directory on PYTHONPATH:

    $Python = $env:UAV_YOLO_PYTHON
    $env:PYTHONPATH = Join-Path $PWD "08_MODEL_UNCERTAINTY\src"
    & $Python -m uav_uncertainty.mc_stability_runner `
      --model "C:\path\to\model.pt" `
      --image "C:\path\to\image.jpg" `
      --samples 10 `
      --seed 42 `
      --imgsz 960 `
      --conf 0.25 `
      --iou 0.45 `
      --match-iou 0.50 `
      --device 0

Use --device auto to let Ultralytics choose, or --device cpu for CPU inference. The
model must be an existing local detection-model .pt file; this CLI does not rely on a
model download. If the model or image resolves inside this Git repository, the CLI emits
a reminder not to commit weights, datasets, or runtime imagery; inference is not blocked.

By default, output is written under:

    08_MODEL_UNCERTAINTY/outputs/<image>_seed<seed>_n<samples>/

The directory contains summary.json and targets.csv. Runtime outputs are ignored by
Git. If either owned result file already exists, the command stops without replacing
it. Choose a different identity with --run-name or --output-dir, or explicitly pass
--overwrite to atomically replace only summary.json and targets.csv. Other files in the
run directory are preserved.

## Tests

The pure-core suite can be run with the available Python 3.12 interpreter without
installing or changing detector dependencies:

    $env:PYTHONPATH = Join-Path $PWD "08_MODEL_UNCERTAINTY\src"
    py -3.12 -m unittest discover -s .\08_MODEL_UNCERTAINTY\tests -v

## Folder structure

    08_MODEL_UNCERTAINTY/
    |-- README.md
    |-- research/github_reference_audit.md
    |-- src/uav_uncertainty/
    |   |-- detection_types.py
    |   |-- perturbations.py
    |   |-- detector_adapter.py
    |   |-- detection_matcher.py
    |   |-- uncertainty_metrics.py
    |   `-- mc_stability_runner.py
    |-- tests/
    |-- examples/README.md
    `-- outputs/.gitignore

## V1 versus future true Monte Carlo Dropout

V1 varies the input while leaving the existing detector architecture and weights
unchanged. It measures sensitivity to a defined family of image-quality changes. It is
not epistemic uncertainty from stochastic network weights and should not be described as
true Monte Carlo Dropout.

A future V2 may investigate true MCDO. The whole model must remain in eval mode so
BatchNorm remains fixed, while only selected dropout layers are activated. Research
should compare late neck/detection-head placement, classification versus localization
uncertainty, N values of 5/10/20/30, latency, and feature caching. V2 is documentation
only here; none of that behavior is implemented.

## Limitations

- Results describe robustness only to the configured mild perturbations.
- Detector/NMS thresholds influence persistence and class observations.
- Greedy IoU clustering can split or merge identities in crowded scenes.
- Pixel variation is resolution-dependent and is not physical-world localization error.
- No ground truth, calibration, PDQ, mAP, covariance model, or reliability threshold is
  calculated.
- GPU kernels and the detector stack may have their own nondeterminism even though input
  generation, matching, metrics, and serialization order are deterministic.
- Real Ultralytics 8.4.107 inference has not yet been validated in this environment
  because the documented dedicated UAV_YOLO_ENV Python executable is unavailable. This
  local validation issue does not affect the pure-core unit tests.
- This is a research diagnostic, not a safety certification or flight-control signal.

## References and roadmap

The conceptual/code audit is in research/github_reference_audit.md. V1 independently
implements its small generic perturbation, matching, and metric core; no external source
code was copied or substantially adapted.

Recommended follow-on work is to validate V1 against a labeled UAV image set with
controlled corruption sweeps, then study matching failure cases before designing V2.
Later adapters may expose results to a dashboard or ROS 2, but adapters must remain
outside the deterministic core and must not directly influence flight or threat logic.
