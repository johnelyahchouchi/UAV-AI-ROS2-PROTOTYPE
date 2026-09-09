# Model Uncertainty and Robustness

This subsystem provides detector-independent analysis used by the Windows live tester.
It keeps two scientifically different methods separate.

## V1 — implemented

V1 measures **input-perturbation robustness**. It runs one clean inference and a
seeded set of mild brightness, contrast, blur, sensor-noise, and JPEG variants. Boxes
are matched class-agnostically by IoU so a class change remains observable.

Per target, V1 reports independent metrics rather than a synthetic score:

- detection persistence and missing sample indices;
- confidence mean and population standard deviation;
- dominant class, class counts, evidence share, agreement, and entropy in bits;
- bounding-box center/size population standard deviations;
- mean IoU to the clean reference box for baseline targets (or to the mean observed
  box for a cluster that appears only under perturbation).

Confidence, class, box-variation, and IoU statistics are conditional on observations
where the target was detected; persistence separately represents missed samples. Mean
reference IoU includes the clean observation for baseline targets. Pixel variation is
resolution-dependent.

These values are not Bayesian uncertainty, calibrated correctness probability, or a
safety guarantee. The interpretation text only describes observed behavior under the
configured perturbations.

The live presentation labels detection persistence at `>= 0.95` as **STABLE**, from
`0.75` through `< 0.95` as **INPUT-SENSITIVE**, and below `0.75` as
**UNSTABLE / REVIEW**. These are operator-facing robustness bands, not accuracy levels
or probabilities.

Matching is deterministic, class-agnostic, one-to-one, and based on a running mean box.
Like any greedy IoU association, it can split or merge identities in crowded scenes;
review the per-sample detections when that limitation matters. Perturbation-only target
clusters are retained and surfaced because they can reveal input-induced detections.

## V2 — runtime implemented; external checkpoint required

V2 measures **MC Dropout model uncertainty**: the input is held fixed while the
validated model's internal dropout masks change. The cleaned runtime was extracted from
the retained Colab notebook and refuses any checkpoint that does not contain exactly six
late-head `Dropout2d(p=0.20)` placements across the three `cv2` and three `cv3`
prediction scales.

Inference keeps the model and every BatchNorm module in evaluation mode, activates only
those six dropout modules, preprocesses the exact frozen frame once, and performs 20
lower-level forwards by default. It does not call the high-level prediction path and it
does not inject random dropout into the normal V1 model.

Objects are associated by geometry before class identity. Each pass retains the
strongest overlapping hypothesis per class, chooses the highest-confidence class
winner, and records within-pass competition. Per target, V2 reports:

- detection persistence and missed-pass indices;
- winner distribution, agreement, entropy, confidence mean/std;
- competition count/rate and normalized confidence **evidence share**/entropy;
- center x/y and width/height population standard deviations;
- mean/minimum IoU to the mean predicted reference box.

Existence, classification, and localization states remain separate. With no detections,
class agreement, entropy, evidence, confidence, and localization statistics are `N/A`.
Reference IoU is predicted-box consistency, not ground-truth IoU. Evidence share is not
a class-correctness probability. V2 is an approximate model/epistemic uncertainty probe,
not a calibrated posterior or safety guarantee.

The V2 runtime is unavailable until `UAV_MCDO_V2_MODEL_PATH` identifies the real external
validated checkpoint and its SHA-256 is present in the trusted model registry. That
checkpoint remains outside source Git. Its independently verified digest and size are
recorded in the trusted model registry; local runtime loading still fails closed if the
selected file does not match them.

## Method comparison

| | V1 input robustness | V2 MC Dropout |
|---|---|---|
| Input | Clean frame plus mild variants | Same exact frame every pass |
| Model | Fixed | Internal validated dropout masks change |
| Question | What changes when the input changes slightly? | What changes when the model representation changes? |
| Default work | 1 clean + 10 variants | 20 same-frame stochastic forwards |
| Checkpoint | Normal trusted detector | Separate trusted dropout-capable checkpoint |

## Live integration

The normal tester performs one base YOLO inference per displayed frame. The canonical
launcher enables a unified workspace that periodically copies the latest raw frame,
runs V1 followed by available validated V2 in a background worker, and refreshes two
scientifically separate status cards. Only one cycle can be active and stale work is
not queued. Normal detection remains visible throughout.

The continuous worker adds periodic method-specific inference work. It does not change
the single-pass base playback path, but shared GPU load may reduce achieved FPS. V2
remains visibly unavailable when its validated external checkpoint is not configured.

Pressing optional `U`:

1. copies the current raw frame;
2. pauses playback/capture;
3. runs V1 directly when no validated V2 checkpoint is loaded; or displays `1` for V1
   and `2` for V2 when it is loaded;
4. displays method-specific per-target metrics and a bounded interpretation;
5. paginates readable target cards in groups of two when necessary;
6. waits for `P`, `U`, or Space to resume.

Press `A`/`D` or `[`/`]` to change detailed inspection pages and `S` to save the
visible page. Use `--continuous-uncertainty-interval` to trade refresh age against
compute load.

For a dedicated local V2 session on Windows, run
`01_WINDOWS_AI\launchers\Start_MC_Dropout_V2.bat`. It opens native pickers for the
trusted base detector, the separately trained V2 checkpoint, and an MP4, and fails
closed when V2 trust or architecture validation fails. Its V1/V2 cards update live;
`U`, then `2`, remains available for a detailed exact-frame V2 report. The checkpoint
remains external and must be enrolled using its
real independently verified SHA-256 before it can be deserialized.

## Code layout

```text
src/uav_uncertainty/
  detection.py       validated detector-independent values
  perturbations.py   deterministic V1 input variants
  matching.py        class-agnostic one-to-one IoU matching
  metrics.py         transparent per-target statistics
  analysis.py        in-memory orchestration
  presentation.py    dimension-consistent, non-probabilistic interpretation labels
  methods.py         explicit V2 runtime/checkpoint capability record
  mc_dropout_v2.py   validated V2 state, clustering, metrics, orchestration
  mc_dropout_ultralytics.py  trusted Ultralytics 8.4.107 lower-level adapter
```

The live-specific conversion stays in `01_WINDOWS_AI/live_tester/uncertainty_adapter.py`
and `mc_dropout_adapter.py`; shared responsive TrueType rendering is in
`presentation_ui.py`. The
detector-independent core has no capture-loop, ROS 2, or file-output dependency; only
the dedicated Ultralytics adapter imports the controlled inference stack lazily.

## Tests

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q .\tests\uncertainty
```

Tests use deterministic fake detectors and require no model, GPU, camera, network, ROS
2, or UAV hardware.
