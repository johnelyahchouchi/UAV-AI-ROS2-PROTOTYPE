# UAV AI ROS 2 Prototype

A public research/engineering prototype for military-object perception, tracking,
threat metadata, Windows-to-ROS 2 transport, model evaluation, and deterministic
mission recommendations. Flight-facing autonomy remains simulated and advisory.

## System story

```text
local image / MP4 / Windows screen
        ↓
trusted Ultralytics YOLO detector
        ├── recorded model testing
        └── live screen/video tester (one pass/frame, FPS + latency)
                    ↓ press U on one exact frame
              on-demand uncertainty
                 ├── V1 input robustness (input changes)
                 └── V2 MC Dropout (same input; model masks change)
                            ↓
                transparent per-target metrics

Windows perception sender ── mutual TLS/TCP ── ROS 2 bridge
                                                  ↓
                           operational / analytics / timeline dashboards

JSON mission snapshot ── deterministic Mission Copilot ── safe recommendations
```

The live path uses one detector inference per frame. Repeated uncertainty analysis
happens only after the operator presses `U`; it never runs in the normal capture loop.

## Quick presentation demo (Windows, no ROS 2 required)

Use CPython 3.11 in the controlled YOLO environment. Set external paths when the
launcher cannot discover the existing local environment/checkpoint:

```powershell
$env:UAV_YOLO_PYTHON = "D:\path\to\UAV_YOLO_ENV\Scripts\python.exe"
$env:UAV_MODEL_PATH = "D:\models\military_kaggle_v1.pt"
# Optional only after the validated V2 digest is registered:
$env:UAV_MCDO_V2_MODEL_PATH = "D:\models\military_kaggle_v2_mcdo.pt"
.\01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat
```

The launcher opens a native MP4 picker and loops the chosen video. Controls:

- `U`: freeze the current raw frame. With validated V2 loaded, choose `1` for V1 or
  `2` for V2; otherwise V1 runs directly and the console explains how to enable V2.
- `P`, `U`, or Space: leave an inspection and resume; `P` also pauses normally.
- `S`: save the current live or inspection view under `08_OUTPUTS/`.
- `H`: toggle the compact HUD. `Q` or Esc exits.

Normal live output shows boxes, labels, confidence, displayed FPS, capture time, and
inference latency. See [the full demo guide](07_DOCUMENTATION/LIVE_SCREEN_MODEL_TESTER.md).

Recommended demo cases are: a clear positive target, a small/compressed/partial positive
target, and an ordinary-traffic/scenery negative control. Provide local media; do not
commit downloaded or copyrighted videos.

## Uncertainty status

- **V1 input-perturbation robustness: implemented.** One clean pass plus deterministic
  brightness, contrast, blur, noise, and JPEG variants report persistence, confidence
  mean/std, class agreement/entropy/evidence share, box variation, and reference IoU.
- **V2 MC Dropout runtime: implemented, checkpoint not present.** It validates six
  late-head `Dropout2d(p=0.20)` layers, keeps BatchNorm in evaluation mode, and runs 20
  lower-level stochastic forwards on one unchanged frame. It becomes selectable only
  with the separate allowlisted external checkpoint configured through
  `UAV_MCDO_V2_MODEL_PATH`; deterministic repeats and V1 weights are never substituted.

These metrics are not correctness probabilities or a safety certification. Details are
in [08_MODEL_UNCERTAINTY/README.md](08_MODEL_UNCERTAINTY/README.md).

## Windows AI and recorded testing

- Active sender: `01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat.py`.
- Protected regression copy: `win_yolo_tcp_sender_botsort_threat_BASELINE.py`.
- Live tester: thin entry point plus reusable modules in `01_WINDOWS_AI/live_tester/`.
- Recorded-video test/export UI: `01_WINDOWS_AI/model_test_dashboard/`.
- Canonical sender launchers: `start_yolo_sender.ps1` and its BAT wrapper.

The optional sender control-panel launcher is retained as a distinct GUI, while the
live presentation uses `Start_Live_Screen_Tester.bat`. See
`01_WINDOWS_AI/launchers/README.md` for the complete launcher inventory.

The recorded dashboard has its own setup and launch instructions in
`01_WINDOWS_AI/model_test_dashboard/README.md`.

## ROS 2 side

`02_ROS2_WINDOWS_MIRROR/` is a source/deployment mirror of the Ubuntu side, not evidence
that ROS 2 is installed on this Windows computer. The bridge receives authenticated
version-2 packets and publishes the existing camera/detection topics. The dashboards are
now named by purpose:

- `uav_operational_dashboard.py`: primary camera/detection/threat presentation.
- `uav_analytics_dashboard.py`: historical charts, map, and target lifetimes.
- `uav_timeline_dashboard.py`: first-seen and classification-change events.

See `02_ROS2_WINDOWS_MIRROR/dashboards/README.md` and the SROS2 deployment guide before
running an Ubuntu deployment.

## Security and model artifacts

Model checkpoints are deserialized only after SHA-256 allowlist verification. The sender
and bridge use mutual TLS 1.3 and fail closed when identities are missing. Dataset ZIPs,
image inputs, URLs, CSV output, and transport framing have shared validation helpers and
regression tests. Read [SECURITY.md](SECURITY.md) before deployment.

Git contains model registries, hashes, configuration, and provenance—not model binaries,
datasets, training plots, or test videos. Keep weights in controlled artifact storage or
release/LFS storage where policy permits. Never commit credentials or camera passwords.

## Tests

Core tests do not require ROS 2, a GPU, a camera, model weights, or network access:

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q tests\live_screen_model_tester tests\uncertainty tests\dashboards
py -3.11 -m pytest -q tests\security
Push-Location 06_AGENTIC_AUTONOMY; py -3.11 -m pytest -q; Pop-Location
```

The GitHub security workflow also compiles Python, audits dependencies, scans relevant
trust-boundary code, and rejects committed private keys.

## Repository map

```text
00_PROJECT_GUIDE/          active registries, runbook, reviewed audit
01_WINDOWS_AI/             sender, live tester, recorded dashboard, launchers, tools
02_ROS2_WINDOWS_MIRROR/    authenticated bridge and three role-based dashboards
04_DATASET_ENGINEERING/    import/build/inspect code; datasets remain external
05_TRAINING/               scripts, configs, args.yaml, results.csv provenance
06_AGENTIC_AUTONOMY/       simulation-first deterministic Mission Copilot
07_DOCUMENTATION/          architecture, protocol, security, testing, and demo guides
08_MODEL_UNCERTAINTY/      V1/V2 cores, retained V2 notebook, no model weights
tests/                     security, live tester, uncertainty, and dashboard tests
uav_security/              shared validation, integrity, TLS, and safe-output helpers
```

`03_MODELS`, `06_TEST_MEDIA`, `08_OUTPUTS`, datasets, caches, and generated runs are
intentionally local/ignored or external rather than source-controlled.

## Current limitations

This is not certified flight software. ROS 2 deployment still needs Ubuntu-side package
verification, SROS2 identities, and hardware-in-the-loop testing. V1 robustness measures
response to selected input variations only. V2 still requires the exact external
validated checkpoint, trust-registry enrollment, and real GPU smoke validation. The
Mission Copilot does not command motors, PX4, ArduPilot, actuators, or harmful actions.
