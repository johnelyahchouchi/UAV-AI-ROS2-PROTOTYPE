# Live Screen / Video Model Tester

The tester runs a trusted local Ultralytics detection checkpoint against a selected MP4,
an automatically discovered MP4, a test image, or a Windows screen region. It has no ROS
2, UAV-control, network, or camera-stream dependency.

## Architecture

`01_WINDOWS_AI/apps/live_screen_model_tester.py` is a compatibility entry point. Reusable
code lives under `01_WINDOWS_AI/live_tester/`:

| Module | Responsibility |
|---|---|
| `domain.py` | Small validated values, protocols, box/filter helpers. |
| `configuration.py` | CLI, paths, model/source selection, monitors, device validation. |
| `sources.py` | MP4 decoding, MSS screen capture, one-time ROI selection. |
| `detector.py` | SHA-256 verification before trusted YOLO loading and one-pass detection. |
| `renderer.py` | Detection boxes and compact performance HUD. |
| `runtime.py` | Frame processing, rolling timings, keyboard loop, screenshots. |
| `uncertainty_adapter.py` | On-demand V1 conversion and inspection display. |
| `mc_dropout_adapter.py` | On-demand validated V2 loading and inspection display. |
| `app.py` | Composition and console summary. |

The uncertainty core is separately testable under `08_MODEL_UNCERTAINTY/src/`.

## Canonical presentation launch

```powershell
$env:UAV_YOLO_PYTHON = "D:\path\to\UAV_YOLO_ENV\Scripts\python.exe"
$env:UAV_MODEL_PATH = "D:\models\military_kaggle_v1.pt"
# Optional; requires the real digest in ACTIVE_MODEL_HASHES.csv:
$env:UAV_MCDO_V2_MODEL_PATH = "D:\models\military_kaggle_v2_mcdo.pt"
.\01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat
```

The BAT file uses paths relative to itself, attempts to locate the existing controlled
environment next to the repository, requires the external trusted checkpoint through
`UAV_MODEL_PATH`, opens a native MP4 picker, and loops the chosen file. The picker—not a
predefined video environment variable—is the default input.

Equivalent explicit examples:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --video "D:\media\demo.mp4" --loop-video

& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --monitor 1 --select-region --device auto

& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --auto-video --loop-video --uncertainty-samples 10
```

Use `--list-monitors` to print monitor bounds. A manual region requires all of `--left`,
`--top`, `--width`, and `--height`. Video mode cannot be combined with region options.

## Normal live behavior

Every normal frame follows the latency-sensitive path:

```text
decode/capture -> one YOLO predict call -> display filter -> boxes/HUD -> preview
```

The HUD shows displayed FPS, capture/decode time, inference time, total frame time,
trusted model name, device, detection count, and source. `--device auto` uses CUDA device
0 when Torch reports it available and otherwise falls back to CPU. No fixed real-time
rate is promised.

Class filters affect display only. `--tank-only` shows `military_tank`; `--classes` accepts
`all` or a comma-separated list. Capture a smaller region or lower `--imgsz` when the
presentation machine needs lower latency.

## Controls

- `Q` or Esc: exit cleanly.
- `P`: pause/resume normal playback.
- `S`: save the current live frame or uncertainty inspection.
- `H`: show/hide the normal HUD.
- `U`: freeze a copy of the exact current raw frame. If V2 is validated and loaded,
  open the method menu; otherwise run V1 directly.
- `1` in the method menu: V1 input-perturbation robustness.
- `2` in the method menu: V2 MC Dropout model uncertainty.
- Esc or Space in the method menu: cancel and resume.
- `P`, `U`, or Space while inspecting: resume the stream.

Screenshots use unique timestamped names under `08_OUTPUTS/live_screen_tester/` by
default. The output tree is ignored by Git.

## On-demand uncertainty sequence

Pressing `U` copies the last raw frame before its live annotations, pauses capture/video
decoding, displays a working notice, runs one clean inference plus the configured input
variants, and renders per-object results. Normal playback does not resume until the
operator presses `P`, `U`, or Space.

The display names the method **V1 input-perturbation robustness** and reports detection
persistence, confidence mean/std, class agreement and entropy, class evidence share,
box center variation, and mean IoU to the clean reference. It does not label evidence as
probability.

The persistence bands are `STABLE` at `>= 0.95`, `INPUT-SENSITIVE` from `0.75` to
`< 0.95`, and `UNSTABLE / REVIEW` below `0.75`. They describe response to the tested
inputs—not accuracy or probability. Width/height variation is shown alongside center
variation, and clusters that appear only under perturbation are called out separately.
The FPS interval is reset after a pause or inspection so blocking analysis time is not
misreported as live throughput.

V2 holds that same raw frame and its preprocessed tensor fixed for 20 lower-level model
forwards while only the checkpoint's six validated `Dropout2d(p=0.20)` modules are
active. The model and all BatchNorm layers remain in evaluation mode. V2 reports
existence, class, and localization behavior separately, including winner/competition
distributions, evidence share, confidence variation, box center/size variation, and
predicted-box reference IoU. It is approximate model/epistemic uncertainty, not a
calibrated correctness probability.

V2 is shown only when a separate checkpoint supplied by `--mcdo-v2-model` or
`UAV_MCDO_V2_MODEL_PATH` passes SHA-256 trust and architecture checks. The validated
checkpoint is intentionally not stored in Git and was not present during integration.
An absent, untrusted, incompatible, or unloadable V2 checkpoint leaves the session and
the direct V1 workflow available; the tester never injects dropout into V1 or uses
deterministic repeats as a substitute.

## Presentation checklist

1. Launch the BAT file and select a local MP4.
2. Confirm the trusted model hash is printed and CUDA is selected when available.
3. Confirm video continues, boxes/labels/confidence appear, and FPS/latency update.
4. Press `U`; without V2, confirm V1 runs directly on the frozen frame.
5. With validated V2 configured, press `U`, choose `1`, and review V1 metrics.
6. Resume, press `U`, choose `2`, and confirm 20 same-frame V2 passes complete.
7. Review the separate existence, class, confidence, competition, and localization data.
8. Press `S` if an inspection screenshot is needed.
9. Press `P`, `U`, or Space and confirm video and fresh FPS timing resume.
10. Press `Q` and confirm a clean exit.

Recommended inputs:

- **Clear positive:** a large, unobstructed military target; expect stable V1 metrics.
- **Difficult positive:** small, compressed, partial, or unusual-angle target; instability
  may appear, but the result is empirical rather than guaranteed.
- **Negative control:** ordinary vehicles/scenery; expect no target or limited false
  positives depending on the trained detector.

Do not add random/copyrighted video to Git. Select operator-provided local media.

## Automated tests

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q `
  .\tests\live_screen_model_tester .\tests\uncertainty
```

The suite covers source validation, monitor/region logic, device fallback, model
verification order, video looping, one-pass processing, safe boxes/labels, timing,
screenshots, V1 metrics, V2 architecture/state/metrics, selector behavior, exact-frame
freeze, failure recovery, and resume behavior.
It uses fakes and needs no monitor, real model, GPU, ROS 2, camera, or network.

## Troubleshooting

- **Verified Python not found:** set `UAV_YOLO_PYTHON` to the controlled CPython 3.11
  executable. The repository deliberately does not use bare `python`.
- **Model integrity failure:** verify provenance and add an approved digest through the
  process in `00_PROJECT_GUIDE/MODEL_REGISTRY.md`; never bypass the check.
- **Preview recursion:** move the preview outside the capture region or use direct MP4
  mode. A full single-monitor capture cannot exclude its own window.
- **Black/protected content:** some hardware overlays, elevated windows, and protected
  video surfaces cannot be captured by MSS; use direct local MP4 decoding.
- **Slow `U` analysis:** repeated passes are intentionally synchronous on the frozen
  frame. Lower `--uncertainty-samples` for a shorter demonstration, but keep at least 1.
- **V2 unavailable:** obtain the exact Colab checkpoint, verify/register its real
  SHA-256, then set `UAV_MCDO_V2_MODEL_PATH`. Do not point this variable at V1 weights.
