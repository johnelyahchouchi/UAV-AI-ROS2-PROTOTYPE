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
| `continuous_uncertainty.py` | Latest-frame worker and unified live V1/V2 workspace. |
| `uncertainty_adapter.py` | V1 analysis and detailed inspection display. |
| `mc_dropout_adapter.py` | Validated V2 analysis and detailed inspection display. |
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

For a V2-focused run without pre-setting model paths, use:

```powershell
.\01_WINDOWS_AI\launchers\Start_MC_Dropout_V2.bat
```

It opens native pickers for the trusted base detector, validated V2 checkpoint, and
MP4. The launcher uses `--require-mcdo-v2`, so an absent, unapproved, or incompatible
V2 checkpoint produces an error instead of silently starting a V1-only session.

Equivalent explicit examples:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --video "D:\media\demo.mp4" --loop-video --continuous-uncertainty

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

## Continuous V1/V2 workspace

`--continuous-uncertainty` composes one 16:9 operator view with the annotated live
video on the left and separate V1 and V2 cards on the right. The first raw-frame copy
is sampled automatically. Later cycles start no faster than
`--continuous-uncertainty-interval` seconds, which defaults to 2 seconds.

Only one uncertainty cycle can run at a time. If it is still working when another
interval arrives, the older worker is allowed to finish and no stale frames are queued.
V1 runs first and V2 follows on the same copied frame. Each panel independently shows
waiting, queued, analyzing, ready, unavailable, or failed state, the sampled frame
number, update age, analysis duration, method-specific metrics, and scientific scope.
A V1 or V2 error is contained in its card while normal live detection continues.

The base playback path still calls the normal detector exactly once per displayed
frame. Continuous V1 uses extra predictions on a periodic copied frame and V2 uses its
own validated repeated-pass model, so their compute load can reduce achieved FPS on a
busy GPU even though the capture/UI loop is not synchronously blocked. Increase the
refresh interval or reduce sample counts if needed for the presentation computer.

## Controls

- `Q` or Esc: exit cleanly.
- `P`: pause/resume normal playback.
- `S`: save the current live frame or uncertainty inspection.
- `H`: show/hide the normal HUD.
- `U`: optional detailed report on a copy of the exact current raw frame. If V2 is
  validated and loaded, open the method menu; otherwise run V1 directly.
- `1` in the method menu: V1 input-perturbation robustness.
- `2` in the method menu: V2 MC Dropout model uncertainty.
- Esc or Space in the method menu: cancel and resume.
- `P`, `U`, or Space while inspecting: resume the stream.
- `A`/`D` or `[`/`]` while inspecting: move between target-card pages when more than
  two targets are present.

Screenshots use unique timestamped names under `08_OUTPUTS/live_screen_tester/` by
default. The output tree is ignored by Git.

## Optional detailed uncertainty sequence

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

V1, V2, their working notices, and the method selector use anti-aliased TrueType text
rendered at the final display resolution through Pillow. The renderer discovers Segoe
UI or Arial from the Windows font directory and falls back to cross-platform sans-serif
fonts without a user-specific path. Inspection output uses a responsive 16:9 split with
an uncropped frozen frame, frame summary, restrained semantic status colors, and at most
two readable target cards per page.

V2 holds that same raw frame and its preprocessed tensor fixed for 20 lower-level model
forwards while only the checkpoint's six validated `Dropout2d(p=0.20)` modules are
active. The model and all BatchNorm layers remain in evaluation mode. V2 reports
existence, class, and localization behavior separately, including winner/competition
distributions, evidence share, confidence variation, box center/size variation, and
predicted-box reference IoU. It is approximate model/epistemic uncertainty, not a
calibrated correctness probability.

V2 is shown only when a separate checkpoint supplied by `--mcdo-v2-model` or
`UAV_MCDO_V2_MODEL_PATH` passes SHA-256 trust and architecture checks. The validated
checkpoint is intentionally not stored in Git. Its digest is recorded in the model
registry after local manual validation.
An absent, untrusted, incompatible, or unloadable V2 checkpoint leaves the session and
the direct V1 workflow available; the tester never injects dropout into V1 or uses
deterministic repeats as a substitute.

## Presentation checklist

1. Launch the BAT file and select a local MP4.
2. Confirm the trusted model hash is printed and CUDA is selected when available.
3. Confirm video, boxes, labels, confidence, FPS, and latency remain visible on the left.
4. Confirm the V1 card moves from analyzing to a populated input-stability result.
5. With validated V2 configured, confirm its separate card populates after V1.
6. Without V2, confirm that card clearly says unavailable and no substitute result appears.
7. Review existence, class, confidence, competition, and localization dimensions.
8. Optionally press `U` and choose a full detailed report for the exact current frame.
9. Press `S` if a full-workspace or detailed-report screenshot is needed.
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
- **Low FPS during live panels:** increase `--continuous-uncertainty-interval`, reduce
  V1/V2 sample counts, or lower image size. The worker does not queue stale frames.
- **Slow `U` analysis:** the optional detailed report is intentionally synchronous on
  the frozen frame. Lower `--uncertainty-samples` if appropriate, but keep at least 1.
- **V2 unavailable:** set `UAV_MCDO_V2_MODEL_PATH` to the separately stored validated
  checkpoint. If using a different artifact, verify its provenance and register its real
  SHA-256 first. Do not point this variable at V1 weights.
