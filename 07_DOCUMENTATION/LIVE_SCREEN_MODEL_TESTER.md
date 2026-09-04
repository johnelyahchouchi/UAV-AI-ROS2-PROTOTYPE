# Live Screen Model Tester

## Purpose and scope

`live_screen_model_tester.py` is a standalone Windows utility for testing a trusted local Ultralytics YOLO checkpoint against an automatically selected MP4 or a selected desktop region. It decodes local video or captures pixels with MSS, runs local inference, and displays a local OpenCV preview.

It does **not** start ROS 2, publish network traffic, contact a cloud service, control a UAV, or modify the existing sender/bridge/dashboard pipeline. All flight-related interpretation remains outside this tool.

## Architecture

```text
CLI configuration
      |
MSSScreenSource  --\
                    -> YoloDetector -> FrameProcessor -> OverlayRenderer
VideoFileSource --/          |               |
                       trusted hash     PerformanceTracker
                                               |
                                       ScreenshotStore
```

The capture source, detector, frame processor, renderer, timing tracker, and screenshot store are separate components. Heavy screen, OpenCV, Torch, and Ultralytics imports are deferred so non-GUI unit tests can use fake frames and fake dependencies.

The expected primary checkpoint is `military_kaggle_v1.pt`, configured externally rather than stored or hardcoded in this repository. Its model-provided `model.names` map remains authoritative. The expected validation labels are `camouflage_soldier`, `weapon`, `military_tank`, `military_truck`, `military_vehicle`, `civilian`, `soldier`, `civilian_vehicle`, `military_artillery`, `trench`, `military_aircraft`, and `military_warship`.

## Prerequisites

- Windows desktop session.
- The controlled Python environment specified by `UAV_YOLO_PYTHON`.
- Dependencies from `requirements-windows.txt`, including `mss==10.2.0`.
- A local `.pt` checkpoint specified with `--model` or `UAV_MODEL_PATH`.
- The checkpoint SHA-256 must be present in `00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`, or in the explicitly supplied `--registry` file.

Install or update the controlled environment from the repository root:

```powershell
& $env:UAV_YOLO_PYTHON -m pip install -r .\requirements-windows.txt
```

Do not load an unknown checkpoint merely to discover whether it works. Register only a model whose provenance and SHA-256 have been reviewed.

## Discover monitor coordinates

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py --list-monitors
```

MSS monitor numbers are one-based in this tool. The output includes each monitor's absolute desktop coordinates and dimensions.

## Usage

Open a native file picker, select an MP4, and loop it continuously:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --select-video --loop-video
```

The file-picker selection is authoritative for that run. It does not read `UAV_VIDEO_PATH` or use automatic discovery.

Automatically select and continuously loop the newest MP4:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --auto-video --loop-video
```

The non-recursive MP4 search checks the repository root, the Windows known Desktop location (including OneDrive redirection), the user Videos directory, and the current directory. Within the first location containing MP4 files, the most recently modified file is selected. Use `--video` or `UAV_VIDEO_PATH` when a specific file is required.

Interactive selection on monitor 1:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --model $env:UAV_MODEL_PATH `
  --monitor 1 `
  --select-region
```

Fixed region with automatic GPU/CPU selection and a 30 FPS cap:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --model C:\models\best.pt `
  --left 100 --top 100 --width 1280 --height 720 `
  --device auto --conf 0.35 --iou 0.45 --imgsz 640 --max-fps 30
```

Primary-model tank-only display:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --monitor 1 --select-region --tank-only
```

Display several exact model classes:

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --monitor 1 --classes "military_tank,cargo_truck"
```

Single-image smoke test (trusted model load, inference, overlay, and output write without opening a preview):

```powershell
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\live_screen_model_tester.py `
  --test-frame C:\images\sample.png --device auto
```

The launcher enforces the configured Python environment:

```powershell
.\01_WINDOWS_AI\launchers\start_live_screen_tester.ps1 -SelectRegion -Monitor 1
```

For the continuous double-click workflow, run:

```bat
01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat
```

With no arguments, the BAT launcher opens a native Browse dialog and waits for the user to select an MP4. It decodes that file directly, loops it at EOF, and therefore cannot capture its own preview recursively. The selected path is printed at startup. Cancelling the dialog exits with a clear message.

The launcher first honors `UAV_YOLO_PYTHON` and `UAV_MODEL_PATH`. When they are unset, it looks for the verified sibling `UAV_YOLO_ENV` and the repository-local `military_kaggle_yolov8s_v1` training checkpoint. The trusted-model SHA-256 check still runs before loading. If those local defaults are absent, configure the environment variables explicitly.

Optional application arguments are forwarded directly to the Python tester, which avoids dependence on the machine's PowerShell execution policy:

```bat
01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat --list-monitors
01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat --video "D:\media\demo.mp4" --loop-video
01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat --auto-video --loop-video
01_WINDOWS_AI\launchers\Start_Live_Screen_Tester.bat --monitor 2 --select-region --max-fps 30
```

The process continues until `Q`, `Esc`, Ctrl+C, or the preview window close button stops it.

## Controls

- `Q` or `Esc`: quit cleanly.
- `P`: pause or resume capture and inference.
- `S`: save the current annotated frame.
- `H`: show or hide the HUD.

Screenshots are written by default to `08_OUTPUTS/live_screen_tester/` with unique timestamped names. That generated-output tree is ignored by Git.

## Configuration behavior

- `--device auto` selects CUDA device 0 when Torch reports it available; otherwise it falls back to CPU.
- Direct video mode plays at the source FPS by default; `--max-fps` can apply a lower cap.
- An explicitly requested unavailable CUDA device fails with an actionable error.
- `--classes all` displays all model detections. Class filtering affects display only; it does not change the model itself.
- `--tank-only` overrides `--classes` and displays only `military_tank`.
- Manual regions require all four coordinates, non-negative `left`/`top`, positive dimensions, and containment inside an active monitor when monitor bounds can be read.
- Interactive selection occurs once at startup and does not modify global desktop settings.
- The preview is placed outside the capture region when monitor geometry permits. A warning is printed when feedback cannot be avoided, such as full-monitor capture on a single monitor.

## Performance notes

The HUD reports rolling displayed FPS plus capture, inference, and total frame-processing time. To improve responsiveness:

- capture only the application area you need;
- lower `--imgsz` if model accuracy remains acceptable;
- use `--device auto` in the verified CUDA environment;
- use `--max-fps` to reduce load when full throughput is unnecessary;
- move the preview to a different monitor if capturing an entire monitor.

No fixed real-time FPS is guaranteed. Performance depends on capture dimensions, model size, inference image size, GPU/CPU availability, and other desktop activity.

## Tests

The focused tests do not require a monitor, GPU, model weights, camera, ROS 2, or network access:

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q .\tests\live_screen_model_tester
```

They cover region validation, monitor selection, automatic MP4 discovery, looping video input, class filters, model path resolution, device fallback, rolling timing, safe box clipping, screenshot path uniqueness, preview placement, and fake-frame processing.

## Troubleshooting

**`mss` is missing**

Install `requirements-windows.txt` using `UAV_YOLO_PYTHON`; do not use the repository's unrelated bare `python` command.

**Model integrity verification failed**

The `.pt` file is not trusted by the active SHA-256 registry. Verify its provenance and follow the model-registry procedure in `00_PROJECT_GUIDE/MODEL_REGISTRY.md`. Do not bypass the check.

**CUDA was requested but is unavailable**

Use `--device auto` for clean CPU fallback, or verify the controlled Torch/CUDA installation before explicitly selecting a GPU.

**Preview appears inside the capture**

Select a smaller region or move the preview to another monitor. Full-monitor capture on a single monitor cannot fully avoid preview feedback.

**Black or protected content**

Some applications, protected video surfaces, elevated windows, and hardware overlays may not be capturable. Test an ordinary desktop window first and match the tester's privilege level to the target application when appropriate.

**Capture becomes invalid after display changes**

Stop and restart the tester after disconnecting, rotating, scaling, or rearranging monitors so MSS can enumerate the new geometry.

## Limitations

- Windows desktop testing only; no ROS or UAV adapter is included.
- The preview is an OpenCV desktop window, not a remote dashboard.
- Automatic MP4 discovery is intentionally non-recursive and selects the newest file in the first matching search location.
- Display filters do not reduce the detector's inference workload.
- Manual non-negative coordinates follow the current safety requirement; use monitor or interactive selection for displays positioned at negative virtual-desktop coordinates.
- Uncertainty is reserved in the internal detection record for future overlays but is not synthesized when the model does not provide it.
