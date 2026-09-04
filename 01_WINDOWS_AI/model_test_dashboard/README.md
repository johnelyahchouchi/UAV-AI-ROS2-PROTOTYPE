# UAV Model Test Dashboard

Local Windows browser dashboard for testing an Ultralytics YOLO detection model
against an uploaded video. It is isolated from the TCP sender, ROS 2 mirror, and
agentic-autonomy subsystem.

## Scope

Version 1 supports:

- drag-and-drop or browse video upload;
- an external Ultralytics detection-model `.pt` file;
- confidence, IoU, image-size, and device controls;
- detection-only or BoT-SORT tracking;
- cooperative cancellation and processing progress;
- annotated H.264 MP4 output;
- a complete detection CSV and a capped browser table;
- class counts, confidence statistics, frame count, timing, FPS, and device data.

It does not support webcams, RTSP, live partial-video output, TCP, or ROS 2.

## Dedicated Python environment

Do not use bare `python` on this workstation. It launches Amesim Python 2.7.

Configure the verified environment explicitly; it may live anywhere outside the
repository:

```text
UAV_YOLO_PYTHON=<absolute path to the verified Python executable>
```

The launcher requires `UAV_YOLO_PYTHON` and fails clearly when it is missing. It
never silently falls back to another Python.

## Install dashboard-only dependencies

The existing CUDA-enabled PyTorch, Torchvision, Ultralytics, NumPy, Matplotlib,
and OpenCV packages must not be upgraded or replaced.

From this directory:

```powershell
$Python = $env:UAV_YOLO_PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) { throw "Set UAV_YOLO_PYTHON first." }

& $Python -m pip install `
    -r requirements.txt `
    -c constraints-yolo-env.txt
```

For development tests:

```powershell
& $Python -m pip install `
    -r requirements-dev.txt `
    -c constraints-yolo-env.txt
```

Verify CUDA after installation:

```powershell
& $Python -c "import numpy, torch; print(numpy.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected on this workstation:

```text
1.26.4
2.13.0+cu130
True
NVIDIA GeForce RTX 2060
```

## Default model

`UAV_MODEL_PATH` takes precedence. Otherwise the dashboard displays the
repository-relative deployment location:

```text
03_MODELS\active\detector\military_kaggle_v1.pt
```

Production model artifacts should remain outside Git. Before loading, the
dashboard hashes the selected `.pt` file and requires that SHA-256 in the trusted
model registry. The model cache is keyed by canonical path, file size, and
modification time. Selecting an unchanged verified model reuses the loaded object.

## Launch

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\01_WINDOWS_AI\model_test_dashboard\launch_dashboard.ps1
```

The local page is:

```text
http://127.0.0.1:7860
```

The launcher prints resolved package versions, CUDA availability, and GPU name
before starting the server. If CUDA is unavailable, Auto resolves to CPU and an
explicit GPU 0 selection reports a clear error.

## Processing architecture

```text
Gradio upload and settings
          |
Validated uploaded-video source adapter
          |
Cached Ultralytics detection model
          |
predict() or BoT-SORT track()
          |
Detection records + annotated OpenCV frames
          |
Complete CSV + OpenCV intermediate MP4
          |
Bundled imageio-ffmpeg H.264 conversion
          |
Validated final MP4 + browser results
```

`source_adapter.py` defines the frame-source boundary. A later version can add a
camera or RTSP implementation without placing source-specific logic in the UI or
inference core.

## Device behavior

- **Auto** selects GPU 0 when CUDA is available; otherwise it selects CPU.
- **GPU 0** passes `device=0` and fails clearly if CUDA device 0 is unavailable.
- **CPU** passes `device="cpu"`.

An explicit GPU selection never silently falls back to CPU. The resolved device
appears before processing and in the final summary.

## Detection report

The complete CSV uses these columns:

```text
frame_number,timestamp_seconds,track_id,class_id,class_name,confidence,x1,y1,x2,y2
```

Frame numbers are one-based. Coordinates are pixels in the original video
resolution. Track ID is blank in detection-only mode or before BoT-SORT assigns
an ID. Total detections means boxes across all frames, not unique objects.

The browser displays at most 10,000 rows to protect memory. The CSV always
contains every successful detection.

## Output and cleanup

Successful jobs are kept under:

```text
outputs/<video-name>_<timestamp>_<job-id>/
```

Each directory contains `annotated.mp4` and `detections.csv`. `outputs/` is
ignored except for its `.gitignore`.

Uploads are copied into an owned `.staging` directory. Cancellation and all
processing failures release resources and delete partial files. Cleanup refuses
to remove anything outside the owned staging root. Completed outputs remain
until the user deletes them.

Cancellation is cooperative. It finishes the current inference operation before
stopping, so cancellation latency can be one frame on a slow device.

## Tests

From this directory:

```powershell
$env:PYTHONPATH = Join-Path $PWD "src"
& $env:UAV_YOLO_PYTHON -m pytest -q
```

Automated tests use fake model results and synthetic videos. They do not require
real model weights, CUDA, ROS 2, TCP, a camera, or internet access.

## Known limitations

- One processing job and one cached model are supported at a time.
- Video decoding depends on codecs supported by the installed OpenCV build.
- H.264 conversion happens after frame inference, so results appear after the
  complete job rather than as a live partial video.
- BoT-SORT IDs can change after occlusions or difficult scene transitions.
- The dashboard is a model evaluation tool, not a safety-certified perception
  system or a flight-control component.
