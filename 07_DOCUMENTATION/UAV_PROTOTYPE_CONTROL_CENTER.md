# UAV Prototype Control Center

New users: open the [dashboard quick catalog](DASHBOARD_QUICK_CATALOG.md) for field definitions, worked examples and graph interpretation. The Analysis Workspace toolbar also has a **Dashboard catalog** button.

The Windows control center is a local Tkinter desktop GUI for the maintained prototype
applications. It centralizes configuration and process status without moving inference,
transport, dashboard, or planning logic into the UI.

## Launch

From Explorer, double-click:

```text
01_WINDOWS_AI\launchers\Start_UAV_Prototype_GUI.bat
```

Or from PowerShell:

```powershell
.\01_WINDOWS_AI\launchers\Start_UAV_Prototype_GUI.bat
```

The launcher first uses a repository `.venv`, then the sibling `UAV_YOLO_ENV`, unless
`UAV_YOLO_PYTHON` already points to the verified Python executable. Create a fresh
environment when needed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-windows.txt
```

The embedded recorded extractor uses `imageio-ffmpeg==0.6.0`, included in
`requirements-windows.txt`. The optional browser-based recorded-analysis dashboard needs its requirements from
`01_WINDOWS_AI/model_test_dashboard/requirements.txt`, installed with the supplied
constraints file so the verified CUDA/Ultralytics stack is not replaced.

## Editable branding

Header text is kept in one small file:

```text
01_WINDOWS_AI\control_center\branding.json
```

Edit `title`, `subtitle`, `author`, or `internship_period`, then restart the control
center. The `logo_path` value is relative to the JSON file unless an absolute path is
used. `logo_max_width` and `logo_max_height` control the displayed size. The
dark-header logo is stored at:

```text
01_WINDOWS_AI\control_center\assets\additess_logo_dark.png
```

Replace that PNG while keeping the same filename to update the logo without changing
code. A wide image with a transparent or dark background works best. If the image is
missing or unreadable, the header falls back to a plain `ADDITESS` text label so the
dashboard still opens.

## GUI sections

### Dashboard

The landing dashboard reports:

- active Python and CUDA/CPU state;
- configurable loopback port for the recorded-analysis dashboard;
- selected model count and managed child processes;
- live tester and recorded-dashboard availability;
- dashboard dependency availability;
- SHA-256 trust status for the selected base and V2 checkpoints;
- selected source and Mission Copilot scenario readiness.

Quick actions launch the configured live tester, open the integrated Analysis Workspace,
secure sender, or deterministic Mission Copilot. Output and documentation folders are
also accessible here.

### Live Tester

Every maintained live tester option is configurable:

- external base and MC Dropout V2 checkpoint paths;
- MP4 picker, explicit MP4, automatic MP4, full monitor, interactive screen region, or
  manual screen coordinates;
- video looping, monitor, confidence, NMS IoU, image size, device, FPS cap, classes,
  tank-only display, output directory, and trusted-model registry;
- V1 perturbation sample count, seed, and matching IoU;
- V2 enable/required state, stochastic pass count, and matching IoU.

The GUI verifies the configured checkpoint digest before launch. The existing runtime
performs the authoritative architecture check and refuses required V2 when the six
validated `Dropout2d(p=0.20)` placements are absent. Normal playback remains one YOLO
pass per displayed frame. When continuous uncertainty is enabled, periodic copied raw
frames refresh separate V1 and V2 cards in a background worker. `U` remains an optional
detailed exact-frame report when using the standalone live-tester launcher. From the
main dashboard, playback and zoom controls appear in the embedded Live view.

### Analysis Workspace

This is the second dashboard inside the same main window, below the existing large
Additess logo. No additional browser server or port is needed for this workspace.

1. In **Live Tester**, browse for the trusted base checkpoint and optional validated
   V2 checkpoint. Enable V2 explicitly if wanted. Keep continuous uncertainty enabled.
2. Press **Launch live tester** or **Start live session**. Select the MP4 if prompted.
   The chosen MP4 is also filled into Recorded extraction, so both views can use it.
3. **Live view** displays video and explanation panels. Use Pause/resume, 2x zoom,
   Save screenshot, or Stop session. Zoom changes the displayed crop only.
4. **V1 extraction + graphs** and **V2 extraction + graphs** update as each method
   completes. Select a source frame, then choose Sample inputs, Object metrics, or
   Complete JSON. All pass detections, box coordinates, transformation parameters,
   target statistics, capture time, duration, and method settings are retained.
5. **Export full JSON** writes a portable export of all records committed when export
   began. **Open saved session** reviews an existing session folder without inference.
   Saved/stopped-session views do not issue live playback commands.
6. **Recorded extraction** uses the same base model, registry, confidence, IoU, image
   size and device settings. Choose an MP4 and press Start extraction. Progress covers
   frame processing and MP4 encoding. Browse rows with scrollbars and 100-row pages;
   Open CSV and Open annotated video reveal the complete published outputs.

The recorded panel runs the existing `VideoProcessor` in a worker and uses detection
only. It does not apply V2 dropout to the recorded export. Cancellation uses the
existing cooperative cleanup path; closing the dashboard waits for cleanup/export.
The first 10,000 detections are available in the paged table; the CSV includes all
detections. Error details remain visible, and a failed run clears previous outputs.

#### Why these graphs

Both methods have two unsmoothed line charts with session capture seconds on the x-axis:

- **Detection and class consistency:** frame-level mean persistence and class
  agreement, each on a fixed 0-1 axis. Persistence describes how often an object was
  detected across inputs/passes; class agreement is conditional on detection.
- **Bounding-box consistency:** frame-level mean reference IoU, also on a fixed
  0-1 axis. V1 uses its clean reference where present (mean box for variant-only
  objects); V2 uses its mean reference box.

The graphs show variation in sampled observations, not calibrated accuracy or a
tracked object's trajectory. Object composition can change between frames. Failed
analyses and undefined no-detection metrics leave gaps; they are never plotted as
zero or perfect consistency. The live history keeps the latest 300 records per
method. Full results remain in the session journal and JSON export.

#### Data flow and storage

`analysis callbacks -> worker -> local SQLite journal -> native tables/graphs`

Each GUI live run creates a unique folder under `08_OUTPUTS/analysis_sessions/`.
`analysis.sqlite3` stores model names/hashes, availability, complete V1/V2 results,
and bounded control messages. `preview.jpg` is the latest display frame, published
through a one-frame background mailbox. The dashboard preview is limited to four
updates per second; inference FPS is reported separately. This is not an archived
video stream. Old or absent previews are visibly stale/unavailable.

The journal schema uses `metadata(key,value)`, `results(seq,payload)` and
`commands(seq,key)`. Result JSON contains `schema_version`, `method`, `frame`,
`captured_utc`, `elapsed_seconds`, `state`, `status`, `analysis`, `samples`, and
`metrics`; completed analyses also include `duration_ms`. Every sample contains its
index, family, parameters and detections. Capture time is host decode/capture UTC,
not original recording time. `metrics` contains frame means for persistence,
agreement, IoU, and the object count. The complete scientific fields remain under
`analysis`, and no pixels are embedded in JSON.

Recorded exports are under `08_OUTPUTS/recorded_extraction/`. All these outputs are
ignored by Git. Session exports run in a worker and use a fixed record boundary;
they do not hold a database read lock throughout a long export.

### Secure Sender

The sender tab exposes model/source, bridge address and port, TLS server name, sender
certificate, sender private key, bridge CA, confidence, IoU, image size, frame stride,
send width, tracker, preview, and military-only settings. It launches the existing
YOLO + BoT-SORT sender; the GUI does not reimplement transport or threat logic.

Mutual-TLS values are passed as environment variables rather than command arguments.
Credential contents are never displayed. Stream URLs are described with the shared
redaction helper and URLs containing credentials are not persisted.

### Mission Copilot

The Mission Copilot tab selects the scenario, deterministic safety policy, and JSON
output, with verbose, validation-only, and require-complete options. It launches the
existing standard-library advisory core. It does not publish flight, actuator, PX4, or
ArduPilot commands.

### Activity Log

All managed child-process console output appears in one log. Each application can be
stopped individually, and **Stop all** terminates managed processes and requests
cooperative cancellation of recorded extraction. The
control center launches argument arrays directly without a command shell.

## Local state and security

Settings are written to `08_OUTPUTS/control_center/settings.json`, which is ignored by
Git. Model weights, videos, generated outputs, certificates, and private keys remain
external. The settings file stores paths but not file contents; credential-bearing
stream URLs are cleared before saving.

Model selection does not grant trust. Add only independently verified hashes through
the process in `00_PROJECT_GUIDE/MODEL_REGISTRY.md`. The GUI and underlying applications
continue to fail closed before PyTorch deserializes an unapproved `.pt` file.

## Boundaries

- ROS 2 dashboards still run on the ROS 2/Ubuntu side; the Windows GUI does not pretend
  that ROS 2 is installed locally.
- The original loopback-only Gradio recorded dashboard remains available through its
  existing launcher; the main workspace reuses its processing backend without Gradio.
- A real V2 inference session requires the external trained V2 checkpoint, its approved
  digest, and a compatible CUDA/Ultralytics environment.
- The GUI is an operator tool, not flight-safety certification.

## Verification

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q .\tests\control_center
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\uav_prototype_control_center.py --smoke-test
```

The smoke test constructs all six main tabs and the nested analysis views, completes layout, prints the tab count, and
closes without starting a model, video, network connection, dashboard server, or
Mission Copilot run.

Integrated regression command:

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q tests/control_center tests/live_screen_model_tester tests/uncertainty 01_WINDOWS_AI/model_test_dashboard/tests
```

Tests use synthetic frames and injected detector outputs; no weights, camera, ROS 2,
internet connection, or GPU is required. Actual model throughput and GPU contention
must be measured with your configured checkpoints. Simultaneous live and recorded
inference share machine resources. No real hardware-health panel is included yet.
