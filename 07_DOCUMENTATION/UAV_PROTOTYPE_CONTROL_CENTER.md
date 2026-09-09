# UAV Prototype Control Center

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

The browser-based recorded-analysis dashboard also needs its requirements from
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

Quick actions launch the configured live tester, recorded model-analysis dashboard,
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
pass per frame; repeated V1/V2 work begins only after `U` freezes the exact raw frame.

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
stopped individually, and **Stop all** terminates every process started by the GUI. The
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
- The recorded model dashboard is a separate loopback-only Gradio application launched
  by the control center.
- A real V2 inference session requires the external trained V2 checkpoint, its approved
  digest, and a compatible CUDA/Ultralytics environment.
- The GUI is an operator tool, not flight-safety certification.

## Verification

```powershell
& $env:UAV_YOLO_PYTHON -m pytest -q .\tests\control_center
& $env:UAV_YOLO_PYTHON .\01_WINDOWS_AI\apps\uav_prototype_control_center.py --smoke-test
```

The smoke test constructs all five tabs, completes layout, prints the tab count, and
closes without starting a model, video, network connection, dashboard server, or
Mission Copilot run.
