# AGENTS.md — UAV AI ROS2 Prototype

## Purpose

This repository is a public research prototype for UAV perception, tracking, ROS 2 integration, dashboards, dataset engineering, and model training.

The current development goal is to add an **Agentic UAV Mission Copilot** without breaking the existing Windows AI and Ubuntu ROS 2 prototype.

Read this file before making changes.

## Repository context

The repository currently separates the system into two main runtime sides:

1. **Windows AI side**
   - Captures video.
   - Runs YOLO detection.
   - Runs BoT-SORT tracking through Ultralytics.
   - Calculates target priority/threat information.
   - Packages frames and detections into TCP messages.

2. **Ubuntu ROS 2 side**
   - Receives TCP data from Windows.
   - Converts received data into ROS 2 messages.
   - Publishes topics for dashboards and other robotic components.

Important existing paths include:

- `00_PROJECT_GUIDE/`
- `01_WINDOWS_AI/apps/`
- `01_WINDOWS_AI/launchers/`
- `02_ROS2_WINDOWS_MIRROR/bridge/`
- `02_ROS2_WINDOWS_MIRROR/dashboards/`
- `04_DATASET_ENGINEERING/`
- `05_TRAINING/`
- `07_DOCUMENTATION/`

Important baseline files include:

- `01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat.py`
- `01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat_BASELINE.py`
- `02_ROS2_WINDOWS_MIRROR/bridge/uav_windows_tcp_frame_bridge.py`

Treat the actual checked-out repository as the source of truth. Inspect it before relying on this summary.

## Mandatory workflow

Before editing:

1. Run `git status`.
2. Inspect the repository tree.
3. Read the root `README.md`.
4. Read relevant files in `00_PROJECT_GUIDE/`.
5. Read the current Windows sender and its baseline copy.
6. Read the ROS 2 bridge and dashboard files.
7. Identify the current Python version, dependency method, tests, and launch commands.
8. Report any mismatch between this file and the repository.

Do not start with broad refactoring.

## Protected existing behavior

Do not:

- Rename or remove existing top-level folders.
- Delete or overwrite the baseline sender.
- Change model paths or dataset locations without a clear migration plan.
- Add model weights, datasets, credentials, camera passwords, private IP assumptions, or generated outputs to Git.
- Replace the current sender, tracker, bridge, or dashboard architecture during the first Agentic Mission Copilot phase.
- Hardcode the old company VM address or TCP port into new logic.
- Assume ROS 2, PX4, a GPU, or model weights are installed on the personal development computer.
- Push, force-push, merge, or create releases unless explicitly requested.

Prefer additive changes in a new folder.

## Current feature direction

Create a new subsystem under:

`06_AGENTIC_AUTONOMY/`

The first version must be a **pure-Python, simulation-first Mission Copilot core** that can run on a normal personal computer without ROS 2 or UAV hardware.

The first version should:

- Read a structured mission snapshot from JSON.
- Maintain a small world model of UAVs, targets, regions, and mission constraints.
- Convert a mission objective into structured tasks.
- Allocate tasks to suitable UAVs.
- Validate recommendations through deterministic safety rules.
- Produce explainable recommendations in JSON and readable console output.
- Include tests and example scenarios.
- Keep all flight actions as recommendations or simulated tasks.

The first version must not:

- Directly control motors.
- Publish velocity, attitude, or actuator commands.
- Call PX4 or ArduPilot.
- Depend on an LLM API.
- Depend on ROS 2.
- Implement weapons, engagement, or harmful-action logic.
- Claim autonomous safety certification.

## Architecture rule

Keep the decision core independent from adapters:

```text
JSON / future ROS 2 adapter
        ↓
Domain models
        ↓
World model
        ↓
Mission planner
        ↓
Task allocator
        ↓
Safety validator
        ↓
Explainable recommendation
        ↓
Console / JSON / future dashboard adapter
```

Future ROS 2 and language-model integrations must be adapters around the deterministic core, not embedded inside it.

## Coding standards

- Use clear Python with type hints.
- Prefer standard-library dependencies for the first phase.
- Use `dataclasses`, `Enum`, and explicit validation where appropriate.
- Keep functions small and domain names clear.
- Avoid hidden global state.
- Separate domain logic from file I/O and CLI code.
- Add docstrings for public classes and functions.
- Use human-readable comments only where the reason is not obvious.
- Handle invalid JSON and missing fields with useful errors.
- Keep Windows path compatibility.
- Use repository-relative paths.
- Never print secrets.

## Testing expectations

Add tests for at least:

- Mission snapshot parsing.
- Low-battery return-home recommendation.
- Critical-battery handling.
- Unavailable UAV exclusion.
- Task assignment based on capability and distance.
- Safety rejection of an invalid task.
- Deterministic output for the same input.
- Explanation/reason fields in every recommendation.

Tests must not require ROS 2, GPU, internet access, cameras, or model weights.

## Change discipline

For every coding task:

1. Explain the files you plan to change.
2. Make the smallest coherent change.
3. Run relevant tests.
4. Show the test result.
5. Summarize what changed and what remains.
6. Do not silently alter unrelated files.

When uncertain about an existing behavior, stop and inspect rather than guessing.

## Documentation expectations

The new subsystem must include:

- `06_AGENTIC_AUTONOMY/README.md`
- Architecture explanation.
- Input and output schemas.
- Example scenarios.
- Run command.
- Test command.
- Current limitations.
- Roadmap for ROS 2 and event-based vision integration.

## Final response format

After work, report:

1. Repository findings.
2. Files created or changed.
3. Architecture implemented.
4. Commands run.
5. Test results.
6. Known limitations.
7. Recommended next step.
