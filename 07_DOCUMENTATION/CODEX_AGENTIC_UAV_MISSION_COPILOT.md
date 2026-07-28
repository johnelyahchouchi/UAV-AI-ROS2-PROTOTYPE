# Agentic UAV Mission Copilot — Codex Project Context and Implementation Plan

## 1. Why this document exists

This document gives Codex enough context to work safely and intelligently on the repository:

`johnelyahchouchi/UAV-AI-ROS2-PROTOTYPE`

Development will take place from a personal computer and personal GitHub access, not from the company account or company workstation.

Codex must inspect the local checkout before changing anything. This document describes the known project state, but the checked-out repository is always the final source of truth.

---

## 2. Existing project in simple terms

The repository contains an existing UAV AI prototype with two connected sides.

### Windows AI side

The Windows side is responsible for:

```text
Video source
→ YOLO object detection
→ BoT-SORT tracking
→ target/threat fields
→ TCP message packaging
→ transmission to Ubuntu
```

Known important files:

```text
01_WINDOWS_AI/apps/live_yolo_stream.py
01_WINDOWS_AI/apps/make_btr_demo_video.py
01_WINDOWS_AI/apps/uav_ai_control_panel.py
01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat.py
01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat_BASELINE.py
```

Known launcher files include:

```text
01_WINDOWS_AI/launchers/Start_Clean_Baseline.ps1
01_WINDOWS_AI/launchers/Start_UAV_AI_Control_Panel.bat
01_WINDOWS_AI/launchers/Start_UAV_Windows_Sender.bat
01_WINDOWS_AI/launchers/start_yolo_sender.ps1
01_WINDOWS_AI/launchers/start_yolo_sender_full_menu.ps1
01_WINDOWS_AI/launchers/start_yolo_sender_menu.ps1
```

The active sender previously loaded video, YOLO, and BoT-SORT successfully. Model weights are intentionally kept outside Git. Codex must not add weights to the repository.

### Ubuntu ROS 2 side

The Ubuntu side is responsible for:

```text
TCP receiver
→ packet decoding
→ ROS 2 messages/topics
→ dashboards and robotic consumers
```

Known bridge file:

```text
02_ROS2_WINDOWS_MIRROR/bridge/uav_windows_tcp_frame_bridge.py
```

Known dashboard code is under:

```text
02_ROS2_WINDOWS_MIRROR/dashboards/
```

The project previously used a company Ubuntu VM and a TCP bridge. That company-specific address must not be assumed on the personal computer. Any future network host and port must be configurable.

### Repository organization

Known top-level structure:

```text
00_PROJECT_GUIDE/
01_WINDOWS_AI/
02_ROS2_WINDOWS_MIRROR/
04_DATASET_ENGINEERING/
05_TRAINING/
07_DOCUMENTATION/
README.md
```

The new Agentic subsystem should use the unused number:

```text
06_AGENTIC_AUTONOMY/
```

Do not reorganize the existing repository in this phase.

---

## 3. Why the next subsystem is Agentic UAV Autonomy

The current prototype detects, tracks, prioritizes, and transmits targets.

It does not yet provide a structured mission-level layer that answers:

- What is happening across the mission?
- Which UAV is available?
- Which UAV is most suitable for a task?
- What should happen when battery is low?
- Should a target be observed, verified, or ignored?
- What recommendation is safe?
- Why did the system make that recommendation?
- How should the mission change after a failure?

The new subsystem will sit above perception and telemetry:

```text
Detections + tracks + UAV state + map state
                    ↓
             Mission world model
                    ↓
             Task generation
                    ↓
             Task allocation
                    ↓
             Safety validation
                    ↓
      Explainable operator recommendation
```

The first version is deliberately not an LLM-controlled drone. It is a deterministic, testable mission-decision core.

---

## 4. Meaning of “Agentic” in this project

For this project, an agentic system must perform a controlled loop:

```text
Observe
→ update world state
→ interpret mission objective
→ create tasks
→ allocate tasks
→ validate safety
→ recommend actions
→ monitor future updates
→ revise the plan
```

The first implementation only needs the first seven steps using static scenario files.

Later phases can add live updates, ROS 2 topics, language input, dashboards, and event-based alerts.

---

## 5. Phase 1 goal

Build a working **Agentic UAV Mission Copilot v1** that runs locally with Python.

It must accept a JSON scenario containing:

- Mission objective.
- UAV states.
- Target states.
- Regions.
- Constraints.

It must return:

- Structured mission tasks.
- UAV-to-task assignments.
- Rejected assignments.
- Safety decisions.
- Human-readable reasons.
- Overall mission recommendation.

### Example input idea

```json
{
  "mission_id": "demo_001",
  "objective": "Inspect the eastern sector, verify the unknown vehicle, maintain overwatch, and return low-battery UAVs.",
  "constraints": {
    "return_home_battery_percent": 25,
    "critical_battery_percent": 10,
    "minimum_link_quality": 0.35,
    "require_operator_approval_for_high_risk": true
  },
  "uavs": [
    {
      "uav_id": "uav_1",
      "status": "available",
      "battery_percent": 72,
      "position": {"x": 0.0, "y": 0.0, "z": 20.0},
      "link_quality": 0.92,
      "localization_healthy": true,
      "capabilities": ["rgb_camera", "tracking", "mapping"]
    },
    {
      "uav_id": "uav_2",
      "status": "available",
      "battery_percent": 61,
      "position": {"x": 80.0, "y": 10.0, "z": 25.0},
      "link_quality": 0.75,
      "localization_healthy": true,
      "capabilities": ["rgb_camera", "tracking", "zoom_camera"]
    },
    {
      "uav_id": "uav_3",
      "status": "available",
      "battery_percent": 18,
      "position": {"x": -40.0, "y": 5.0, "z": 18.0},
      "link_quality": 0.80,
      "localization_healthy": true,
      "capabilities": ["rgb_camera", "mapping"]
    }
  ],
  "targets": [
    {
      "target_id": "target_07",
      "target_class": "unknown_vehicle",
      "confidence": 0.63,
      "priority": "high",
      "position": {"x": 95.0, "y": 15.0, "z": 0.0},
      "moving": true,
      "last_observed_by": "uav_2"
    }
  ],
  "regions": [
    {
      "region_id": "east_sector",
      "center": {"x": 100.0, "y": 0.0, "z": 0.0},
      "status": "partially_explored",
      "risk_level": "medium"
    }
  ]
}
```

### Example output idea

```json
{
  "mission_id": "demo_001",
  "status": "recommendation_ready",
  "assignments": [
    {
      "task_id": "task_return_uav_3",
      "task_type": "return_home",
      "assigned_uav": "uav_3",
      "decision": "approved",
      "reasons": [
        "battery below return-home threshold",
        "localization is healthy",
        "communication link is sufficient"
      ]
    },
    {
      "task_id": "task_verify_target_07",
      "task_type": "verify_target",
      "assigned_uav": "uav_2",
      "decision": "approved",
      "reasons": [
        "already observing target",
        "closest suitable UAV",
        "zoom camera available",
        "battery reserve is sufficient"
      ]
    },
    {
      "task_id": "task_inspect_east_sector",
      "task_type": "inspect_region",
      "assigned_uav": "uav_1",
      "decision": "approved",
      "reasons": [
        "mapping capability available",
        "battery reserve is sufficient",
        "UAV is not assigned to the high-priority target"
      ]
    }
  ],
  "operator_message": "Return UAV 3, keep UAV 2 on target verification, and assign UAV 1 to the eastern-sector inspection."
}
```

The exact schema may be improved after repository inspection, but the core ideas must remain.

---

## 6. Required domain concepts

Codex should implement explicit domain objects.

### UAV state

Suggested fields:

```text
uav_id
status
battery_percent
position
link_quality
localization_healthy
capabilities
current_task
```

### Target state

Suggested fields:

```text
target_id
target_class
confidence
priority
position
moving
last_observed_by
```

### Region state

Suggested fields:

```text
region_id
center
status
risk_level
```

### Mission constraints

Suggested fields:

```text
return_home_battery_percent
critical_battery_percent
minimum_link_quality
require_operator_approval_for_high_risk
```

### Task

Suggested fields:

```text
task_id
task_type
priority
required_capabilities
target_id or region_id
status
assigned_uav
```

### Safety decision

Suggested fields:

```text
approved
risk_level
reasons
required_operator_approval
```

### Recommendation

Suggested fields:

```text
assignment
decision
reasons
alternatives
confidence or utility score
```

Use enums for bounded values where practical.

---

## 7. Initial mission skills

Phase 1 may use these simulated skills:

```text
inspect_region
verify_target
maintain_overwatch
track_target
return_home
hold_position
request_operator_review
```

These are logical mission actions only. They do not command flight hardware.

Every skill should define:

- Required capabilities.
- Basic preconditions.
- Safety conditions.
- Success meaning.
- Explanation text.

Example:

### `verify_target`

Required capability:

```text
rgb_camera or zoom_camera
```

Preconditions:

```text
target exists
target position is known
UAV is available
battery is above reserve
localization is healthy
link quality is sufficient
```

Output:

```text
approved recommendation or rejected recommendation with reasons
```

---

## 8. Deterministic planning rules for v1

Use simple, documented rules.

### Battery rules

- Battery below critical threshold:
  - Recommend immediate safe return or landing policy.
  - Do not assign a new mission task.

- Battery below return-home threshold:
  - Recommend return home.
  - Do not assign inspection, tracking, or overwatch.

### Health rules

- Exclude UAVs with unhealthy localization from navigation-dependent tasks.
- Exclude unavailable or failed UAVs.
- Avoid assigning new tasks when link quality is below the configured minimum.
- Keep the reason for every exclusion.

### Capability rules

- `inspect_region` requires mapping or camera capability.
- `verify_target` requires a suitable camera.
- `track_target` requires tracking capability.
- `maintain_overwatch` requires a camera and sufficient battery.

### Allocation rules

Use a transparent utility score based on:

```text
capability match
battery reserve
distance to task
current workload
link quality
continuity with an existing target observation
```

Do not hide weights. Put them in a configuration file or clearly named constants.

Possible initial formula:

```text
utility =
    capability_score
  + battery_score
  + link_score
  + continuity_bonus
  - normalized_distance_penalty
  - workload_penalty
```

Return a score breakdown for explainability.

### Human approval

Any recommendation marked high risk should be returned as:

```text
pending_operator_approval
```

The first version must never imply that approval occurred automatically.

---

## 9. Proposed file structure

Codex should inspect the repository conventions and then create something close to:

```text
06_AGENTIC_AUTONOMY/
├── README.md
├── mission_copilot/
│   ├── __init__.py
│   ├── models.py
│   ├── world_model.py
│   ├── task_factory.py
│   ├── allocator.py
│   ├── safety.py
│   ├── planner.py
│   ├── explain.py
│   ├── io.py
│   └── cli.py
├── config/
│   └── default_policy.json
├── examples/
│   ├── scenario_basic.json
│   ├── scenario_low_battery.json
│   └── scenario_unavailable_uav.json
└── tests/
    ├── test_io.py
    ├── test_allocator.py
    ├── test_safety.py
    └── test_planner.py
```

Do not create unnecessary frameworks.

If the repository already has a preferred packaging/testing pattern, follow it.

---

## 10. Command-line behavior

The desired user experience is:

```powershell
python -m mission_copilot.cli --scenario examples/scenario_basic.json
```

or an equivalent repository-compatible command.

Expected console output:

```text
MISSION COPILOT — demo_001

UAV 3:
RETURN HOME
Reason: battery 18% is below the 25% return threshold.

UAV 2:
VERIFY TARGET target_07
Reason: already observing the target, suitable sensor, sufficient battery.

UAV 1:
INSPECT REGION east_sector
Reason: available mapping capability and sufficient reserve.

Operator approval required: No
JSON result saved to: outputs/demo_001_recommendation.json
```

Generated outputs should be ignored by Git unless they are small fixed examples.

---

## 11. Testing requirements

At minimum, automated tests must prove:

1. Valid scenario JSON is parsed.
2. Invalid or incomplete JSON produces a useful error.
3. A low-battery UAV is assigned return-home.
4. A critical-battery UAV is never assigned a normal mission task.
5. An unavailable UAV is excluded.
6. A UAV without the required capability is excluded.
7. The nearest suitable UAV is preferred when other scores are equal.
8. Target-observation continuity gives the current observer a documented bonus.
9. A low-link or unhealthy-localization UAV is not assigned navigation tasks.
10. Every decision contains at least one explanation reason.
11. The same input produces the same output.
12. No test requires ROS 2, model weights, a GPU, or a network connection.

---

## 12. Documentation requirements

The subsystem README must explain:

- The problem it solves.
- What “Agentic UAV Mission Copilot” means.
- What the current version does.
- What it deliberately does not do.
- Architecture.
- Input schema.
- Output schema.
- Run command.
- Test command.
- Example result.
- Safety boundaries.
- Current limitations.
- Future ROS 2 integration.
- Future language-agent integration.
- Future event-based vision integration.

Documentation should be technical, clear, and natural. Avoid marketing claims.

---

## 13. Integration roadmap after Phase 1

### Phase 2 — ROS 2 adapter

Add a separate adapter that converts ROS 2 messages into the same domain objects used by the pure-Python core.

Possible future inputs:

```text
/uav_1/state
/uav_2/state
/uav_3/state
/swarm/targets
/swarm/map_status
/swarm/alerts
/operator/mission_objective
```

Possible future outputs:

```text
/swarm/mission_recommendations
/swarm/task_assignments
/swarm/safety_decisions
```

The deterministic core must remain reusable without ROS 2.

### Phase 3 — Dashboard integration

Display:

- Current mission objective.
- UAV availability.
- Recommended assignments.
- Rejected actions.
- Safety reasons.
- Operator approval controls.
- Mission history.

### Phase 4 — Language interface

A language model may translate operator language into a constrained mission schema.

The language model must not send flight commands.

```text
Operator language
→ constrained structured objective
→ schema validation
→ deterministic planner
→ safety validator
→ operator approval
```

### Phase 5 — Event-based vision integration

An event-based perception subsystem may later publish structured urgent observations:

```json
{
  "event_type": "fast_intruder",
  "target_id": "intruder_04",
  "bearing_deg": 122.0,
  "relative_speed_mps": 11.2,
  "collision_probability": 0.76,
  "time_to_conflict_sec": 1.8,
  "uncertainty": 0.12
}
```

The Mission Copilot would then:

```text
receive urgent event
→ update world model
→ pause lower-priority task
→ propose safe response
→ reallocate remaining task
→ request approval when required
```

Event-camera algorithms are not part of Phase 1.

---

## 14. Important non-goals

Do not implement these during the first coding task:

- Full LLM or VLM integration.
- Direct ROS 2 dependency.
- PX4 or ArduPilot commands.
- Autonomous motor control.
- Event-camera processing.
- Neural task allocation.
- Reinforcement learning.
- New YOLO training.
- Tracker replacement.
- Repository-wide restructuring.
- Security-sensitive network redesign.
- Company-specific credentials or paths.

The first objective is a small, correct, documented, testable mission intelligence core.

---

## 15. Codex execution sequence

Codex should follow this sequence.

### Step 1 — Audit

Report:

- Exact repository tree.
- Current Python organization.
- Current dependency method.
- Current tests.
- Existing data schemas in sender and bridge.
- Existing threat and mission-related fields.
- Existing documentation style.
- Conflicts with this proposal.

Do not modify files during the audit unless explicitly asked.

### Step 2 — Plan

Propose:

- Exact files to create.
- Existing files, if any, that need minimal edits.
- Data model.
- Test strategy.
- Run command.
- Risks and assumptions.

### Step 3 — Implement Phase 1

Create the pure-Python mission core, examples, tests, and documentation.

### Step 4 — Validate

Run:

- Syntax checks.
- Unit tests.
- Example CLI scenario.
- `git diff --check`.
- Any existing lightweight repository checks that do not require unavailable hardware or ROS 2.

### Step 5 — Report

Return:

- Files changed.
- Design decisions.
- Test outputs.
- Example recommendation.
- Known limitations.
- Exact next task.

---

## 16. Success criteria

Phase 1 is successful when:

- A fresh clone can run the Mission Copilot example on a normal personal computer.
- It does not require ROS 2, a GPU, weights, or a live UAV.
- It creates safe and explainable task recommendations.
- Low-battery and unhealthy UAVs are handled correctly.
- The output is deterministic.
- The logic is covered by tests.
- Existing Windows and ROS 2 code remains unchanged unless a very small documented integration reference is required.
- The design is ready for a later ROS 2 adapter.

---

## 17. First research question after implementation

Once the deterministic core works, the next useful research question is:

> How should live UAV detections, target tracks, battery state, communication quality, and event-based urgent alerts be transformed into a reliable shared world model that supports safe, explainable mission replanning?

That question connects the current prototype to agentic autonomy without prematurely placing an unconstrained language model in the control loop.
