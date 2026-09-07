# Phase 3 — ROS 2 Mission-State Adapter

## Scope and safety boundary

Phase 3 converts a strict ordered event stream into the complete Phase 2
`MissionHistory` contract. The core is pure Python and works offline. The optional
ROS 2 shell is an adapter around that core.

Outputs are limited to:

- complete mission-state snapshot history;
- structured adapter diagnostics;
- existing Phase 2 advisory replanning results.

There is no command publisher or flight-command API. Return-home and landing
recommendations remain advisory records and are never executed.

The subsystem version remains `0.2.0` for Phase 1/2 compatibility. The independent
adapter contract and diagnostics version is `1.0`; the adapter implementation
version is `0.1.0`.

## Canonical stream and event contract

`schemas/adapter_event.schema.json` defines a stream object:

| Field | Rule |
|---|---|
| `schema_version` | Exactly `"1.0"` |
| `mission_id` | Non-empty string shared by all events |
| `events` | Non-empty array in strictly increasing `sequence` order |

Every event has exactly these top-level fields:

| Field | Rule |
|---|---|
| `schema_version` | Exactly `"1.0"` |
| `mission_id` | Non-empty string |
| `event_id` | Non-empty idempotency identifier |
| `sequence` | Nonnegative integer; adapter ingestion order |
| `event_type` | One of the event types below |
| `observed_at` | `{clock_id, sec, nanosec}` |
| `source` | Source metadata |
| `payload` | Strict payload selected by `event_type` |

`observed_at.sec` is nonnegative, `nanosec` is `0..999999999`, and every event in
one mission uses the same non-empty `clock_id`. It is a deterministic scenario or
ROS clock value; the core never reads the wall clock.

Source metadata requires `source_id` and may contain `source_node`, `topic`,
`message_type`, `source_uav_id`, `source_session_id`, `source_timestamp`, and
nonnegative `upstream_sequence`. The ROS shell replaces an incoming event's
`sequence` with its serialized ingestion sequence and preserves the producer value
as `source.upstream_sequence`. `source_timestamp` is preserved separately from the
adapter's comparison timestamp.

Unknown fields, numeric strings, booleans used as numbers, non-finite values,
invalid enums, empty identifiers, malformed timestamps, and out-of-order state
updates are rejected with an `AdapterEventError`.

`link_state: LOST` must include `link_quality: 0.0` in the same update. This
prevents a prior high-quality value from surviving an explicit loss declaration;
the Phase 1 safety rule therefore rejects the UAV while Phase 2 can still emit its
configured link-loss advisory.

## Event types

| Event | Purpose and strict payload |
|---|---|
| `MISSION_CONFIGURED` | Static `scenario_id`, regions, operating region, exclusion regions; must be first |
| `REGION_UPDATED` | One explicit Phase 1-compatible region |
| `UAV_CONFIGURED` | `uav_id`, capabilities, positive `max_workload`, optional positive `max_task_distance` |
| `UAV_STATE_UPDATED` | `uav_id` and at least one dynamic field: position, availability, battery, link, workload, mission status, or current target |
| `TARGET_OBSERVED` | Explicit global identity or local identity metadata, class/confidence/status, optional world position/priority, capabilities, continuity |
| `TARGET_STATE_UPDATED` | Existing target id, required status, and optional planning-state changes |
| `TASK_CREATED` | Explicit request, reference, priority, capabilities, reason, and initial `PENDING` or `ACTIVE` lifecycle |
| `TASK_UPDATED` | Existing nonterminal request and at least one priority/capability change |
| `TASK_LIFECYCLE_CHANGED` | Existing request, lifecycle state, optional reason |
| `SNAPSHOT_TICK` | Periodic deterministic trigger with optional reason |
| `SNAPSHOT_REQUESTED` | Explicit deterministic trigger with optional reason |

Target tasks require one `target_id` and forbid `region_id`. Region tasks require
one `region_id` and forbid `target_id`. Only `TASK_CREATED` creates planning work.
`TARGET_OBSERVED` never creates or activates a task, regardless of class or
confidence.

Task transitions are:

```text
PENDING -> PENDING | ACTIVE | CANCELLED
ACTIVE  -> ACTIVE | COMPLETED | CANCELLED
COMPLETED -> COMPLETED
CANCELLED -> CANCELLED
```

Completed and cancelled requests remain in every later complete snapshot because
Phase 2 forbids disappearing requests.

## Canonical battery representation

The normalized event contract uses only `battery_percent`, in percent from `0` to
`100`. Units are never inferred from the number.

The `sensor_msgs/BatteryState` mapper alone converts its specified fractional
`percentage` value from `0.0..1.0` to `0..100`. A missing, boolean, non-finite, or
out-of-range mapper value is rejected.

The policy's `battery.clamp_tolerance_percent` may clamp only a tiny floating-point
overshoot just outside `0..100`. The default is `0.001` percentage points. Every
clamp emits `BATTERY_CLAMPED_WITHIN_TOLERANCE`; larger deviations fail. This is a
numerical tolerance, not unit inference.

## Freshness and snapshot projection

`config/ros2_adapter_policy.json` stores independent research-default freshness
thresholds for position, availability, battery, link, workload, and targets.
Freshness is calculated from event timestamps in the common mission clock:

```text
age = snapshot timestamp - field observation timestamp
fresh when age <= configured threshold
stale when age > configured threshold
```

A UAV missing any required dynamic field is omitted and cannot be eligible. A UAV
with a stale position, availability, battery, link state/quality, workload, or
mission status retains the last value only for audit output and is projected as
`UNAVAILABLE`. Unknown availability/link/mission status and explicitly unsafe
states are also projected `UNAVAILABLE`. No safe value is fabricated.

UAV positions and target positions must use the exact configured planning frame,
initially `mission_local`. There is no TF2 conversion. A stale target is projected
`LOST`. A target referenced by a task must have a valid planning position and
priority or snapshot generation fails clearly.

The builder sorts UAVs, targets, regions, requests, capabilities, and lifecycle
records. Snapshot IDs use scenario identity plus a zero-padded monotonic snapshot
sequence. A produced history is passed through the in-memory Phase 2 parser before
publication.

## Target identity

Identity precedence is:

1. an explicitly supplied globally stable `global_target_id`;
2. `target:<source UAV>:<source session>:<local track>`.

Targets are never fused across UAVs. A tracker restart must use a new source
session. A legacy observation without a global ID or without both session and local
track identity emits `TARGET_IDENTITY_INSUFFICIENT` and cannot enter planner target
state.

Legacy bounding boxes and pixel centers are not converted to world coordinates.
They may be retained as diagnostic observations, but cannot be referenced by a
planning task until explicit planning-frame state is supplied.

## Triggering, order, limits, and determinism

The hybrid trigger policy emits a snapshot for:

- explicit `SNAPSHOT_REQUESTED`;
- a periodic `SNAPSHOT_TICK` when state is dirty;
- an unchanged heartbeat at the configured interval;
- configured immediate safety changes such as unavailable UAV, lost link,
  below-reserve battery, or terminal task lifecycle.

The ROS shell serializes callbacks through one queue and uses a lock-protected
monotonic ingestion counter. The state store applies one event at a time. Duplicate
`event_id` values with identical normalized content are ignored idempotently;
conflicting duplicates fail.

Determinism means identical canonical output for an identical normalized ordered
event stream and policies. It does not promise identical ordering for inherently
nondeterministic DDS arrivals before the shell assigns its ingestion sequence.

The prototype accumulates full Phase 2 history and never truncates it. Policy
limits bound message bytes, snapshots, UAVs, targets, tasks, and diagnostics. The
operation fails before a limit would be exceeded. Persistent checkpointing and a
bounded-history protocol are future work.

## Policy and schemas

- `config/ros2_adapter_policy.json` — default policy and ROS topic bindings.
- `schemas/ros2_adapter_policy.schema.json` — strict policy schema.
- `schemas/adapter_event.schema.json` — strict stream/event/payload schema.
- `schemas/mission_state_sequence.schema.json` — existing Phase 2 output contract.

The legacy `/uav_1/coco_detections` binding intentionally defaults
`source_session_id` to `null`. In that state, local tracker IDs are diagnostic only.
Configure a new stable session value whenever that tracker process starts if local
identity is to be used.

## Offline demonstrations

Activate any Python 3.11+ environment, then run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONPATH="$PWD\06_AGENTIC_AUTONOMY\src"

python -m agentic_autonomy.ros2_adapter `
  --events 06_AGENTIC_AUTONOMY\scenarios\adapter\basic_adapter_events.json `
  --adapter-policy 06_AGENTIC_AUTONOMY\config\ros2_adapter_policy.json `
  --planner-policy 06_AGENTIC_AUTONOMY\config\default_policy.json `
  --snapshot-output 06_AGENTIC_AUTONOMY\outputs\adapter_basic_history.json `
  --diagnostics-output 06_AGENTIC_AUTONOMY\outputs\adapter_basic_diagnostics.json `
  --replanning-output 06_AGENTIC_AUTONOMY\outputs\adapter_basic_replanning.json `
  --verbose
```

Replace `basic_adapter_events.json` with `stale_link_loss_events.json` and use
corresponding output names for the stale/link-loss demonstration.

## Future ROS node launch

No ament package or custom message is created in Phase 3. In a future Ubuntu ROS 2
environment with the relevant standard message packages sourced and this `src`
directory on `PYTHONPATH`, the expected launch command is:

```bash
python3 -m agentic_autonomy.ros2_adapter.ros_node \
  --adapter-policy 06_AGENTIC_AUTONOMY/config/ros2_adapter_policy.json \
  --planner-policy 06_AGENTIC_AUTONOMY/config/default_policy.json
```

The shell subscribes to strict `std_msgs/String` canonical events and any
configured legacy detection, `BatteryState`, and `PoseStamped` topics. It provides
a standard `Trigger` snapshot service and publishes `String` snapshot,
diagnostics, and advisory documents. ROS imports and resources are created only
after this command calls `main()`.
