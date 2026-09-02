# Agentic UAV Mission Copilot

An offline, deterministic mission-task generator, allocator, safety validator, and explanation engine. It uses only the Python standard library at runtime and has no ROS 2, GPU, model, camera, network, PX4, or LLM dependency.

## Run

Activate any Python 3.11+ environment and run from the repository root:

```powershell
python --version
$env:PYTHONPATH="$PWD\06_AGENTIC_AUTONOMY\src"
python -m agentic_autonomy --scenario 06_AGENTIC_AUTONOMY\scenarios\basic_reconnaissance.json --output 06_AGENTIC_AUTONOMY\outputs\basic_reconnaissance_plan.json --verbose
```

## Test

```powershell
python -m pytest 06_AGENTIC_AUTONOMY\tests -q
```

See `docs/` for architecture, schemas, policy, safety rules, and limitations.

## Dynamic replanning

Phase 2 processes an ordered file of complete mission-state snapshots:

```powershell
$env:PYTHONPATH="$PWD\06_AGENTIC_AUTONOMY\src"
python -m agentic_autonomy.replan_cli --sequence 06_AGENTIC_AUTONOMY\scenarios\replanning\battery_degradation.json --policy 06_AGENTIC_AUTONOMY\config\default_policy.json --output 06_AGENTIC_AUTONOMY\outputs\replanning\battery_degradation_result.json --verbose
```

See `docs/DYNAMIC_REPLANNING.md` for lifecycle, hysteresis, workload, advisory return-home, and fingerprint rules.

## Mission-state adapter

Phase 3 adds a ROS-independent event-to-snapshot adapter plus an optional, isolated
ROS 2 shell. The offline replay is the supported demonstration in this repository:

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

The adapter accepts explicit mission configuration, UAV state, target observations,
task/lifecycle events, and snapshot triggers. A detection can update target state
but never creates a task. Missing or stale required UAV telemetry is never replaced
with a safe-looking default. See `docs/ROS2_MISSION_STATE_ADAPTER.md` for the exact
contract, policy, identity, freshness, determinism, and future ROS launch command.
