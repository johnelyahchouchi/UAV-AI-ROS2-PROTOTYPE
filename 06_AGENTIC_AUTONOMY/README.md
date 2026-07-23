# Agentic UAV Mission Copilot — Phase 1

An offline, deterministic mission-task generator, allocator, safety validator, and explanation engine. It uses only the Python standard library at runtime and has no ROS 2, GPU, model, camera, network, PX4, or LLM dependency.

## Run

On this workstation, bare `python` launches the Amesim Python 2.7 environment. Do not use it for this subsystem. Verify and invoke the installed CPython 3.11 interpreter explicitly from the repository root in PowerShell:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe' --version
$env:PYTHONPATH="$PWD\06_AGENTIC_AUTONOMY\src"
& 'C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe' -m agentic_autonomy --scenario 06_AGENTIC_AUTONOMY\scenarios\basic_reconnaissance.json --output 06_AGENTIC_AUTONOMY\outputs\basic_reconnaissance_plan.json --verbose
```

## Test

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe' -m pytest 06_AGENTIC_AUTONOMY\tests -q
```

See `docs/` for architecture, schemas, policy, safety rules, and limitations.

## Dynamic replanning

Phase 2 processes an ordered file of complete mission-state snapshots:

```powershell
$env:PYTHONPATH="$PWD\06_AGENTIC_AUTONOMY\src"
& 'C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe' -m agentic_autonomy.replan_cli --sequence 06_AGENTIC_AUTONOMY\scenarios\replanning\battery_degradation.json --policy 06_AGENTIC_AUTONOMY\config\default_policy.json --output 06_AGENTIC_AUTONOMY\outputs\replanning\battery_degradation_result.json --verbose
```

See `docs/DYNAMIC_REPLANNING.md` for lifecycle, hysteresis, workload, advisory return-home, and fingerprint rules.
