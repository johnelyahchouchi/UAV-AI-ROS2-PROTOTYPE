# 08 - Testing and Validation

Automated tests are separated by dependency boundary.

```powershell
# Shared security, live tester, V1 uncertainty, and static dashboard tests
py -3.11 -m pytest -q .\tests

# Recorded model-test dashboard (controlled YOLO environment)
Push-Location .\01_WINDOWS_AI\model_test_dashboard
$env:PYTHONPATH = (Join-Path $PWD "src")
& $env:UAV_YOLO_PYTHON -m pytest -q
Pop-Location

# Simulation-first Mission Copilot
Push-Location .\06_AGENTIC_AUTONOMY
py -3.11 -m pytest -q
Pop-Location
```

The root suites cover configuration/path validation, input and URL validation, model
integrity, safe ZIP/CSV/image handling, TLS transport, live source/region/device behavior,
one-pass normal processing, exact-frame `U` inspection, V1 perturbations/matching/metrics,
V2 architecture and inference-state enforcement, 20-pass clustering/competition metrics,
selector/failure recovery, and static ROS dashboard source expectations. Fake models,
fake detectors, and synthetic media avoid
model, GPU, network, camera, ROS 2, and UAV dependencies.

Repository-wide `compileall`, PowerShell parser checks, JSON/YAML parsing, Markdown-link
checks, launcher `--help` checks, private-key scans, dependency checks, and the configured
Bandit/pip-audit CI provide additional validation.

## Manual validation still required

Automated tests do not establish detector accuracy, flight safety, ROS 2 deployment
correctness, SROS2 policy correctness, or real-time performance. Before a presentation,
use the external allowlisted checkpoint and operator-selected video to run the checklist
in `LIVE_SCREEN_MODEL_TESTER.md`. V2 additionally needs the exact trained checkpoint and
a real stochastic GPU smoke test. Validate the sender/bridge/dashboards separately on
the actual Ubuntu ROS 2 deployment.
