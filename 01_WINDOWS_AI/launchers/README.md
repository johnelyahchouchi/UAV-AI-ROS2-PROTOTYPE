# Windows Launchers

| Launcher | Purpose |
|---|---|
| `Start_Live_Screen_Tester.bat` | Canonical presentation entry point. Opens the MP4 picker by default and forwards optional tester CLI arguments. |
| `Start_MC_Dropout_V2.bat` | Dedicated local V2 entry point. Opens native pickers for the trusted base detector, validated V2 checkpoint, and MP4; fails closed if V2 cannot be verified. |
| `start_yolo_sender.ps1` | Canonical parameterized Windows sender launcher. Requires external model/source and TLS configuration. |
| `Start_UAV_Windows_Sender.bat` | Double-click wrapper for the sender PowerShell launcher. |
| `Start_UAV_AI_Control_Panel.bat` | Optional GUI configuration for the same active sender. |

The recorded model-test dashboard has its own `model_test_dashboard/launch_dashboard.ps1`
because it is a separate batch-analysis application. All launchers resolve repository
code relative to their own location, use the verified `UAV_YOLO_PYTHON`, and avoid
workstation usernames or company-network defaults.

The live launcher inherits optional `UAV_MCDO_V2_MODEL_PATH`. When that separate
external checkpoint passes the trusted-hash and six-layer architecture checks, `U`
opens the V1/V2 method menu. Without it, `U` continues to run V1 directly.

For a V2-focused local run after cloning or pulling the repository, double-click
`Start_MC_Dropout_V2.bat`. It does not require model or video environment variables:
select both external `.pt` files and an MP4 in the three dialogs. The files remain
outside Git. Both model digests must already be approved in
`00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`; the launcher never bypasses trust or
architecture validation.
