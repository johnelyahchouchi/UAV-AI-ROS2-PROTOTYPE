# Windows Launchers

| Launcher | Purpose |
|---|---|
| `Start_UAV_Prototype_GUI.bat` | Full desktop control center and readiness dashboard. Configures and launches the maintained live tester, V1/V2 inspection, recorded dashboard, secure sender, and Mission Copilot. |
| `Start_Live_Screen_Tester.bat` | Canonical presentation entry point. Opens the MP4 picker by default and forwards optional tester CLI arguments. |
| `Start_MC_Dropout_V2.bat` | Dedicated local V2 entry point. Opens native pickers for the trusted base detector, validated V2 checkpoint, and MP4; fails closed if V2 cannot be verified. |
| `start_yolo_sender.ps1` | Canonical parameterized Windows sender launcher. Requires external model/source and TLS configuration. |
| `Start_UAV_Windows_Sender.bat` | Double-click wrapper for the sender PowerShell launcher. |
| `Start_UAV_AI_Control_Panel.bat` | Optional GUI configuration for the same active sender. |

The recorded model-test dashboard has its own `model_test_dashboard/launch_dashboard.ps1`
because it is a separate batch-analysis application. All launchers resolve repository
code relative to their own location, use the verified `UAV_YOLO_PYTHON`, and avoid
workstation usernames or company-network defaults.

The live launcher enables the continuous uncertainty workspace and inherits optional
`UAV_MCDO_V2_MODEL_PATH`. V1 and V2 occupy separate live cards. When the external V2
checkpoint passes the trusted-hash and six-layer architecture checks, the V2 card
updates automatically; otherwise it remains explicitly unavailable. `U` is optional
and opens the detailed exact-frame method report.

For a V2-focused local run after cloning or pulling the repository, double-click
`Start_MC_Dropout_V2.bat`. It does not require model or video environment variables:
select both external `.pt` files and an MP4 in the three dialogs. The files remain
outside Git. Both model digests must already be approved in
`00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`; the launcher never bypasses trust or
architecture validation.

For the complete graphical workflow, run `Start_UAV_Prototype_GUI.bat`. The GUI stores
workstation-only settings under ignored `08_OUTPUTS/control_center/`, never stores a
credential-bearing stream URL, and keeps model weights, media, certificates, and keys
outside Git. The Activity Log captures child-process output without constructing shell
commands.
