@echo off
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
if defined UAV_YOLO_PYTHON (
  "%UAV_YOLO_PYTHON%" "%PROJECT_ROOT%\01_WINDOWS_AI\apps\uav_ai_control_panel.py"
) else (
  python "%PROJECT_ROOT%\01_WINDOWS_AI\apps\uav_ai_control_panel.py"
)
pause
