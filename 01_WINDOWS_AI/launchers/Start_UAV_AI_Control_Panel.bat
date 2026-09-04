@echo off
if "%UAV_YOLO_PYTHON%"=="" (
  echo Set UAV_YOLO_PYTHON to the verified Python executable.
  exit /b 1
)
"%UAV_YOLO_PYTHON%" "%~dp0..\apps\uav_ai_control_panel.py"
pause
