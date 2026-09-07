@echo off
setlocal
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"

if not defined UAV_YOLO_PYTHON (
  for %%I in ("%PROJECT_ROOT%\..\UAV_YOLO_ENV\Scripts\python.exe") do if exist "%%~fI" set "UAV_YOLO_PYTHON=%%~fI"
)
if not exist "%UAV_YOLO_PYTHON%" (
  echo ERROR: Set UAV_YOLO_PYTHON to the verified Python executable.
  pause
  exit /b 2
)
"%UAV_YOLO_PYTHON%" "%~dp0..\apps\uav_ai_control_panel.py"
set "EXIT_CODE=%ERRORLEVEL%"
pause
endlocal & exit /b %EXIT_CODE%
