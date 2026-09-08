@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "CONTROL_CENTER=%~dp0..\apps\uav_prototype_control_center.py"

if not defined UAV_YOLO_PYTHON if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "UAV_YOLO_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not defined UAV_YOLO_PYTHON for %%I in ("%PROJECT_ROOT%\..\UAV_YOLO_ENV\Scripts\python.exe") do if exist "%%~fI" set "UAV_YOLO_PYTHON=%%~fI"

if not exist "%CONTROL_CENTER%" (
  echo ERROR: UAV Prototype Control Center was not found.
  pause
  exit /b 2
)

if not defined UAV_YOLO_PYTHON (
  echo ERROR: No verified Python environment was found.
  echo Create .venv from requirements-windows.txt or set UAV_YOLO_PYTHON.
  pause
  exit /b 2
)

if not exist "%UAV_YOLO_PYTHON%" (
  echo ERROR: UAV_YOLO_PYTHON does not point to an existing file.
  pause
  exit /b 2
)

"%UAV_YOLO_PYTHON%" "%CONTROL_CENTER%" %*

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo UAV Prototype Control Center stopped with exit code %EXIT_CODE%.
  pause
)

endlocal & exit /b %EXIT_CODE%
