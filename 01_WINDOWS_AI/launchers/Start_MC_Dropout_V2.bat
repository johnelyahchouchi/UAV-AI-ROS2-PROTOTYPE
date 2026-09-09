@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "LIVE_TESTER=%~dp0..\apps\live_screen_model_tester.py"

if not defined UAV_YOLO_PYTHON if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "UAV_YOLO_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not defined UAV_YOLO_PYTHON for %%I in ("%PROJECT_ROOT%\..\UAV_YOLO_ENV\Scripts\python.exe") do if exist "%%~fI" set "UAV_YOLO_PYTHON=%%~fI"

if not exist "%LIVE_TESTER%" (
  echo ERROR: Live screen tester application was not found.
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

echo MC Dropout V2 local tester
echo 1. Select the trusted base detector checkpoint.
echo 2. Select the separately trained V2 MC Dropout checkpoint.
echo 3. Select an MP4 video.
echo.
echo Both checkpoints must be allowlisted in 00_PROJECT_GUIDE\ACTIVE_MODEL_HASHES.csv.
echo The V2 checkpoint must contain the validated six Dropout2d layers.
echo V1 and V2 update in separate live panels. U opens the optional detailed report.
echo.

"%UAV_YOLO_PYTHON%" "%LIVE_TESTER%" --select-model --select-mcdo-v2-model --require-mcdo-v2 --select-video --loop-video --continuous-uncertainty %*

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo MC Dropout V2 tester stopped with exit code %EXIT_CODE%.
  pause
)

endlocal & exit /b %EXIT_CODE%
