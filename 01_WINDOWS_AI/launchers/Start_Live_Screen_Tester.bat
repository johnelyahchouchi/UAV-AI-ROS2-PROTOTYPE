@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "LIVE_TESTER=%~dp0..\apps\live_screen_model_tester.py"

if not defined UAV_YOLO_PYTHON (
  for %%I in ("%PROJECT_ROOT%\..\UAV_YOLO_ENV\Scripts\python.exe") do if exist "%%~fI" set "UAV_YOLO_PYTHON=%%~fI"
)

if not defined UAV_MODEL_PATH (
  for %%I in ("%PROJECT_ROOT%\05_TRAINING\detection_runs\military_kaggle_yolov8s_v1\weights\best.pt") do if exist "%%~fI" set "UAV_MODEL_PATH=%%~fI"
)

if not exist "%LIVE_TESTER%" (
  echo ERROR: Live screen tester application was not found.
  pause
  exit /b 2
)

if not defined UAV_YOLO_PYTHON (
  echo ERROR: The verified UAV_YOLO_ENV was not found next to the repository.
  echo Set UAV_YOLO_PYTHON to the verified Python executable and try again.
  pause
  exit /b 2
)

if not exist "%UAV_YOLO_PYTHON%" (
  echo ERROR: UAV_YOLO_PYTHON does not point to an existing file.
  pause
  exit /b 2
)

if "%~1"=="" (
  if not defined UAV_MODEL_PATH (
    echo ERROR: The local military detector was not found.
    echo Set UAV_MODEL_PATH to an existing trusted .pt checkpoint and try again.
    pause
    exit /b 2
  )
  if not exist "%UAV_MODEL_PATH%" (
    echo ERROR: UAV_MODEL_PATH does not point to an existing file.
    pause
    exit /b 2
  )

  echo Select an MP4 video in the file picker to start continuous detection.
  echo Controls: Q or ESC quit, P pause, S screenshot, H toggle HUD.
  "%UAV_YOLO_PYTHON%" "%LIVE_TESTER%" --select-video --loop-video
) else (
  "%UAV_YOLO_PYTHON%" "%LIVE_TESTER%" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Live screen tester stopped with exit code %EXIT_CODE%.
  pause
)

endlocal & exit /b %EXIT_CODE%
