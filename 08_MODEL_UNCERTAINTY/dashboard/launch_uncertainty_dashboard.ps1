Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$DashboardRoot = $PSScriptRoot
$UncertaintyRoot = Split-Path -Parent $DashboardRoot
$DefaultPython = Join-Path $env:USERPROFILE "Desktop\UAV_YOLO_ENV\Scripts\python.exe"
$PythonExe = $env:UAV_YOLO_PYTHON

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = $DefaultPython
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw @"
The dedicated UAV YOLO Python executable was not found.

Set UAV_YOLO_PYTHON to the full path of the verified interpreter, or restore:
$DefaultPython

The launcher will not fall back to bare python or create another environment.
"@
}

$CoreSrc = Join-Path $UncertaintyRoot "src"
$DashboardSrc = Join-Path $DashboardRoot "src"
$env:PYTHONPATH = "$CoreSrc;$DashboardSrc"
$DashboardPort = $env:UAV_UNCERTAINTY_DASHBOARD_PORT

if ([string]::IsNullOrWhiteSpace($DashboardPort)) {
    $DashboardPort = "7861"
}

Write-Host ""
Write-Host "=== UAV MODEL UNCERTAINTY DASHBOARD ===" -ForegroundColor Cyan
Write-Host "Python executable: $PythonExe"

$verificationCode = @'
import sys
import cv2
import gradio
import matplotlib
import numpy
import torch
import torchvision
import ultralytics
import uav_uncertainty
import uav_uncertainty_dashboard

print("Python:", sys.version.split()[0])
print("Gradio:", gradio.__version__)
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if numpy.__version__.split('.')[0] != '1':
    raise SystemExit("Unsupported NumPy major version: expected the verified NumPy 1.x environment.")

if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
else:
    print("GPU 0: unavailable; Auto will follow Ultralytics device selection and CPU remains selectable.")
'@

$verificationCode | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "The dedicated UAV YOLO environment verification failed."
}

Write-Host ""
Write-Host "Starting local dashboard at http://127.0.0.1:$DashboardPort" -ForegroundColor Green
Write-Host ""

& $PythonExe -m uav_uncertainty_dashboard.app

if ($LASTEXITCODE -ne 0) {
    throw "The uncertainty dashboard stopped with exit code $LASTEXITCODE."
}
