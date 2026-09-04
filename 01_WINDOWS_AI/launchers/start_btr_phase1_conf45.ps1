Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Launcher = Join-Path $PSScriptRoot "start_yolo_sender.ps1"
$Model = $env:UAV_BTR_MODEL_PATH
$Source = $env:UAV_TEST_VIDEO

if ([string]::IsNullOrWhiteSpace($Model)) {
    throw "Set UAV_BTR_MODEL_PATH to a trusted local checkpoint."
}
if ([string]::IsNullOrWhiteSpace($Source)) {
    throw "Set UAV_TEST_VIDEO to the local BTR test video."
}

& $Launcher -Model $Model -Source $Source -Confidence 0.45 -ImageSize 640 -Stride 1
