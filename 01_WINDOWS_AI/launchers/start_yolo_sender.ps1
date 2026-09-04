param(
    [string]$Target = $(if ($env:UAV_BRIDGE_HOST) { $env:UAV_BRIDGE_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:UAV_BRIDGE_PORT) { [int]$env:UAV_BRIDGE_PORT } else { 5010 }),
    [string]$Source = "",
    [string]$Model = $env:UAV_MODEL_PATH,
    [double]$Confidence = 0.25,
    [double]$IoU = 0.45,
    [int]$ImageSize = 640,
    [int]$Stride = 2,
    [int]$SendWidth = 960
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Sender = Join-Path $ProjectRoot "01_WINDOWS_AI\apps\win_yolo_tcp_sender_botsort_threat.py"
$PythonExe = $env:UAV_YOLO_PYTHON

if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Set UAV_YOLO_PYTHON to the verified Python executable. Bare python is not used."
}
if ([string]::IsNullOrWhiteSpace($Model) -or -not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Set UAV_MODEL_PATH to an existing, trusted local .pt checkpoint."
}
if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = Join-Path $ProjectRoot "06_TEST_MEDIA\videos\vehicles.mp4"
}

$SenderArguments = @(
    "--target", $Target,
    "--port", $Port,
    "--source", $Source,
    "--model", $Model,
    "--conf", $Confidence,
    "--iou", $IoU,
    "--imgsz", $ImageSize,
    "--stride", $Stride,
    "--send_width", $SendWidth,
    "--show", 1,
    "--military_only", 1,
    "--tracker", "botsort.yaml"
)

& $PythonExe $Sender @SenderArguments
if ($LASTEXITCODE -ne 0) {
    throw "The sender stopped with exit code $LASTEXITCODE."
}
