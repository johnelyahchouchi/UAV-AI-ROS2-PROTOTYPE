param(
    [string]$Target = $(if ($env:UAV_BRIDGE_HOST) { $env:UAV_BRIDGE_HOST } else { "127.0.0.1" }),

    [int]$Port = 5010,

    [string]$Source = "",

    [string]$Model = $env:UAV_MODEL_PATH,

    [double]$Confidence = 0.25,

    [double]$IoU = 0.45,

    [int]$ImageSize = 960,

    [int]$SendWidth = 960
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$Sender = Join-Path $ProjectRoot `
    "01_WINDOWS_AI\apps\win_yolo_tcp_sender_botsort_threat.py"

$VideoFolder = Join-Path $ProjectRoot `
    "06_TEST_MEDIA\videos"

if ([string]::IsNullOrWhiteSpace($Source)) {
    $preferredVideo = Join-Path $VideoFolder "vehicles.mp4"

    if (Test-Path -LiteralPath $preferredVideo) {
        $Source = $preferredVideo
    }
    else {
        $firstVideo = Get-ChildItem $VideoFolder `
            -File `
            -Include *.mp4, *.avi, *.mov |
            Select-Object -First 1

        if (-not $firstVideo) {
            throw "No test video was found in: $VideoFolder"
        }

        $Source = $firstVideo.FullName
    }
}

if (-not (Test-Path -LiteralPath $Sender)) {
    throw "Sender not found: $Sender"
}

if ([string]::IsNullOrWhiteSpace($Model) -or -not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Set UAV_MODEL_PATH to an existing, trusted local detector checkpoint."
}

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Video source not found: $Source"
}

$Python = $env:UAV_YOLO_PYTHON
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Set UAV_YOLO_PYTHON to the verified Python executable. Bare python is not used."
}

Write-Host ""
Write-Host "=== CLEAN UAV AI BASELINE ===" -ForegroundColor Cyan
Write-Host "Sender : $Sender"
Write-Host "Model  : $Model"
Write-Host "Source : configured"
Write-Host "Target : $Target`:$Port"
Write-Host "Tracker: botsort.yaml"
Write-Host ""

& $Python $Sender `
    --target $Target `
    --port $Port `
    --source $Source `
    --model $Model `
    --conf $Confidence `
    --iou $IoU `
    --imgsz $ImageSize `
    --stride 1 `
    --send_width $SendWidth `
    --show 1 `
    --military_only 1 `
    --tracker "botsort.yaml"

if ($LASTEXITCODE -ne 0) {
    throw "The sender stopped with exit code $LASTEXITCODE."
}
