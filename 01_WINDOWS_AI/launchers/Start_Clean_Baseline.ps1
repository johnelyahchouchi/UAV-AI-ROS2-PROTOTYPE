param(
    [Parameter(Mandatory = $true)]
    [string]$Target,

    [int]$Port = 5010,

    [string]$Source = "",

    [double]$Confidence = 0.25,

    [double]$IoU = 0.45,

    [int]$ImageSize = 960,

    [int]$SendWidth = 960
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "portable_paths.ps1")

$Sender = $SenderScript
$Model = $ActiveDetectorModel
$VideoFolder = Join-Path $TestMediaDirectory "videos"

if ([string]::IsNullOrWhiteSpace($Source)) {
    $preferredVideo = Resolve-UavPath `
        "UAV_DEFAULT_VIDEO_PATH" `
        "06_TEST_MEDIA\videos\vehicles.mp4"

    if (Test-Path -LiteralPath $preferredVideo) {
        $Source = $preferredVideo
    }
    else {
        if (-not (Test-Path -LiteralPath $VideoFolder -PathType Container)) {
            throw "Test-media folder not found: $VideoFolder. Set UAV_TEST_MEDIA_DIR or pass -Source."
        }

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
elseif (-not [IO.Path]::IsPathRooted($Source)) {
    $Source = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Source))
}

if (-not (Test-Path -LiteralPath $Sender)) {
    throw "Sender not found: $Sender"
}

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Detector model not found: $Model. Set UAV_MODEL_PATH or place the model in the expected models directory."
}

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Video source not found: $Source. Set UAV_DEFAULT_VIDEO_PATH, set UAV_TEST_MEDIA_DIR, or pass -Source."
}

Write-Host ""
Write-Host "=== CLEAN UAV AI BASELINE ===" -ForegroundColor Cyan
Write-Host "Sender : $Sender"
Write-Host "Model  : $Model"
Write-Host "Source : $Source"
Write-Host "Target : $Target`:$Port"
Write-Host "Tracker: botsort.yaml"
Write-Host ""

& $PythonExecutable $Sender `
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
