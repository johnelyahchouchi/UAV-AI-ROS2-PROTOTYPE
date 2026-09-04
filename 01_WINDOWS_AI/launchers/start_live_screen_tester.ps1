param(
    [string]$Model = $env:UAV_MODEL_PATH,
    [int]$Monitor = 1,
    [switch]$ListMonitors,
    [switch]$SelectRegion,
    [int]$Left = -1,
    [int]$Top = -1,
    [int]$Width = -1,
    [int]$Height = -1,
    [double]$Confidence = 0.25,
    [double]$IoU = 0.45,
    [int]$ImageSize = 640,
    [string]$Device = "auto",
    [double]$MaxFPS = 0,
    [string]$Classes = "all",
    [switch]$TankOnly,
    [string]$TestFrame = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Tester = Join-Path $ProjectRoot "01_WINDOWS_AI\apps\live_screen_model_tester.py"
$PythonExe = $env:UAV_YOLO_PYTHON

if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Set UAV_YOLO_PYTHON to the verified Python executable. Bare python is not used."
}

$TesterArguments = @()
if ($ListMonitors) {
    $TesterArguments += "--list-monitors"
} else {
    if ([string]::IsNullOrWhiteSpace($Model) -or -not (Test-Path -LiteralPath $Model -PathType Leaf)) {
        throw "Set UAV_MODEL_PATH or pass -Model with an existing, trusted local .pt checkpoint."
    }
    $TesterArguments += @(
        "--model", $Model,
        "--monitor", $Monitor,
        "--conf", $Confidence,
        "--iou", $IoU,
        "--imgsz", $ImageSize,
        "--device", $Device,
        "--max-fps", $MaxFPS,
        "--classes", $Classes
    )

    $ManualNames = @("Left", "Top", "Width", "Height")
    $ManualProvided = @($ManualNames | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count
    if ($ManualProvided -ne 0 -and $ManualProvided -ne 4) {
        throw "Pass -Left, -Top, -Width, and -Height together."
    }
    if ($ManualProvided -eq 4) {
        if ($SelectRegion) {
            throw "Manual coordinates cannot be combined with -SelectRegion."
        }
        $TesterArguments += @(
            "--left", $Left,
            "--top", $Top,
            "--width", $Width,
            "--height", $Height
        )
    } elseif ($SelectRegion) {
        $TesterArguments += "--select-region"
    }
    if ($TankOnly) {
        $TesterArguments += "--tank-only"
    }
    if (-not [string]::IsNullOrWhiteSpace($TestFrame)) {
        $TesterArguments += @("--test-frame", $TestFrame)
    }
}

& $PythonExe $Tester @TesterArguments
if ($LASTEXITCODE -ne 0) {
    throw "The live screen model tester stopped with exit code $LASTEXITCODE."
}
