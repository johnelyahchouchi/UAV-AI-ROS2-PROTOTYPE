Set-StrictMode -Version Latest

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

function Resolve-UavPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentVariable,

        [Parameter(Mandatory = $true)]
        [string]$RepositoryDefault
    )

    $configured = [Environment]::GetEnvironmentVariable($EnvironmentVariable)
    if ([string]::IsNullOrWhiteSpace($configured)) {
        if ([IO.Path]::IsPathRooted($RepositoryDefault)) {
            return [IO.Path]::GetFullPath($RepositoryDefault)
        }

        return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $RepositoryDefault))
    }

    if ([IO.Path]::IsPathRooted($configured)) {
        return [IO.Path]::GetFullPath($configured)
    }

    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $configured))
}

$SenderScript = Join-Path $ProjectRoot "01_WINDOWS_AI\apps\win_yolo_tcp_sender_botsort_threat.py"
$ModelsDirectory = Resolve-UavPath "UAV_MODELS_DIR" "03_MODELS"
$TestMediaDirectory = Resolve-UavPath "UAV_TEST_MEDIA_DIR" "06_TEST_MEDIA"
$ActiveDetectorModel = Resolve-UavPath `
    "UAV_MODEL_PATH" `
    (Join-Path $ModelsDirectory "active\detector\military_kaggle_v1.pt")
$BaseYoloModel = Resolve-UavPath `
    "UAV_BASE_MODEL_PATH" `
    (Join-Path $ModelsDirectory "base_weights\yolov8n.pt")
$BtrModel = Resolve-UavPath `
    "UAV_BTR_MODEL_PATH" `
    (Join-Path $ModelsDirectory "experimental\btr_best_v2.pt")

$configuredPython = [Environment]::GetEnvironmentVariable("UAV_YOLO_PYTHON")
if ([string]::IsNullOrWhiteSpace($configuredPython)) {
    $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
elseif ([IO.Path]::IsPathRooted($configuredPython)) {
    $PythonExecutable = [IO.Path]::GetFullPath($configuredPython)
}
else {
    $PythonExecutable = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $configuredPython))
}

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable. Set UAV_YOLO_PYTHON."
}
