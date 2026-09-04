Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Launcher = Join-Path $PSScriptRoot "start_yolo_sender.ps1"
Write-Host "1 = Default trusted detector + repository test video"
Write-Host "2 = Default trusted detector + approved YouTube URL"
Write-Host "3 = Trusted BTR detector + custom source"
$Choice = Read-Host "Choose mode"

switch ($Choice) {
    "1" {
        $Source = ""
        $Model = $env:UAV_MODEL_PATH
    }
    "2" {
        $Source = Read-Host "Paste the full YouTube URL"
        $Model = $env:UAV_MODEL_PATH
    }
    "3" {
        $Source = Read-Host "Paste the video/image/source path"
        $Model = $env:UAV_BTR_MODEL_PATH
        if ([string]::IsNullOrWhiteSpace($Model)) {
            throw "Set UAV_BTR_MODEL_PATH to a trusted local checkpoint."
        }
    }
    default { throw "Invalid mode selection." }
}

& $Launcher -Source $Source -Model $Model
