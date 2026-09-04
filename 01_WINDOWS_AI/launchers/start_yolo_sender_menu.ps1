Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Launcher = Join-Path $PSScriptRoot "start_yolo_sender.ps1"
Write-Host "1 = Repository test video"
Write-Host "2 = Approved YouTube URL"
Write-Host "3 = Local webcam"
$Choice = Read-Host "Choose mode"

switch ($Choice) {
    "1" { $Source = "" }
    "2" { $Source = Read-Host "Paste the full YouTube URL" }
    "3" { $Source = "0" }
    default { throw "Invalid mode selection." }
}

& $Launcher -Source $Source
