param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$CleanProject = "",
    [switch]$ArchiveCandidates
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CleanProject)) {
    $CleanProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
}

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source folder not found: $Source"
}

if (-not (Test-Path -LiteralPath $CleanProject)) {
    throw "Clean project not found: $CleanProject"
}

$reportFolder = Join-Path $CleanProject "00_PROJECT_GUIDE"
New-Item -ItemType Directory -Path $reportFolder -Force | Out-Null

$activeSenderName = $null
$selectionFile = Join-Path $reportFolder "ACTIVE_SENDER_SELECTED.txt"

if (Test-Path -LiteralPath $selectionFile) {
    $selectedPathLine = Get-Content -LiteralPath $selectionFile |
        Where-Object { $_ -match '^[A-Za-z]:\\' } |
        Select-Object -First 1

    if ($selectedPathLine) {
        $activeSenderName = Split-Path -Leaf $selectedPathLine
    }
}

$protectedNames = @(
    $activeSenderName,
    "UAV_Mission_Control_Center.py",
    "uav_ai_control_panel.py",
    "uav_windows_tcp_frame_bridge.py",
    "uav_clean_target_dashboard_v5.py",
    "uav_analytics_dashboard_v2.py",
    "uav_tank_type_timeline_dashboard_v1.py"
) | Where-Object { $_ }

$allowedExtensions = @(".py", ".ps1", ".bat", ".md", ".txt", ".json", ".yaml", ".yml")

$candidates = Get-ChildItem -LiteralPath $Source -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $allowedExtensions -contains $_.Extension.ToLowerInvariant()
    } |
    Where-Object {
        $_.Name -notin $protectedNames
    } |
    Where-Object {
        $_.Name -match '(?i)(backup|old|before|obsolete|deprecated|temporary|temp|copy)' -or
        $_.Name -match '^win_yolo_tcp_sender_v[123]\.py$' -or
        $_.Name -match '^uav_clean_target_dashboard_v[1-4]\.py$' -or
        $_.Name -eq 'uav_analytics_dashboard_v1.py'
    } |
    Sort-Object FullName -Unique

$csvPath = Join-Path $reportFolder "OLD_CODE_CANDIDATES.csv"

$candidates |
    Select-Object FullName, Name, Extension, Length, LastWriteTime |
    Export-Csv -NoTypeInformation -Encoding UTF8 $csvPath

Write-Host ""
Write-Host "Old-code candidate report created:" -ForegroundColor Cyan
Write-Host $csvPath
Write-Host ""
Write-Host "Protected active files:" -ForegroundColor Yellow
$protectedNames | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "VMware files, datasets, model weights, videos, and training runs are not included." -ForegroundColor Green

if (-not $ArchiveCandidates) {
    Write-Host ""
    Write-Host "No files were moved." -ForegroundColor Green
    Write-Host "Open OLD_CODE_CANDIDATES.csv and review every row." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After review, rerun with:" -ForegroundColor Cyan
    Write-Host ".\02_Review_And_Archive_Old_UAV_Files.ps1 -ArchiveCandidates"
    exit
}

$confirmation = Read-Host "Type ARCHIVE exactly to move every candidate into a dated archive"
if ($confirmation -cne "ARCHIVE") {
    Write-Host "Cancelled. Nothing was moved." -ForegroundColor Yellow
    exit
}

$archiveRoot = Join-Path $Source ("_ARCHIVE_OLD_CODE_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null

foreach ($file in $candidates) {
    $relativePath = $file.FullName.Substring($Source.Length).TrimStart("\")
    $destinationPath = Join-Path $archiveRoot $relativePath
    $destinationDirectory = Split-Path -Parent $destinationPath

    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Move-Item -LiteralPath $file.FullName -Destination $destinationPath -Force
}

Write-Host ""
Write-Host "Candidates moved to:" -ForegroundColor Green
Write-Host $archiveRoot
Write-Host ""
Write-Host "Do not permanently delete this archive until the clean system has been tested." -ForegroundColor Yellow
