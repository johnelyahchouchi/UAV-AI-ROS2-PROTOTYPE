$VM_IP = "192.168.153.128"

. (Join-Path $PSScriptRoot "portable_paths.ps1")

$Source = Resolve-UavPath "UAV_TANK_VIDEO_PATH" "06_TEST_MEDIA\videos\tank_real_test.mp4"

Write-Host "======================================"
Write-Host " PHASE 1 - BTR CONFIDENCE FILTER"
Write-Host " Only detections >= 0.45 are accepted"
Write-Host "======================================"

& $PythonExecutable $SenderScript `
  --target $VM_IP `
  --source $Source `
  --model $BtrModel `
  --conf 0.45 `
  --imgsz 640 `
  --stride 1 `
  --send_width 960 `
  --show 1
