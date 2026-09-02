$VM_IP = "192.168.153.128"

. (Join-Path $PSScriptRoot "portable_paths.ps1")

Write-Host ""
Write-Host "======================================"
Write-Host " UAV AI + ROS2 FULL YOLO SENDER"
Write-Host "======================================"
Write-Host "1 = COCO model + vehicles.mp4"
Write-Host "2 = COCO model + live stream URL"
Write-Host "3 = BTR trained model + btr_demo.mp4"
Write-Host "4 = BTR trained model + custom video/source"
Write-Host "5 = COCO model + webcam"
Write-Host "======================================"
Write-Host ""

$choice = Read-Host "Choose mode"

if ($choice -eq "1") {
    $MODEL = $BaseYoloModel
    $SOURCE = Join-Path $TestMediaDirectory "videos\vehicles.mp4"
    $CONF = "0.25"
}
elseif ($choice -eq "2") {
    $MODEL = $BaseYoloModel
    $SOURCE = Read-Host "Paste live stream URL"
    $CONF = "0.25"
}
elseif ($choice -eq "3") {
    $MODEL = $BtrModel
    $SOURCE = Resolve-UavPath "UAV_BTR_DEMO_VIDEO_PATH" "outputs\windows_ai\btr_demo.mp4"
    $CONF = "0.25"
}
elseif ($choice -eq "4") {
    $MODEL = $BtrModel
    $SOURCE = Read-Host "Paste BTR video/image/source path"
    $CONF = "0.25"
}
elseif ($choice -eq "5") {
    $MODEL = $BaseYoloModel
    $SOURCE = "0"
    $CONF = "0.25"
}
else {
    Write-Host "Invalid choice. Using COCO + vehicles.mp4"
    $MODEL = $BaseYoloModel
    $SOURCE = Join-Path $TestMediaDirectory "videos\vehicles.mp4"
    $CONF = "0.25"
}

Write-Host ""
Write-Host "Starting sender..."
Write-Host "Model: $MODEL"
Write-Host "Source: $SOURCE"
Write-Host "Target VM: $VM_IP"
Write-Host ""

& $PythonExecutable $SenderScript `
  --target $VM_IP `
  --source $SOURCE `
  --model $MODEL `
  --conf $CONF `
  --imgsz 640 `
  --stride 1 `
  --send_width 960 `
  --show 1
