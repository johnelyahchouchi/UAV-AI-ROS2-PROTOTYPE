$VM_IP = "192.168.153.128"

cd "$env:USERPROFILE\Desktop\uav_ai_company"

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
    $MODEL = "yolov8n.pt"
    $SOURCE = "vehicles.mp4"
    $CONF = "0.25"
}
elseif ($choice -eq "2") {
    $MODEL = "yolov8n.pt"
    $SOURCE = Read-Host "Paste live stream URL"
    $CONF = "0.25"
}
elseif ($choice -eq "3") {
    $MODEL = "btr_best.pt"
    $SOURCE = "btr_demo.mp4"
    $CONF = "0.25"
}
elseif ($choice -eq "4") {
    $MODEL = "btr_best.pt"
    $SOURCE = Read-Host "Paste BTR video/image/source path"
    $CONF = "0.25"
}
elseif ($choice -eq "5") {
    $MODEL = "yolov8n.pt"
    $SOURCE = "0"
    $CONF = "0.25"
}
else {
    Write-Host "Invalid choice. Using COCO + vehicles.mp4"
    $MODEL = "yolov8n.pt"
    $SOURCE = "vehicles.mp4"
    $CONF = "0.25"
}

Write-Host ""
Write-Host "Starting sender..."
Write-Host "Model: $MODEL"
Write-Host "Source: $SOURCE"
Write-Host "Target VM: $VM_IP"
Write-Host ""

python win_yolo_tcp_sender.py `
  --target $VM_IP `
  --source $SOURCE `
  --model $MODEL `
  --conf $CONF `
  --imgsz 640 `
  --stride 1 `
  --send_width 960 `
  --show 1