$VM_IP = "192.168.153.128"

cd "$env:USERPROFILE\Desktop\uav_ai_company"

Write-Host ""
Write-Host "======================================"
Write-Host " UAV AI + ROS2 YOLO Sender"
Write-Host "======================================"
Write-Host "1 = Test video vehicles.mp4"
Write-Host "2 = Live stream URL"
Write-Host "3 = Local webcam"
Write-Host "======================================"
Write-Host ""

$choice = Read-Host "Choose mode"

if ($choice -eq "1") {
    $SOURCE = "vehicles.mp4"
}
elseif ($choice -eq "2") {
    $SOURCE = Read-Host "Paste live stream URL"
}
elseif ($choice -eq "3") {
    $SOURCE = "0"
}
else {
    Write-Host "Invalid choice. Using vehicles.mp4"
    $SOURCE = "vehicles.mp4"
}

python win_yolo_tcp_sender.py `
  --target $VM_IP `
  --source $SOURCE `
  --model yolov8n.pt `
  --conf 0.25 `
  --imgsz 640 `
  --stride 2 `
  --send_width 960 `
  --show 1