$VM_IP = "192.168.153.128"

cd "$env:USERPROFILE\Desktop\uav_ai_company"

Write-Host "======================================"
Write-Host " PHASE 1 - BTR CONFIDENCE FILTER"
Write-Host " Only detections >= 0.45 are accepted"
Write-Host "======================================"

python win_yolo_tcp_sender.py `
  --target $VM_IP `
  --source tank_real_test.mp4 `
  --model btr_best_v2.pt `
  --conf 0.45 `
  --imgsz 640 `
  --stride 1 `
  --send_width 960 `
  --show 1