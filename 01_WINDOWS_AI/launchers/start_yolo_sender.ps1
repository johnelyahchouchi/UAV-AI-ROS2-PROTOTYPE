$VM_IP = "192.168.153.128"

cd "$env:USERPROFILE\Desktop\uav_ai_company"

python win_yolo_tcp_sender.py `
  --target $VM_IP `
  --source vehicles.mp4 `
  --model yolov8n.pt `
  --conf 0.25 `
  --imgsz 640 `
  --stride 2 `
  --send_width 960 `
  --show 1