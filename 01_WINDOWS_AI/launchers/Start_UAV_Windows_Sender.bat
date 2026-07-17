@echo off
cd /d "%USERPROFILE%\Desktop\uav_ai_company"

python win_yolo_tcp_sender_botsort_clean.py ^
  --target 192.168.153.128 ^
  --source 1minutesVIEWdroneVIDEOTANKS.mp4 ^
  --model military_kaggle_v1.pt ^
  --conf 0.25 ^
  --imgsz 960 ^
  --tracker botsort.yaml ^
  --send_width 960 ^
  --show 1

pause