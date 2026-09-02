$VM_IP = "192.168.153.128"

. (Join-Path $PSScriptRoot "portable_paths.ps1")

$Source = Join-Path $TestMediaDirectory "videos\vehicles.mp4"

& $PythonExecutable $SenderScript `
  --target $VM_IP `
  --source $Source `
  --model $BaseYoloModel `
  --conf 0.25 `
  --imgsz 640 `
  --stride 2 `
  --send_width 960 `
  --show 1
