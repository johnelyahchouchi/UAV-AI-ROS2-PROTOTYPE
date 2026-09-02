@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Clean_Baseline.ps1" -Target 192.168.153.128

pause
