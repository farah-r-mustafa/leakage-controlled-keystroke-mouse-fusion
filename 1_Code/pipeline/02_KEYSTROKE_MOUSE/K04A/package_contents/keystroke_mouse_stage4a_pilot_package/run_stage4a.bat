@echo off
setlocal
py -3.12 "%~dp0keystroke_mouse_stage4a_pilot.py" --input-stage3b "<local_path>" --input-stage3c "<local_path>" --output "<local_path>" --overwrite
pause
