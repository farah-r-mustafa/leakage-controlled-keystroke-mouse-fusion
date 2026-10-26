@echo off
setlocal
python "%~dp0keystroke_mouse_stage2c_pilot.py" --input-stage2b "<local_path>" --output "<local_path>"
echo.
echo Exit code: %ERRORLEVEL%
pause
