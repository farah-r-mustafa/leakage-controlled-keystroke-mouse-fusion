@echo off
setlocal
cd /d "%~dp0"
py -3.12 keystroke_mouse_stage3c_deep_pilot.py ^
  --input-stage2c "<local_path>" ^
  --input-stage3a "<local_path>" ^
  --output "<local_path>" ^
  --overwrite
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
