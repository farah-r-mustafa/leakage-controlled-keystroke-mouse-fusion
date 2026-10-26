@echo off
setlocal
cd /d "%~dp0"

py -3.12 keystroke_mouse_stage4b_pilot.py ^
  --input-stage4a "<local_path>" ^
  --input-stage3b "<local_path>" ^
  --input-stage3c "<local_path>" ^
  --input-stage2a "<local_path>" ^
  --output "<local_path>" ^
  --overwrite

set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
