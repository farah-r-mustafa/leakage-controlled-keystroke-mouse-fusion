@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%keystroke_mouse_stage3a_pilot.py" --input-stage2c "<local_path>" --output "<local_path>"
echo.
echo Exit code: %ERRORLEVEL%
pause
