@echo off
setlocal
cd /d "%~dp0"

set "DATASET=<local_path>"
set "OUTPUT=<local_path>"

python keystroke_mouse_stage2b_pilot.py --input "%DATASET%" --output "%OUTPUT%"
if errorlevel 1 (
  echo.
  echo Stage 2B ended with an error. Review warnings_and_failures.csv and execution_log.txt.
  pause
  exit /b 1
)

echo.
echo Stage 2B completed. Review ZIP:
echo %OUTPUT%.zip
pause
