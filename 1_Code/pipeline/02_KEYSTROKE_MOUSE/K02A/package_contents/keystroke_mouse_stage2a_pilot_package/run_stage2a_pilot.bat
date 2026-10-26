@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage:
  echo   run_stage2a_pilot.bat "<local_path>" "<local_path>"
  echo.
  echo You may also run the Python command shown in README_AR.md.
  exit /b 2
)

set "INPUT_ZIP=%~1"
set "OUTPUT_DIR=%~2"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=%~dp0keystroke_mouse_stage2a_review"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 "%~dp0keystroke_mouse_stage2a_pilot.py" --input "%INPUT_ZIP%" --output "%OUTPUT_DIR%"
) else (
  python "%~dp0keystroke_mouse_stage2a_pilot.py" --input "%INPUT_ZIP%" --output "%OUTPUT_DIR%"
)

set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo Stage 2A failed with exit code %RC%.
  exit /b %RC%
)

echo.
echo Stage 2A completed successfully.
echo Review ZIP: %OUTPUT_DIR%.zip
exit /b 0
