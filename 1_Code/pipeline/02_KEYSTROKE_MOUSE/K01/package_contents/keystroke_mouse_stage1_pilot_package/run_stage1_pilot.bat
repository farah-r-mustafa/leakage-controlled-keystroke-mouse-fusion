@echo off
setlocal
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" set "INPUT_FILE=%SCRIPT_DIR%Keystroke and Mouse Dynamics for UEBA Dataset.zip"
set "OUTPUT_DIR=%SCRIPT_DIR%keystroke_mouse_stage1_review"

if not exist "%INPUT_FILE%" (
  echo [ERROR] Dataset ZIP not found:
  echo %INPUT_FILE%
  echo.
  echo Usage:
  echo   run_stage1_pilot.bat "<local_path>"
  exit /b 2
)

python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not available in PATH.
  exit /b 3
)

python "%SCRIPT_DIR%keystroke_mouse_stage1_pilot.py" --input "%INPUT_FILE%" --output "%OUTPUT_DIR%" --sample-rows 5
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [ERROR] Stage 1 failed with exit code %RC%.
  exit /b %RC%
)

echo.
echo Stage 1 completed successfully.
echo Review ZIP:
echo %OUTPUT_DIR%.zip
pause
endlocal
