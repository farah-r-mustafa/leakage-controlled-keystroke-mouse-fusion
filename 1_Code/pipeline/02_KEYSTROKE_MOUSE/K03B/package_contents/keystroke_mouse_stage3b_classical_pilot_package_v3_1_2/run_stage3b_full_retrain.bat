@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "STAGE2C=<local_path>"
set "STAGE3A=<local_path>"
set "OUTPUT=<local_path>"

echo ============================================================
echo Stage 3B v3.1.2 - Full Retraining
echo Existing Stage 3B output will be overwritten.
echo Close any previous CSV/ZIP files before continuing.
echo ============================================================

py -3.12 "%SCRIPT_DIR%keystroke_mouse_stage3b_classical_pilot.py" --input-stage2c "%STAGE2C%" --input-stage3a "%STAGE3A%" --output "%OUTPUT%" --overwrite
set EXITCODE=%ERRORLEVEL%
echo.
if "%EXITCODE%"=="0" (
  echo Stage 3B completed successfully.
  echo Review: %OUTPUT%.zip
) else (
  echo Stage 3B failed with exit code %EXITCODE%.
)
echo.
pause
exit /b %EXITCODE%
