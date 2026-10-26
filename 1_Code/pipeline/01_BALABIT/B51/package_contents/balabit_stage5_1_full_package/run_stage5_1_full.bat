@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Balabit Stage 5.1 FULL - Unified Statistical Validation
echo User bootstrap: 10000 ^| Session-block bootstrap: 2000
echo No training, no inference, no threshold refitting.
echo ============================================================
python --version
if errorlevel 1 (
  echo ERROR: Python was not found in PATH.
  pause
  exit /b 1
)
python "%~dp001_run_stage5_1_full.py" --user-replicates 10000 --session-replicates 2000 --seed 2026
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo FULL execution stopped safely. Review run_console.log and 00_full_manifest.json.
  pause
  exit /b %RC%
)
python "%~dp002_verify_stage5_1_full.py"
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (
  echo Stage 5.1 Full and verification completed successfully.
) else (
  echo Execution finished, but verification requires review.
)
pause
exit /b %RC%
