@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
title Balabit Stage 2 - PILOT
set "SCRIPT=%~dp001_balabit_preprocess_windows.py"

echo ============================================================
echo   Balabit Stage 2 - PILOT

echo ============================================================
echo Processes up to 5 training sessions per user and windows 30,50.
echo Source data remains unchanged.
echo.
set /p "DATA_ROOT=Paste the full Balabit RAW dataset folder path: "
if "%DATA_ROOT%"=="" goto :no_path
if not exist "%DATA_ROOT%" goto :bad_path

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :pip_fail

"%PY%" "%SCRIPT%" --data-root "%DATA_ROOT%" --output "%~dp0balabit_stage2_pilot_results" --window-sizes 30,50 --overlap 0.50 --max-sessions-per-user 5
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
 echo PILOT completed. Upload:
 echo "%~dp0balabit_stage2_pilot_results.zip"
) else (
 echo PILOT ended with exit code %RC%.
 echo Upload run_log.txt and session_errors.csv.
)
pause
popd
exit /b %RC%
:no_path
echo ERROR: No path entered.
pause
popd
exit /b 2
:bad_path
echo ERROR: Dataset folder does not exist.
pause
popd
exit /b 2
:pip_fail
echo ERROR: Could not install requirements.
echo Try: py -m pip install pandas numpy
pause
popd
exit /b 4
