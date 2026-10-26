@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Balabit Stage 4 Full
pushd "%~dp0"

echo ============================================================
echo   Balabit Stage 4 FULL - Sequential Tensor Builder
echo ============================================================
echo.
echo This may create more than 1 GB of local tensor files.
echo Ensure sufficient free disk space.
echo.

set /p "DATA_ROOT=Paste the RAW Balabit dataset folder path: "
if "%DATA_ROOT%"=="" goto :badinput
if not exist "%DATA_ROOT%" (
  echo ERROR: Raw dataset folder not found.
  pause
  popd
  exit /b 2
)

set /p "STAGE2_ROOT=Paste the Stage 2 FULL results folder path: "
if "%STAGE2_ROOT%"=="" goto :badinput
if not exist "%STAGE2_ROOT%\protocol_manifest.csv" (
  echo ERROR: protocol_manifest.csv was not found in that Stage 2 folder.
  pause
  popd
  exit /b 2
)

set "OUTPUT_DIR=%~dp0balabit_stage4_full"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py"
) else (
  set "PY_CMD=python"
)

echo.
echo Installing/checking requirements...
"%PY_CMD%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo ERROR: Could not install pandas/numpy.
  pause
  popd
  exit /b 3
)

echo.
echo Building full tensors...
"%PY_CMD%" "%~dp001_balabit_build_sequential_tensors.py" ^
  --data-root "%DATA_ROOT%" ^
  --stage2-root "%STAGE2_ROOT%" ^
  --output "%OUTPUT_DIR%" ^
  --window-sizes 30,50,70,90

set "EXIT_CODE=%errorlevel%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Full execution completed successfully.
  echo Upload this small review file:
  echo "%~dp0balabit_stage4_full_review.zip"
  echo.
  echo IMPORTANT: Keep this full local folder for Stage 5:
  echo "%OUTPUT_DIR%"
) else (
  echo Full execution failed with exit code %EXIT_CODE%.
  echo Upload run_log.txt and stage4_errors.csv.
)
pause
popd
exit /b %EXIT_CODE%

:badinput
echo ERROR: A required path was not entered.
pause
popd
exit /b 2
