@echo off
setlocal EnableExtensions
chcp 65001 >nul
pushd "%~dp0"
title Balabit Stage 3 Full

echo ============================================================
echo   Balabit Stage 3 FULL - Claims and Baselines
echo ============================================================
echo.
echo Select the Stage 2 RESULTS folder containing window_features.
echo Run the PILOT first and review it before this full run.
echo.
set /p "STAGE2_DIR=Paste the full Stage 2 results folder path: "
if "%STAGE2_DIR%"=="" goto :badpath
if not exist "%STAGE2_DIR%\window_features" goto :badpath

set "OUTPUT_DIR=%~dp0balabit_stage3_full_results"
call :python
if errorlevel 1 goto :end

"%PY_CMD%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :installfail

"%PY_CMD%" "%~dp003_balabit_claims_baselines.py" ^
  --stage2-dir "%STAGE2_DIR%" ^
  --output "%OUTPUT_DIR%" ^
  --window-sizes 30,50,70,90 ^
  --models centroid,one_class_svm,linear_svm,random_forest ^
  --negative-ratio 1.0 ^
  --max-train-positive 5000 ^
  --max-train-negative 5000 ^
  --ocsvm-max-positive 3000 ^
  --rf-trees 150 ^
  --rf-max-depth 20 ^
  --save-predictions
set "EXIT_CODE=%errorlevel%"
goto :report

:python
where py >nul 2>nul
if %errorlevel%==0 (set "PY_CMD=py" & exit /b 0)
where python >nul 2>nul
if %errorlevel%==0 (set "PY_CMD=python" & exit /b 0)
echo ERROR: Python was not found in PATH.
exit /b 1

:badpath
echo ERROR: The selected folder is invalid or has no window_features folder.
set "EXIT_CODE=2"
goto :report

:installfail
echo ERROR: Could not install Python requirements.
set "EXIT_CODE=4"
goto :report

:report
echo.
if "%EXIT_CODE%"=="0" (
  echo Full execution completed successfully.
  echo Upload: "%OUTPUT_DIR%.zip"
) else (
  echo Full execution ended with code %EXIT_CODE%.
  echo Upload run_log.txt and stage3_errors.csv from the output folder.
)
goto :end

:end
echo.
pause
popd
exit /b %EXIT_CODE%
