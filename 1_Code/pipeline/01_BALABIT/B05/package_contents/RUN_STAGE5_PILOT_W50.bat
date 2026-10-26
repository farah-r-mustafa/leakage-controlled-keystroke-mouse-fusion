@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Balabit Stage 5 Pilot W50
pushd "%~dp0"

echo ============================================================
echo   Balabit Stage 5 PILOT - CNN / BiLSTM / CNN+BiLSTM
echo   Window size: 50
echo ============================================================
echo.
echo Select the FULL Stage 4 folder containing users.csv and tensors\w50
echo.

set /p "STAGE4_ROOT=Paste the Balabit Stage 4 FULL results folder path: "
if "%STAGE4_ROOT%"=="" goto :badinput
if not exist "%STAGE4_ROOT%\users.csv" (
  echo ERROR: users.csv was not found.
  pause
  popd
  exit /b 2
)
if not exist "%STAGE4_ROOT%\tensors\w50\train_X.npy" (
  echo ERROR: tensors\w50\train_X.npy was not found.
  pause
  popd
  exit /b 2
)

set "OUTPUT_DIR=%~dp0balabit_stage5_pilot_w50"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py"
) else (
  set "PY_CMD=python"
)

echo.
echo Checking/installing required packages...
"%PY_CMD%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo ERROR: Package installation failed.
  pause
  popd
  exit /b 3
)

echo.
echo Training three pilot models. This may take some time on CPU...
"%PY_CMD%" "%~dp001_balabit_stage5_deep_models.py" ^
  --stage4-root "%STAGE4_ROOT%" ^
  --output "%OUTPUT_DIR%" ^
  --window-size 50 ^
  --models cnn,bilstm,hybrid ^
  --max-train-per-user 2500 ^
  --max-validation-per-user 800 ^
  --max-test-per-user 1000 ^
  --epochs 15 ^
  --patience 4 ^
  --batch-size 128 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --seed 42 ^
  --device auto ^
  --num-workers 0

set "EXIT_CODE=%errorlevel%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Stage 5 Pilot completed successfully.
  echo Upload this review file:
  echo "%~dp0balabit_stage5_pilot_w50_review.zip"
  echo.
  echo Keep the complete local folder:
  echo "%OUTPUT_DIR%"
) else (
  echo Stage 5 Pilot stopped with exit code %EXIT_CODE%.
  echo Upload run_log.txt and stage5_errors.csv from:
  echo "%OUTPUT_DIR%"
)
pause
popd
exit /b %EXIT_CODE%

:badinput
echo ERROR: No path was entered.
pause
popd
exit /b 2
