@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: RUN_STAGE5_1_FULL.bat "<local_path>" [output_folder]
  exit /b 1
)
set "INPUT=%~1"
if "%~2"=="" (
  set "OUTPUT=%~dp0keystroke_mouse_stage5_1_full_output"
) else (
  set "OUTPUT=%~2"
)
python keystroke_mouse_stage5_1_pilot.py --input "%INPUT%" --output "%OUTPUT%" --mode full --bootstrap-reps 2000 --seed 5102026
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo Stage 5.1 Full FAILED with exit code %ERR%.
  exit /b %ERR%
)
echo Stage 5.1 Full completed successfully: "%OUTPUT%"
endlocal
