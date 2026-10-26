@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: RUN_STAGE5_1_PILOT.bat "<local_path>" [output_folder]
  exit /b 1
)
set "INPUT=%~1"
if "%~2"=="" (
  set "OUTPUT=%~dp0keystroke_mouse_stage5_1_pilot_output"
) else (
  set "OUTPUT=%~2"
)
python keystroke_mouse_stage5_1_pilot.py --input "%INPUT%" --output "%OUTPUT%" --mode pilot --bootstrap-reps 100 --seed 5102026
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo Stage 5.1 Pilot FAILED with exit code %ERR%.
  exit /b %ERR%
)
echo Stage 5.1 Pilot completed successfully: "%OUTPUT%"
endlocal
