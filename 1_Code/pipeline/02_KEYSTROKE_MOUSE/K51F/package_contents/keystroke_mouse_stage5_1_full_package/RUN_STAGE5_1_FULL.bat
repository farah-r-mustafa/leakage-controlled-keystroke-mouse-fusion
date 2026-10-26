@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: RUN_STAGE5_1_FULL.bat "<local_path>" "<local_path>" [output_folder]
  exit /b 1
)
if "%~2"=="" (
  echo ERROR: Successful Stage 5.1 Pilot output ZIP/folder is required.
  exit /b 1
)
set "INPUT=%~1"
set "PILOT=%~2"
if "%~3"=="" (
  set "OUTPUT=%~dp0keystroke_mouse_stage5_1_full_output"
) else (
  set "OUTPUT=%~3"
)
python keystroke_mouse_stage5_1_full.py --input "%INPUT%" --pilot-output "%PILOT%" --output "%OUTPUT%" --bootstrap-reps 5000 --seed 5102026
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo Stage 5.1 Full FAILED with exit code %ERR%.
  exit /b %ERR%
)
echo Stage 5.1 Full completed successfully: "%OUTPUT%"
endlocal
