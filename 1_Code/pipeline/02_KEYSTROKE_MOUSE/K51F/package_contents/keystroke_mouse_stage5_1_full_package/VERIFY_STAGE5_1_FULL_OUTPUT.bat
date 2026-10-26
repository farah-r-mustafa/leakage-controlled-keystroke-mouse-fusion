@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: VERIFY_STAGE5_1_FULL_OUTPUT.bat "<local_path>"
  exit /b 1
)
python VERIFY_STAGE5_1_FULL_OUTPUT.py "%~1"
exit /b %ERRORLEVEL%
