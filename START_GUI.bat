@echo off
setlocal
cd /d "%~dp0"
set HF_HOME=%CD%\data\models\huggingface
set PYTHONPATH=%CD%\app
set PYTHONUNBUFFERED=1
set DEPTH_FUSION_CONSOLE_EVENTS=1
set DEPTH_FUSION_TEE_STDIO=1
if not defined DEPTH_FUSION_VERBOSE_THIRD_PARTY set DEPTH_FUSION_VERBOSE_THIRD_PARTY=0

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run setup_env.bat first.
  pause
  exit /b 1
)

if not exist "data\logs" mkdir "data\logs"
echo [INFO] Event log: %CD%\data\logs\events.log
echo [INFO] Launching GUI...
".venv\Scripts\python.exe" -u app\main_gui.py
set EXIT_CODE=%ERRORLEVEL%
echo [INFO] GUI exited with code %EXIT_CODE%.
echo [INFO] Event log saved at: %CD%\data\logs\events.log
pause
exit /b %EXIT_CODE%
