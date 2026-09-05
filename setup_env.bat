@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_CMD=py -3.11"
%PYTHON_CMD% --version >nul 2>nul
if errorlevel 1 set "PYTHON_CMD=python"

%PYTHON_CMD% --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10+ was not found.
  exit /b 1
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10+ is required.
  %PYTHON_CMD% --version
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating .venv...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 exit /b 1
)

set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

if defined TORCH_INDEX_URL (
  echo [INFO] Installing PyTorch from TORCH_INDEX_URL=%TORCH_INDEX_URL%
  "%PY%" -m pip install torch torchvision --index-url "%TORCH_INDEX_URL%"
) else (
  echo [INFO] Installing standard PyTorch wheels.
  echo [INFO] Set TORCH_INDEX_URL when a specific CUDA wheel source is required.
  "%PY%" -m pip install torch torchvision
)
if errorlevel 1 exit /b 1

"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

"%PY%" -m pip check
if errorlevel 1 exit /b 1

"%PY%" scripts\verify_environment.py
if errorlevel 1 exit /b 1

if not exist "data\models" mkdir "data\models"
if not exist "data\logs" mkdir "data\logs"
if not exist "data\cache" mkdir "data\cache"

echo [OK] Base environment is ready.
echo [INFO] External 4DHumans / WHAM / Human Parsing model assets still need their project-specific setup.
exit /b 0
