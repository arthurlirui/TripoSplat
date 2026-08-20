@echo off
REM TripoSplat environment setup for Windows CMD/PowerShell.
REM Run this before executing any Python script:
REM   setup_env.bat
REM Or in PowerShell:
REM   cmd /c setup_env.bat

REM Add conda env Scripts to PATH (for ninja, needed by gsplat JIT)
set PATH=C:\Users\pcl\.conda\envs\rtg132\Scripts;C:\Users\pcl\.conda\envs\rtg132\Library\bin;%PATH%

REM Force UTF-8 (fixes MSVC encoding error during gsplat compilation)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Set Python executable
set PYTHON=C:\Users\pcl\.conda\envs\rtg132\python.exe

echo TripoSplat environment configured:
echo   Python: %PYTHON%
echo   PYTHONUTF8=%PYTHONUTF8%
echo   PATH includes rtg132 Scripts
echo.
echo Usage: %PYTHON% your_script.py args...
