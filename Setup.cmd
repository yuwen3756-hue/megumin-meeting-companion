@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\install.py %*
) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" scripts\install.py %*
) else (
  py -3.12 scripts\install.py %*
)
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo Setup did not finish. Check the error above; you can safely run Setup.cmd again.
pause
exit /b 1
