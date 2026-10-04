@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run Setup.cmd once first.
  pause
  exit /b 1
)
start "MeguminMeet" ".venv\Scripts\pythonw.exe" "main.py" %*
