@echo off
echo Starting SD Prompt Assistant...
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo.
echo Starting FastAPI server...
python -m backend.main

pause
