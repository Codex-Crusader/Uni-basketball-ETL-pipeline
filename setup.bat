@echo off
REM ============================================================
REM  NCAA Basketball Predictor — Windows Setup Script
REM  Run this once after downloading or cloning the project.
REM  Double-click it or run from PowerShell / Command Prompt.
REM ============================================================

echo.
echo  NCAA Basketball Predictor v2.5.1 — Setup
echo  ==========================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Download Python 3.8+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo  Python found:
python --version
echo.

REM Create virtual environment if it doesn't already exist
if not exist ".venv" (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists, skipping creation.
)

echo.
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo  Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo  ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Setup complete.
echo  ============================================================
echo.
echo  To use the predictor, activate the environment first:
echo.
echo      .venv\Scripts\activate
echo.
echo  Then run one of these commands:
echo.
echo      python main.py --fetch            Fetch ~2900 real NCAA games
echo      python main.py --generate-synthetic   Use offline synthetic data
echo      python main.py --train            Train all models
echo      python main.py --serve            Start dashboard at localhost:5000
echo.
echo  Quick start (no internet required):
echo.
echo      python main.py --generate-synthetic
echo      python main.py --train
echo      python main.py --serve
echo.
pause