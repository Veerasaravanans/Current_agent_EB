@echo off
REM ===========================================================================
REM  Launch AI Agent in GUI Mode
REM  Windows Batch Script
REM ===========================================================================

echo ========================================================================
echo   Neural AI Agent - GUI Mode
echo   Enhanced with RAG + LangChain + LangGraph
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Launch GUI
echo Starting GUI...
python gui_controller.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: GUI failed to start
    echo.
    echo Common fixes:
    echo   1. Install PyQt6: pip install PyQt6
    echo   2. Initialize RAG: python init_rag_system.py
    echo   3. Check dependencies: pip install -r requirements.txt
    pause
)
