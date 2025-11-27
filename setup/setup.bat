@echo off
REM =============================================================================
REM AI Agent Framework - Windows Setup Script
REM =============================================================================
REM Author: Veera Saravanan
REM Framework: Neural AI Agent v2.0 (RAG-Enhanced)
REM Description: Automated installation script for Windows
REM =============================================================================

echo ============================================================================
echo AI AGENT FRAMEWORK - WINDOWS SETUP
echo Enhanced with RAG, LangChain ^& LangGraph
echo ============================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detected
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists
    set /p RECREATE="Remove and recreate? (y/N): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q venv
        python -m venv venv
    )
) else (
    python -m venv venv
)

echo [OK] Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] No NVIDIA GPU detected
    echo Installing PyTorch (CPU version)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo [OK] PyTorch installed
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo [OK] All dependencies installed
echo.

REM Create directories
echo Creating directory structure...
if not exist vector_db mkdir vector_db
if not exist reference_icons mkdir reference_icons
if not exist knowledge_base mkdir knowledge_base
if not exist test_reports mkdir test_reports
if not exist logs mkdir logs
if not exist screenshots mkdir screenshots
if not exist prompts mkdir prompts
if not exist prompts\component_specific mkdir prompts\component_specific

echo [OK] Directories created
echo.

REM Create .env file
if not exist .env (
    echo Creating .env configuration file...
    (
        echo # AI Agent Framework Configuration
        echo OLLAMA_BASE_URL=http://localhost:11434
        echo OLLAMA_MODEL=llava:7b
        echo RAG_ENABLED=true
        echo VECTOR_DB_PATH=./vector_db
        echo LANGCHAIN_ENABLED=true
        echo LANGGRAPH_ENABLED=true
        echo VOICE_ENABLED=false
    ) > .env
    echo [OK] .env file created
) else (
    echo [INFO] .env file already exists
)
echo.

REM Check Ollama
echo Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama not found!
    echo Please install Ollama from: https://ollama.ai
    echo After installation, run:
    echo   ollama serve
    echo   ollama pull llava:7b
) else (
    echo [OK] Ollama installed
    ollama --version
    
    REM Check if llava model is downloaded
    ollama list | findstr "llava:7b" >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo Downloading llava:7b model (this may take several minutes)...
        ollama pull llava:7b
    ) else (
        echo [OK] llava:7b model already downloaded
    )
)
echo.

REM Initialize RAG system
if exist init_rag_system.py (
    echo Initializing RAG system...
    python init_rag_system.py
    echo [OK] RAG system initialized
) else (
    echo [WARNING] init_rag_system.py not found
    echo You can run it manually later: python init_rag_system.py
)
echo.

REM Check ADB
echo Checking ADB installation...
adb version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ADB not found
    echo Please install Android Platform Tools for device testing
    echo Download from: https://developer.android.com/studio/releases/platform-tools
) else (
    echo [OK] ADB installed
    adb version
    
    echo.
    echo Checking for connected devices...
    adb devices
)
echo.

REM Run verification
echo Running verification tests...
python -c "import torch; import PIL; import cv2; import ollama; import langchain; import chromadb; print('[OK] All imports successful')"
echo.

REM Print summary
echo ============================================================================
echo INSTALLATION COMPLETE!
echo ============================================================================
echo.
echo Next Steps:
echo.
echo 1. Ensure Ollama is running:
echo    ollama serve
echo.
echo 2. Connect Android device:
echo    adb devices
echo.
echo 3. Run your first test:
echo    python gui_controller_voice.py
echo    # or
echo    python prompt_driven_agent.py --test-id "NAID-24430"
echo.
echo Documentation:
echo   README.md - Full documentation
echo   QUICK_START.md - Quick start guide
echo   SETUP_INSTRUCTIONS.txt - Detailed setup
echo.
echo ============================================================================
echo Your AI Agent framework is ready to use!
echo ============================================================================
echo.

pause
