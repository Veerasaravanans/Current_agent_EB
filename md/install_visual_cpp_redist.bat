@echo off
REM install_visual_cpp_redist.bat - Download and install Visual C++ Redistributable
REM This is required for PyTorch DLL loading on Windows

echo ========================================================================
echo    Microsoft Visual C++ Redistributable Installer
echo ========================================================================
echo.
echo This will download and install the required Visual C++ libraries
echo for PyTorch to work properly on Windows.
echo.
echo File size: ~25 MB
echo.

pause

echo.
echo Downloading Visual C++ Redistributable...
echo.

REM Download using PowerShell
powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe'}"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Download failed!
    echo.
    echo Please download manually from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Download complete!
echo.
echo Installing Visual C++ Redistributable...
echo Please follow the installer prompts...
echo.

start /wait %TEMP%\vc_redist.x64.exe /install /passive /norestart

if %errorlevel% equ 0 (
    echo.
    echo ========================================================================
    echo ✅ Visual C++ Redistributable installed successfully!
    echo ========================================================================
    echo.
    echo Next steps:
    echo 1. Run: python fix_pytorch_dll_error.py
    echo 2. Then try: python gui_controller_voice.py
    echo.
) else (
    echo.
    echo ⚠️  Installation completed with warnings or was cancelled
    echo.
)

REM Clean up
del /f /q "%TEMP%\vc_redist.x64.exe" 2>nul

pause
