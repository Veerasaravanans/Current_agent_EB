#!/usr/bin/env python3
"""
verify_setup.py - Verification Script for AI Agent Framework

This script verifies that the AI Agent framework is properly installed
and configured. It checks all dependencies, configurations, and services.
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.NC}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.NC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.NC}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.NC}")

def check_python_version():
    """Check Python version."""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_module(module_name, display_name=None):
    """Check if a Python module is installed."""
    if display_name is None:
        display_name = module_name
    
    try:
        importlib.import_module(module_name)
        print_success(f"{display_name}")
        return True
    except ImportError:
        print_error(f"{display_name} not installed")
        return False

def check_core_dependencies():
    """Check core AI and vision dependencies."""
    print_header("Checking Core Dependencies")
    
    modules = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ollama", "Ollama"),
    ]
    
    results = [check_module(module, display) for module, display in modules]
    return all(results)

def check_ocr_dependencies():
    """Check OCR dependencies."""
    print_header("Checking OCR Dependencies")
    
    modules = [
        ("easyocr", "EasyOCR"),
        ("paddleocr", "PaddleOCR"),
    ]
    
    results = [check_module(module, display) for module, display in modules]
    
    # Pytesseract is optional
    try:
        import pytesseract
        print_success("Pytesseract (optional)")
    except ImportError:
        print_warning("Pytesseract not installed (optional)")
    
    return any(results)  # At least one OCR should work

def check_langchain_dependencies():
    """Check LangChain ecosystem dependencies."""
    print_header("Checking LangChain Dependencies")
    
    modules = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_community", "LangChain Community"),
        ("langchain_ollama", "LangChain Ollama"),
        ("langgraph", "LangGraph"),
    ]
    
    results = [check_module(module, display) for module, display in modules]
    return all(results)

def check_rag_dependencies():
    """Check RAG dependencies."""
    print_header("Checking RAG Dependencies")
    
    modules = [
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
    ]
    
    results = [check_module(module, display) for module, display in modules]
    return all(results)

def check_gui_dependencies():
    """Check GUI dependencies."""
    print_header("Checking GUI Dependencies (Optional)")
    
    modules = [
        ("PyQt6", "PyQt6"),
        ("pyttsx3", "pyttsx3"),
    ]
    
    for module, display in modules:
        try:
            importlib.import_module(module)
            print_success(f"{display}")
        except ImportError:
            print_warning(f"{display} not installed (optional for GUI)")
    
    return True  # Not critical

def check_directories():
    """Check required directories."""
    print_header("Checking Directory Structure")
    
    directories = [
        "prompts",
        "prompts/component_specific",
        "vector_db",
        "reference_icons",
        "knowledge_base",
        "test_reports",
        "logs",
        "screenshots",
    ]
    
    results = []
    for directory in directories:
        path = Path(directory)
        if path.exists() and path.is_dir():
            print_success(f"{directory}/")
            results.append(True)
        else:
            print_error(f"{directory}/ not found")
            results.append(False)
    
    return all(results)

def check_prompt_files():
    """Check prompt files."""
    print_header("Checking Prompt Files")
    
    prompt_files = [
        "prompts/base_prompts.md",
        "prompts/error_handling.md",
        "prompts/learned_solutions.md",
        "prompts/custom_commands.md",
    ]
    
    results = []
    for prompt_file in prompt_files:
        path = Path(prompt_file)
        if path.exists() and path.is_file():
            size_kb = path.stat().st_size / 1024
            print_success(f"{prompt_file} ({size_kb:.1f} KB)")
            results.append(True)
        else:
            print_warning(f"{prompt_file} not found")
            results.append(False)
    
    return any(results)  # At least some prompts should exist

def check_ollama_service():
    """Check if Ollama service is running."""
    print_header("Checking Ollama Service")
    
    # Check if ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Ollama installed: {version}")
        else:
            print_error("Ollama not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("Ollama not found")
        return False
    
    # Check if ollama service is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print_success("Ollama service is running")
        else:
            print_error("Ollama service not responding")
            return False
    except Exception as e:
        print_warning("Ollama service not running. Start with: ollama serve")
        return False
    
    # Check if llava model is downloaded
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "llava:7b" in result.stdout or "llava" in result.stdout:
            print_success("llava:7b model downloaded")
            return True
        else:
            print_warning("llava:7b model not found. Download with: ollama pull llava:7b")
            return False
    except subprocess.TimeoutExpired:
        print_warning("Could not check Ollama models")
        return False

def check_vector_database():
    """Check if vector database is initialized."""
    print_header("Checking Vector Database")
    
    vector_db_path = Path("vector_db")
    
    if not vector_db_path.exists():
        print_error("vector_db/ directory not found")
        return False
    
    # Check for ChromaDB files
    chroma_file = vector_db_path / "chroma.sqlite3"
    if chroma_file.exists():
        size_mb = chroma_file.stat().st_size / (1024 * 1024)
        print_success(f"ChromaDB initialized ({size_mb:.2f} MB)")
        return True
    else:
        print_warning("ChromaDB not initialized. Run: python init_rag_system.py")
        return False

def check_adb():
    """Check ADB installation and devices."""
    print_header("Checking ADB (Android Debug Bridge)")
    
    try:
        result = subprocess.run(
            ["adb", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"ADB installed: {version}")
        else:
            print_warning("ADB not found (required for device testing)")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("ADB not found (required for device testing)")
        return False
    
    # Check for connected devices
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        devices = [line for line in result.stdout.split('\n') if '\tdevice' in line]
        if devices:
            print_success(f"{len(devices)} Android device(s) connected")
            for device in devices:
                print_info(f"  {device}")
            return True
        else:
            print_warning("No Android devices connected")
            return False
    except subprocess.TimeoutExpired:
        print_warning("Could not check for devices")
        return False

def check_gpu():
    """Check GPU availability."""
    print_header("Checking GPU Support")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU available: {gpu_name}")
            print_info(f"  CUDA version: {torch.version.cuda}")
            print_info(f"  cuDNN version: {torch.backends.cudnn.version()}")
            return True
        else:
            print_info("No GPU detected, using CPU (slower)")
            return True
    except Exception as e:
        print_warning(f"Could not check GPU: {e}")
        return True

def print_summary(checks):
    """Print summary of all checks."""
    print_header("VERIFICATION SUMMARY")
    
    total = len(checks)
    passed = sum(1 for check in checks.values() if check)
    failed = total - passed
    
    print(f"\nTotal checks: {total}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    
    print(f"\n{Colors.CYAN}{'─' * 80}{Colors.NC}")
    
    if all(checks.values()):
        print(f"\n{Colors.GREEN}🎉 All checks passed! Your AI Agent framework is ready to use.{Colors.NC}")
        print(f"\n{Colors.YELLOW}Next steps:{Colors.NC}")
        print(f"  1. Connect Android device: {Colors.CYAN}adb devices{Colors.NC}")
        print(f"  2. Run test: {Colors.CYAN}python3 gui_controller_voice.py{Colors.NC}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some checks failed. Please review the errors above.{Colors.NC}")
        if not checks.get("ollama", False):
            print(f"\n{Colors.CYAN}To start Ollama:{Colors.NC}")
            print(f"  ollama serve")
            print(f"  ollama pull llava:7b")
        if not checks.get("vector_db", False):
            print(f"\n{Colors.CYAN}To initialize RAG system:{Colors.NC}")
            print(f"  python3 init_rag_system.py")
    
    print(f"\n{Colors.CYAN}{'─' * 80}{Colors.NC}\n")

def main():
    """Main verification function."""
    print_header("AI AGENT FRAMEWORK - SETUP VERIFICATION")
    print(f"{Colors.CYAN}Enhanced with RAG, LangChain & LangGraph{Colors.NC}\n")
    
    # Run all checks
    checks = {
        "python": check_python_version(),
        "core": check_core_dependencies(),
        "ocr": check_ocr_dependencies(),
        "langchain": check_langchain_dependencies(),
        "rag": check_rag_dependencies(),
        "gui": check_gui_dependencies(),
        "directories": check_directories(),
        "prompts": check_prompt_files(),
        "ollama": check_ollama_service(),
        "vector_db": check_vector_database(),
        "adb": check_adb(),
        "gpu": check_gpu(),
    }
    
    # Print summary
    print_summary(checks)
    
    # Return exit code
    if all(checks[key] for key in ["python", "core", "langchain", "rag", "directories"]):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
