#!/bin/bash

# =============================================================================
# AI Agent Framework - Complete Automated Setup Script
# =============================================================================
# Author: Veera Saravanan
# Framework: Neural AI Agent v2.0 (RAG-Enhanced)
# Description: Automated installation and configuration script
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions for colored output
print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================

PYTHON_CMD=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
USE_VENV=true
GPU_SUPPORT=false
SKIP_OLLAMA=false

# =============================================================================
# DETECT PYTHON
# =============================================================================

detect_python() {
    print_step "Detecting Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    else
        print_error "Python not found! Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION detected ($PYTHON_CMD)"
}

# =============================================================================
# CHECK SYSTEM REQUIREMENTS
# =============================================================================

check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="Windows"
    else
        OS="Unknown"
    fi
    print_success "Operating System: $OS"
    
    # Check disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (10GB+ recommended)"
    else
        print_success "Disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Check RAM
    if [[ "$OS" == "Linux" ]]; then
        TOTAL_RAM=$(free -g | awk 'NR==2 {print $2}')
        if [ "$TOTAL_RAM" -lt 8 ]; then
            print_warning "Low RAM: ${TOTAL_RAM}GB detected (8GB+ recommended)"
        else
            print_success "RAM: ${TOTAL_RAM}GB"
        fi
    fi
    
    # Check for GPU (NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        print_success "GPU detected: $GPU_INFO"
        GPU_SUPPORT=true
    else
        print_warning "No NVIDIA GPU detected. Will use CPU (slower)"
    fi
}

# =============================================================================
# CREATE VIRTUAL ENVIRONMENT
# =============================================================================

create_venv() {
    if [ "$USE_VENV" = true ]; then
        print_step "Creating virtual environment..."
        
        if [ -d "$VENV_DIR" ]; then
            print_warning "Virtual environment already exists at $VENV_DIR"
            read -p "Remove and recreate? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_DIR"
            else
                print_step "Using existing virtual environment"
                return
            fi
        fi
        
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created at $VENV_DIR"
        
        # Activate virtual environment
        source "$VENV_DIR/bin/activate"
        print_success "Virtual environment activated"
        
        # Update pip
        pip install --upgrade pip setuptools wheel
    else
        print_warning "Skipping virtual environment creation"
    fi
}

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

install_dependencies() {
    print_step "Installing Python dependencies..."
    
    # Install PyTorch first
    if [ "$GPU_SUPPORT" = true ]; then
        print_step "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_step "Installing PyTorch (CPU version)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    print_success "PyTorch installed"
    
    # Install rest of requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_step "Installing requirements from requirements.txt..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
        print_success "All dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# =============================================================================
# INSTALL OLLAMA
# =============================================================================

install_ollama() {
    if [ "$SKIP_OLLAMA" = true ]; then
        print_warning "Skipping Ollama installation"
        return
    fi
    
    print_step "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>&1 || echo "unknown")
        print_success "Ollama already installed: $OLLAMA_VERSION"
    else
        print_step "Installing Ollama..."
        
        if [[ "$OS" == "Linux" ]] || [[ "$OS" == "macOS" ]]; then
            curl -fsSL https://ollama.ai/install.sh | sh
            print_success "Ollama installed"
        else
            print_warning "Please install Ollama manually from: https://ollama.ai"
            read -p "Press Enter when Ollama is installed..."
        fi
    fi
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama is running"
    else
        print_step "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        sleep 5
        
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama started (PID: $OLLAMA_PID)"
        else
            print_warning "Failed to start Ollama. Please start manually: ollama serve"
        fi
    fi
    
    # Download llava model
    print_step "Downloading llava:7b model (this may take several minutes)..."
    if ollama list | grep -q "llava:7b"; then
        print_success "llava:7b already downloaded"
    else
        ollama pull llava:7b
        print_success "llava:7b model downloaded"
    fi
}

# =============================================================================
# CREATE DIRECTORY STRUCTURE
# =============================================================================

create_directories() {
    print_step "Creating directory structure..."
    
    DIRS=(
        "vector_db"
        "reference_icons"
        "knowledge_base"
        "test_reports"
        "logs"
        "screenshots"
        "prompts"
        "prompts/component_specific"
    )
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$SCRIPT_DIR/$dir"
    done
    
    print_success "Directory structure created"
}

# =============================================================================
# INITIALIZE PROMPT FILES
# =============================================================================

initialize_prompts() {
    print_step "Checking prompt files..."
    
    if [ ! -f "$SCRIPT_DIR/prompts/base_prompts.md" ]; then
        print_warning "Prompt files not found. Please ensure prompts are in ./prompts/ directory"
    else
        print_success "Prompt files found"
    fi
}

# =============================================================================
# INITIALIZE RAG SYSTEM
# =============================================================================

initialize_rag() {
    print_step "Initializing RAG system (embeddings and vector database)..."
    
    if [ -f "$SCRIPT_DIR/init_rag_system.py" ]; then
        $PYTHON_CMD "$SCRIPT_DIR/init_rag_system.py"
        print_success "RAG system initialized"
    else
        print_warning "init_rag_system.py not found. Skipping RAG initialization."
        print_warning "You can run it manually later: python init_rag_system.py"
    fi
}

# =============================================================================
# VERIFY ADB INSTALLATION
# =============================================================================

verify_adb() {
    print_step "Checking ADB (Android Debug Bridge)..."
    
    if command -v adb &> /dev/null; then
        ADB_VERSION=$(adb version | head -n 1)
        print_success "ADB installed: $ADB_VERSION"
        
        # Check for connected devices
        DEVICES=$(adb devices | grep -v "List" | grep "device" | wc -l)
        if [ "$DEVICES" -gt 0 ]; then
            print_success "$DEVICES Android device(s) connected"
            adb devices
        else
            print_warning "No Android devices connected"
            print_warning "Please connect your Android device with USB debugging enabled"
        fi
    else
        print_warning "ADB not found. Install platform-tools for Android device testing"
        if [[ "$OS" == "Linux" ]]; then
            print_warning "Install with: sudo apt-get install android-tools-adb"
        elif [[ "$OS" == "macOS" ]]; then
            print_warning "Install with: brew install android-platform-tools"
        fi
    fi
}

# =============================================================================
# CREATE ENVIRONMENT FILE
# =============================================================================

create_env_file() {
    print_step "Creating environment configuration..."
    
    ENV_FILE="$SCRIPT_DIR/.env"
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << 'EOF'
# AI Agent Framework Configuration
# Edit these values as needed

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:7b

# RAG Configuration
RAG_ENABLED=true
VECTOR_DB_PATH=./vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LangChain Configuration
LANGCHAIN_ENABLED=true
LANGCHAIN_TEMPERATURE=0.3

# LangGraph Configuration
LANGGRAPH_ENABLED=true

# Voice Interface
VOICE_ENABLED=false

# Device Configuration
ADB_TIMEOUT=10
ACTION_DELAY=1.0

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
EOF
        print_success "Environment file created: .env"
    else
        print_warning ".env file already exists"
    fi
}

# =============================================================================
# RUN VERIFICATION TESTS
# =============================================================================

run_verification() {
    print_step "Running verification tests..."
    
    # Test Python imports
    print_step "Testing Python imports..."
    $PYTHON_CMD - << 'EOF'
import sys
try:
    import torch
    import PIL
    import cv2
    import numpy
    import ollama
    print("✅ Core dependencies OK")
    
    import langchain
    import chromadb
    import sentence_transformers
    print("✅ RAG dependencies OK")
    
    print("✅ All imports successful")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Verification tests passed"
    else
        print_error "Verification tests failed"
        return 1
    fi
}

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print_summary() {
    print_header "INSTALLATION COMPLETE!"
    
    echo -e "${GREEN}"
    echo "✅ Python $PYTHON_VERSION installed"
    echo "✅ Dependencies installed"
    echo "✅ Directory structure created"
    if [ "$SKIP_OLLAMA" = false ]; then
        echo "✅ Ollama installed and llava:7b downloaded"
    fi
    echo "✅ Configuration files created"
    echo -e "${NC}"
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}NEXT STEPS:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ "$USE_VENV" = true ]; then
        echo -e "1. ${YELLOW}Activate virtual environment:${NC}"
        echo -e "   ${CYAN}source venv/bin/activate${NC}"
        echo ""
    fi
    
    if [ "$SKIP_OLLAMA" = false ]; then
        echo -e "2. ${YELLOW}Ensure Ollama is running:${NC}"
        echo -e "   ${CYAN}ollama serve${NC}"
        echo ""
    fi
    
    echo -e "3. ${YELLOW}Connect Android device:${NC}"
    echo -e "   ${CYAN}adb devices${NC}"
    echo ""
    
    echo -e "4. ${YELLOW}Run your first test:${NC}"
    echo -e "   ${CYAN}python3 gui_controller_voice.py${NC}"
    echo -e "   ${CYAN}# or${NC}"
    echo -e "   ${CYAN}python3 prompt_driven_agent.py --test-id \"NAID-24430\"${NC}"
    echo ""
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}DOCUMENTATION:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  📖 Full documentation:    ${CYAN}README.md${NC}"
    echo -e "  ⚡ Quick start guide:     ${CYAN}QUICK_START.md${NC}"
    echo -e "  🏗️  Architecture:          ${CYAN}ARCHITECTURE_DIAGRAM.md${NC}"
    echo -e "  🔧 Setup instructions:    ${CYAN}SETUP_INSTRUCTIONS.txt${NC}"
    echo ""
    
    echo -e "${GREEN}🚀 Your AI Agent framework is ready to use!${NC}"
    echo ""
}

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-venv)
                USE_VENV=false
                shift
                ;;
            --gpu)
                GPU_SUPPORT=true
                shift
                ;;
            --no-gpu)
                GPU_SUPPORT=false
                shift
                ;;
            --skip-ollama)
                SKIP_OLLAMA=true
                shift
                ;;
            --help|-h)
                echo "AI Agent Framework Setup Script"
                echo ""
                echo "Usage: ./setup.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --no-venv        Skip virtual environment creation"
                echo "  --gpu            Force GPU support (install CUDA PyTorch)"
                echo "  --no-gpu         Force CPU only (install CPU PyTorch)"
                echo "  --skip-ollama    Skip Ollama installation"
                echo "  --help, -h       Show this help message"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    parse_args "$@"
    
    print_header "AI AGENT FRAMEWORK - AUTOMATED SETUP"
    echo -e "${CYAN}Enhanced with RAG, LangChain & LangGraph${NC}"
    echo ""
    
    detect_python
    check_system_requirements
    create_venv
    install_dependencies
    install_ollama
    create_directories
    initialize_prompts
    create_env_file
    initialize_rag
    verify_adb
    run_verification
    
    echo ""
    print_summary
}

# Run main function
main "$@"
