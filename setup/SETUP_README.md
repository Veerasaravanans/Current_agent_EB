# Setup Files for AI Agent Framework

This directory contains comprehensive setup and installation scripts for the AI Agent Framework.

## 📋 Available Setup Files

### 1. **setup.sh** (Recommended for Linux/macOS)
Automated bash script for Unix-based systems.

**Features:**
- Auto-detects Python installation
- Creates virtual environment
- Installs all dependencies (CPU/GPU)
- Installs and configures Ollama
- Downloads llava:7b model
- Initializes RAG system
- Creates directory structure
- Runs verification tests

**Usage:**
```bash
chmod +x setup.sh
./setup.sh

# Options:
./setup.sh --no-venv      # Skip virtual environment
./setup.sh --gpu          # Force GPU support
./setup.sh --skip-ollama  # Skip Ollama installation
./setup.sh --help         # Show help
```

### 2. **setup.bat** (Windows)
Automated batch script for Windows systems.

**Features:**
- Detects Python installation
- Creates virtual environment
- Installs dependencies with appropriate PyTorch version
- Creates directory structure
- Guides Ollama installation
- Initializes RAG system

**Usage:**
```cmd
setup.bat
```
Double-click the file or run from Command Prompt.

### 3. **setup.py** (Python Package Installation)
Standard Python package setup file.

**Features:**
- Installs framework as a Python package
- Manages all dependencies
- Creates command-line entry points
- Supports development mode

**Usage:**
```bash
# Install in development mode (recommended)
pip install -e .

# Install normally
pip install .

# Install with development dependencies
pip install -e ".[dev]"
```

**Entry Points Created:**
- `ai-agent` - Main CLI
- `ai-agent-gui` - GUI interface
- `ai-agent-init` - RAG initialization

### 4. **verify_setup.py** (Verification Script)
Comprehensive verification script to check installation.

**Features:**
- Verifies Python version
- Checks all dependencies
- Tests Ollama service
- Validates directory structure
- Checks vector database
- Tests ADB connectivity
- Detects GPU support

**Usage:**
```bash
python3 verify_setup.py
```

## 🚀 Quick Start

### Linux/macOS (Recommended)
```bash
# 1. Clone or download the framework
cd ai-agent-main

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Verify installation
python3 verify_setup.py

# 4. Activate environment (if using venv)
source venv/bin/activate

# 5. Run the framework
python3 gui_controller_voice.py
```

### Windows
```cmd
REM 1. Navigate to directory
cd ai-agent-main

REM 2. Run setup
setup.bat

REM 3. Verify installation
python verify_setup.py

REM 4. Activate environment
venv\Scripts\activate.bat

REM 5. Run the framework
python gui_controller_voice.py
```

### Python Package Installation
```bash
# Install in editable mode
pip install -e .

# Verify installation
ai-agent --help
ai-agent-gui
```

## 📦 What Gets Installed

### Core Components
- ✅ Python dependencies (60+ packages)
- ✅ PyTorch (CPU or GPU version)
- ✅ Ollama + llava:7b model (~4.7GB)
- ✅ LangChain ecosystem
- ✅ RAG dependencies (ChromaDB, Sentence Transformers)
- ✅ OCR engines (EasyOCR, PaddleOCR)
- ✅ GUI libraries (PyQt6)
- ✅ Voice interface (pyttsx3, SpeechRecognition)

### Directory Structure Created
```
ai-agent-main/
├── vector_db/              # ChromaDB storage
├── reference_icons/        # UI element icons
├── knowledge_base/         # Test case Excel files
├── test_reports/           # Generated test reports
├── logs/                   # Application logs
├── screenshots/            # Captured screenshots
├── prompts/                # AI prompts
│   └── component_specific/ # Component-specific prompts
└── venv/                   # Virtual environment (optional)
```

### Configuration Files
- `.env` - Environment variables
- `config.py` - Framework configuration

## ⚙️ Configuration

### Environment Variables (.env)
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:7b
RAG_ENABLED=true
LANGCHAIN_ENABLED=true
LANGGRAPH_ENABLED=true
VOICE_ENABLED=false
```

### Custom Configuration (config.py)
Edit `config.py` to customize:
- RAG settings (context size, embedding model)
- LangChain parameters (temperature, retries)
- Retry strategies
- Vision settings
- Voice interface
- Logging levels

## 🔍 Verification Checklist

After running setup, verify:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] Ollama service running (`ollama serve`)
- [ ] llava:7b model downloaded (`ollama list`)
- [ ] Vector database initialized (`ls vector_db/`)
- [ ] Prompts directory populated (`ls prompts/`)
- [ ] ADB working (`adb devices`)
- [ ] Virtual environment created (if using)

Run verification script:
```bash
python3 verify_setup.py
```

## 🔧 Troubleshooting

### Python not found
- **Linux/macOS:** Install Python 3.8+ from package manager or python.org
- **Windows:** Download from python.org, ensure "Add to PATH" is checked

### Permission denied (Linux/macOS)
```bash
chmod +x setup.sh
chmod +x verify_setup.py
```

### Ollama connection failed
```bash
# Start Ollama service
ollama serve

# In another terminal, pull model
ollama pull llava:7b
```

### Virtual environment issues
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### GPU not detected
- Install NVIDIA drivers
- Install CUDA toolkit
- Reinstall PyTorch with CUDA:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### ChromaDB errors
```bash
# Reinstall ChromaDB
pip uninstall chromadb
pip install chromadb

# Re-initialize
python3 init_rag_system.py
```

### ADB not found
- **Linux:** `sudo apt-get install android-tools-adb`
- **macOS:** `brew install android-platform-tools`
- **Windows:** Download Android Platform Tools

## 📚 Post-Installation Steps

### 1. Configure Prompts
Place your prompt files in `prompts/`:
- `base_prompts.md` - Core instructions
- `error_handling.md` - Error solutions
- `learned_solutions.md` - Auto-learned fixes
- `custom_commands.md` - ADB commands

### 2. Add Test Cases
Place Excel test case files in `knowledge_base/`

### 3. Add Reference Icons
Place UI element icons in `reference_icons/` for image matching

### 4. Initialize RAG
```bash
python3 init_rag_system.py
```
This embeds all prompts into the vector database.

### 5. Connect Device
```bash
# Enable USB debugging on Android device
adb devices
```

### 6. Run First Test
```bash
# GUI mode
python3 gui_controller_voice.py

# CLI mode
python3 prompt_driven_agent.py --test-id "NAID-24430"

# Traditional mode (no RAG)
python3 prompt_driven_agent.py --test-id "NAID-24430" --traditional
```

## 🎯 Performance Optimization

### For CPU-Only Systems
- Use smaller context size: Edit `config.py`, set `max_context_size=2000`
- Reduce retries: Set `max_retries=5`
- Use fast mode: `Presets.fast_mode()`

### For GPU Systems
- Install CUDA version of PyTorch
- Enable GPU in config
- Larger context sizes work well

### For Low Memory Systems
- Close other applications
- Use traditional mode (less memory)
- Reduce screenshot quality

## 📖 Additional Documentation

- **README.md** - Full framework documentation
- **QUICK_START.md** - Quick start guide
- **SETUP_INSTRUCTIONS.txt** - Detailed setup instructions
- **ARCHITECTURE_DIAGRAM.md** - System architecture
- **INTEGRATION_GUIDE.md** - Integration guide

## 🆘 Getting Help

1. Run verification: `python3 verify_setup.py`
2. Check logs: `cat logs/agent.log`
3. Review documentation: `README.md`
4. Check Ollama: `ollama list`
5. Test imports: `python3 test_imports.py`

## ✅ All Set!

Once setup is complete and verification passes, you're ready to:
- 🚗 Test Android Automotive systems
- 🧠 Use AI-powered visual understanding
- 📊 Generate automated test reports
- 🎯 Achieve 10x faster testing than manual

**Happy Testing!** 🎉

---

**Created by:** Veera Saravanan  
**Framework:** Neural AI Agent v2.0 (RAG-Enhanced)  
**License:** MIT
