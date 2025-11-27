"""
check_gui_dependencies.py - Pre-flight Check for GUI Launch

Run this before launching the GUI to verify all dependencies are installed
and configured correctly.

Usage:
    python check_gui_dependencies.py
"""

import sys
import subprocess

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_python_version():
    """Check Python version."""
    print("\n✓ Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required")
        return False
    
    print("  ✅ Python version OK")
    return True

def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"  ✅ {package_name:30} - Installed")
        return True
    except ImportError:
        print(f"  ❌ {package_name:30} - NOT INSTALLED")
        return False

def check_python_packages():
    """Check all required Python packages."""
    print("\n✓ Checking Python packages...")
    
    packages = {
        'PyQt6': 'PyQt6',
        'sentence_transformers': 'sentence_transformers',
        'chromadb': 'chromadb',
        'langchain': 'langchain',
        'langchain_ollama': 'langchain_ollama',
        'langgraph': 'langgraph',
        'ollama': 'ollama',
        'pyttsx3': 'pyttsx3',
        'speech_recognition': 'speech_recognition',
        'easyocr': 'easyocr',
        'paddleocr': 'paddleocr',
        'opencv-python': 'cv2',
        'pandas': 'pandas',
        'xlsxwriter': 'xlsxwriter',
        'Pillow': 'PIL',
    }
    
    all_installed = True
    missing = []
    
    for package, import_name in packages.items():
        if not check_package(package, import_name):
            all_installed = False
            missing.append(package)
    
    if not all_installed:
        print("\n⚠️  Missing packages detected!")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
    
    return all_installed

def check_ollama():
    """Check if Ollama is running and models are available."""
    print("\n✓ Checking Ollama...")
    
    try:
        import ollama
        
        # Check if server is running
        try:
            models = ollama.list()
            print("  ✅ Ollama server is running")
            
            # List available models
            model_names = [m.model for m in models.models]
            print(f"  Available models: {', '.join(model_names)}")
            
            # Check for llava
            has_llava = any('llava' in name for name in model_names)
            
            if has_llava:
                print("  ✅ llava model found")
                return True
            else:
                print("  ⚠️  llava model not found")
                print("  Install with: ollama pull llava:7b")
                return False
                
        except Exception as e:
            print(f"  ❌ Ollama server not running")
            print(f"  Error: {e}")
            print("  Start with: ollama serve")
            return False
            
    except ImportError:
        print("  ❌ Ollama package not installed")
        print("  Install with: pip install ollama")
        return False

def check_config():
    """Check configuration file."""
    print("\n✓ Checking configuration...")
    
    try:
        from config import AgentConfig
        
        print(f"  RAG Enabled: {AgentConfig.RAG_ENABLED}")
        print(f"  LangChain Enabled: {AgentConfig.LANGCHAIN_ENABLED}")
        print(f"  Voice Enabled: {AgentConfig.VOICE_ENABLED}")
        print(f"  LLM Model: {AgentConfig.LANGCHAIN_SETTINGS.get('model', 'unknown')}")
        
        # Check model name
        model = AgentConfig.LANGCHAIN_SETTINGS.get('model', '')
        if not model:
            print("  ⚠️  No model configured in config.py")
            return False
        
        print("  ✅ Configuration loaded")
        return True
        
    except ImportError as e:
        print(f"  ❌ Cannot load config.py")
        print(f"  Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Configuration error")
        print(f"  Error: {e}")
        return False

def check_prompts_directory():
    """Check if prompts directory exists."""
    print("\n✓ Checking prompts directory...")
    
    from pathlib import Path
    
    prompts_dir = Path("./prompts")
    
    if not prompts_dir.exists():
        print("  ❌ Prompts directory not found")
        print("  Create: ./prompts/")
        return False
    
    # Check for key files
    required_files = [
        'base_prompts.md',
        'error_handling.md',
        'learned_solutions.md'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = prompts_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ⚠️  {filename} - missing (optional)")
    
    print("  ✅ Prompts directory exists")
    return True

def check_embeddings():
    """Check if embeddings are initialized."""
    print("\n✓ Checking embeddings...")
    
    from pathlib import Path
    
    vector_db = Path("./vector_db")
    
    if not vector_db.exists():
        print("  ⚠️  Vector database not initialized")
        print("  Run: python init_rag_system.py")
        return False
    
    print("  ✅ Vector database found")
    return True

def main():
    """Run all checks."""
    print_header("GUI Dependencies Pre-flight Check")
    
    results = {
        'Python Version': check_python_version(),
        'Python Packages': check_python_packages(),
        'Ollama': check_ollama(),
        'Configuration': check_config(),
        'Prompts Directory': check_prompts_directory(),
        'Embeddings': check_embeddings(),
    }
    
    print_header("Summary")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status:12} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("\n✅ SUCCESS! All checks passed. GUI is ready to launch.")
        print("\nLaunch GUI with:")
        print("  python gui_controller.py")
        print("  python gui_controller_voice.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before launching GUI.")
        print("\nQuick fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull model: ollama pull llava:7b")
        print("  4. Initialize RAG: python init_rag_system.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())