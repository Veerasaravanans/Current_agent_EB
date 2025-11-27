"""
fix_pytorch_dll_error.py - Fix PyTorch DLL initialization error on Windows

This script fixes the common PyTorch DLL error by:
1. Uninstalling incompatible PyTorch versions
2. Installing CPU-only PyTorch (lighter and more compatible)
3. Reinstalling easyocr with the fixed PyTorch

Run this script to resolve: OSError: [WinError 1114] DLL initialization routine failed
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n{'='*70}")
    print(f"⚡ {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
    else:
        print(f"⚠️  {description} - Completed with warnings (this may be normal)")
    
    return result.returncode

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        PyTorch DLL Error Fix for Windows                        ║
║                                                                  ║
║  This will fix: OSError [WinError 1114]                         ║
║  DLL initialization routine failed                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  IMPORTANT: This will reinstall PyTorch and EasyOCR")
    print("This may take 2-5 minutes depending on your internet speed.\n")
    
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Cancelled by user.")
        return
    
    print("\n🚀 Starting fix process...")
    
    # Step 1: Uninstall existing torch packages
    run_command(
        "pip uninstall -y torch torchvision torchaudio",
        "Step 1: Uninstalling existing PyTorch packages"
    )
    
    # Step 2: Install PyTorch CPU version (more stable on Windows)
    run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Step 2: Installing PyTorch CPU version (stable for Windows)"
    )
    
    # Step 3: Reinstall easyocr
    run_command(
        "pip install --force-reinstall --no-cache-dir easyocr",
        "Step 3: Reinstalling EasyOCR"
    )
    
    print("\n" + "="*70)
    print("✅ FIX COMPLETED!")
    print("="*70)
    
    # Test the fix
    print("\n🧪 Testing if the fix worked...")
    print("\nTrying to import torch and easyocr...\n")
    
    test_code = """
import sys
try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"   PyTorch version: {torch.__version__}")
    
    import easyocr
    print("✅ EasyOCR imported successfully")
    
    print("\\n" + "="*70)
    print("🎉 SUCCESS! PyTorch and EasyOCR are working correctly")
    print("="*70)
    print("\\nYou can now run: python gui_controller_voice.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\\nIf the error persists, you need to install:")
    print("   Microsoft Visual C++ Redistributable")
    print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    sys.exit(1)
"""
    
    subprocess.run([sys.executable, "-c", test_code])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
