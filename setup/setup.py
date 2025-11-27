"""
setup.py - Python Package Setup for AI Agent Framework

This file enables installation of the AI Agent framework as a Python package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ai-agent-automotive",
    version="2.0.0",
    author="Veera Saravanan",
    author_email="",
    description="Neural AI Agent for Automotive Testing - Enhanced with RAG, LangChain & LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core AI & Vision
        "ollama>=0.1.6",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "adb-shell>=0.4.4",
        
        # OCR
        "easyocr>=1.7.0",
        "paddleocr>=2.7.0",
        "pytesseract>=0.3.10",
        
        # Excel/PDF Support
        "PyPDF2>=3.0.0",
        "openpyxl>=3.1.0",
        "pandas>=2.0.0",
        "xlsxwriter>=3.1.0",
        
        # GUI & Voice
        "PyQt6>=6.5.0",
        "pyttsx3>=2.90",
        "SpeechRecognition>=3.10.0",
        "pyaudio>=0.2.13",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "prompt-toolkit>=3.0.0",
        "tqdm>=4.66.0",
        "colorama>=0.4.6",
        "scikit-image>=0.21.0",
        "torch>=2.0.0",
        
        # LangChain Ecosystem
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-ollama>=0.0.1",
        "langgraph>=0.0.20",
        
        # Embeddings & Vector Database
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        
        # Structured Output & NLP
        "pydantic>=2.5.0",
        "jsonschema>=4.20.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.19.0",
        "typing-extensions>=4.8.0",
        "loguru>=0.7.2",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",  # Will be installed with CUDA support separately
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-agent=prompt_driven_agent:main",
            "ai-agent-gui=gui_controller_voice:main",
            "ai-agent-init=init_rag_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "prompts/*.md",
            "prompts/component_specific/*.md",
            "reference_icons/*",
            "knowledge_base/*.xlsx",
        ],
    },
    zip_safe=False,
)
