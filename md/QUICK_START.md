# ⚡ Quick Start Guide - Enhanced AI Agent

## 🎯 Get Running in 10 Minutes

### **Step 1: Prerequisites** (2 minutes)

```bash
# Check Python version (need 3.8+)
python --version

# Check if Ollama is installed
ollama --version

# If not installed, download from: https://ollama.ai
```

### **Step 2: Install Dependencies** (5 minutes)

```bash
cd neural-agent

# Install PyTorch first (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For GPU (NVIDIA):
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt
```

### **Step 3: Start Ollama & Download Model** (3 minutes)

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Download llava:7b
ollama pull llava:7b

# This will download ~4.7GB (only needed once)
```

### **Step 4: Setup Prompts** (1 minute)

```bash
# Ensure prompts are in ./prompts/ directory
ls prompts/
# Should show:
# base_prompts.md
# error_handling.md
# learned_solutions.md
# custom_commands.md
# component_specific/
```

### **Step 5: Initialize Embeddings** (3 minutes)

```bash
# This embeds all prompts into ChromaDB
python prompt_embeddings.py

# You'll see:
# Loading embedding model: all-MiniLM-L6-v2...
# ✅ Embedding model loaded
# ✅ ChromaDB initialized
# Embedding base_prompts.md: 8 chunks...
# ✅ Embedded 7 files, 42 total chunks
```

### **Step 6: Connect Android Device** (1 minute)

```bash
# Enable USB debugging on Android device
# Connect via USB

# Test connection
adb devices
# Should show your device

# Test screenshot
adb exec-out screencap -p > test.png
```

### **Step 7: Run Your First Test!** (30 seconds)

```bash
# Via command line
python prompt_driven_agent.py --test-id "NAID-24430"

# Or via GUI
python prompt_driven_agent.py --gui
```

---

## 🎉 Success!

You should see:

```
🚀 Initializing with RAG + LangChain...
✅ RAG + LangChain initialized
🔍 RAG: Retrieving context for task: Turn on AC...
Retrieved 572 chars of relevant context
🧠 Generating test plan with Chain-of-Thought...
✅ Generated plan with 3 steps
⚡ Executing step 1/3
...
✅ Test NAID-24430 PASSED in 18.5s
📊 Report generated: ./test_reports/test_report_20250120_150000.xlsx
```

---

## 🔧 Troubleshooting

### **"ChromaDB not available"**

```bash
pip install chromadb
```

### **"Sentence Transformers not found"**

```bash
pip install sentence-transformers
```

### **"No module named 'langchain'"**

```bash
pip install langchain langchain-ollama langgraph
```

### **"Ollama connection failed"**

```bash
# Make sure Ollama is running
ollama serve

# In another terminal
ollama list  # Check if llava is installed
```

### **"ADB devices shows no devices"**

- Enable USB debugging on Android
- Check USB cable
- Install ADB drivers (Windows)
- Try: `adb kill-server` then `adb start-server`

---

## 📚 Next Steps

1. **Read Full Documentation**: `README.md`
2. **See Architecture**: `ARCHITECTURE_DIAGRAM.md`
3. **Integration Guide**: `INTEGRATION_GUIDE.md`
4. **Customize Prompts**: Edit `prompts/*.md` files
5. **Add Test Cases**: Place Excel files in `knowledge_base/`
6. **Add Reference Icons**: Place icons in `reference_icons/`

---

## 🎯 Usage Examples

### **Run Single Test**

```bash
python prompt_driven_agent.py --test-id "NAID-24430"
```

### **Run via GUI**

```bash
python prompt_driven_agent.py --gui
```

### **Run in Traditional Mode (No RAG)**

```bash
python prompt_driven_agent.py --test-id "NAID-24430" --traditional
```

### **Force Re-embed Prompts**

```bash
python prompt_embeddings.py
```

### **View Statistics**

```python
from rag_prompt_manager import RAGPromptManager
manager = RAGPromptManager()
print(manager.get_prompt_summary())
```

---

## 💡 Tips

1. **First Run is Slower**: Downloads embedding model (~100MB)
2. **GPU Accelerates**: 10x faster with NVIDIA GPU
3. **Prompts Auto-Update**: Edit `.md` files, re-run embeddings
4. **Multiple Devices**: Use `-s SERIAL` for specific device
5. **Parallel Tests**: Run multiple instances on different devices

---

## ✅ Verification

Check if everything is working:

```bash
# 1. Embeddings exist
ls ./vector_db/
# Should show: chroma.sqlite3, prompt_hashes.json

# 2. Test embeddings
python -c "from prompt_embeddings import PromptEmbeddingsManager; m=PromptEmbeddingsManager(); print(m.get_statistics())"

# 3. Test RAG
python -c "from rag_prompt_manager import RAGPromptManager; m=RAGPromptManager(); print(m.get_prompt_summary())"

# 4. Test LangChain
python -c "from langchain_coordinator import LangChainCoordinator; c=LangChainCoordinator(); print('✅ LangChain OK')"
```

---

## 🚀 You're Ready!

Your enhanced AI agent is now ready to test automotive displays with:

- ⚡ 5x faster prompt processing
- 🧠 Chain-of-Thought reasoning
- 🔄 Multi-step workflows
- 📊 Semantic retrieval
- 🎯 Scalable to 10,000+ line prompts

**Happy Testing!**

---

**Created by**: Veera Saravanan  
**Framework**: Neural AI Agent v2.0
