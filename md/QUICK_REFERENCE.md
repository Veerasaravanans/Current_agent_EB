# 📖 Quick Reference Card - Enhanced Automotive AI Agent

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize vector database
python enhanced_prompt_manager.py

# 3. Start Ollama
ollama serve

# 4. Run your first test
python prompt_driven_agent.py --test-id "YOUR-TEST-ID"
```

---

## 🔑 Key Commands

### **Run Tests**
```bash
# Single test by ID
python prompt_driven_agent.py --test-id "NAID-24430"

# All tests for component
python prompt_driven_agent.py --component hvac

# With voice interface
python prompt_driven_agent.py --test-id "NAID-24430" --voice

# GUI mode
python prompt_driven_agent.py --gui
```

### **Test Components**
```bash
# Test RAG retrieval
python enhanced_prompt_manager.py

# Test LangChain
python langchain_integration.py

# Test LangGraph workflow
python langgraph_workflow.py

# Verify installation
python test_installation.py
```

### **Manage Vector Database**
```bash
# Rebuild vector DB (if prompts updated)
python -c "
from enhanced_prompt_manager import EnhancedPromptManager
pm = EnhancedPromptManager()
pm.load_and_embed_prompts(force_rebuild=True)
"

# Check DB status
python -c "
from enhanced_prompt_manager import EnhancedPromptManager
pm = EnhancedPromptManager()
pm.load_and_embed_prompts()
print(pm.get_prompt_statistics())
"
```

---

## 📁 File Quick Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `enhanced_prompt_manager.py` | RAG + Embeddings | Prompt handling |
| `langchain_integration.py` | LLM orchestration | Decision making |
| `langgraph_workflow.py` | State machine workflows | Multi-step tests |
| `prompt_driven_agent.py` | Main entry point | Run tests |
| `vision_coordinator.py` | OCR + Image matching | Find elements |
| `automotive_operating_system.py` | ADB control | Execute actions |

---

## 🔧 Configuration Options

### **Enhanced Prompt Manager**
```python
pm = EnhancedPromptManager(
    prompts_dir="./prompts",           # Prompt files location
    vector_db_dir="./vector_db",       # ChromaDB storage
    embedding_model="all-MiniLM-L6-v2" # Sentence transformer
)
```

### **RAG Retrieval**
```python
results = pm.retrieve_relevant_prompts(
    query="How to tap AC?",  # What you need guidance on
    component="hvac",        # Optional component filter
    top_k=5,                 # Number of chunks to retrieve
    min_score=0.5            # Minimum relevance (0.0-1.0)
)
```

### **LangChain Orchestrator**
```python
orchestrator = LangChainTestOrchestrator(
    model_name="moondream",              # Ollama model
    base_url="http://localhost:11434"    # Ollama server
)
```

### **LangGraph Workflow**
```python
workflow = AutomotiveTestWorkflow(
    vision_coordinator=vision,  # VisionCoordinator instance
    operating_system=os_ctrl    # AutomotiveOperatingSystem instance
)

result = workflow.run_test(
    test_id="TEST-001",
    objective="Tap AC button",
    component="hvac",
    total_steps=1
)
```

---

## 🎯 Common Tasks

### **Task 1: Run Test with RAG**
```python
from enhanced_prompt_manager import EnhancedPromptManager

pm = EnhancedPromptManager()
pm.load_and_embed_prompts()

# Get optimized prompt for objective
prompt = pm.get_system_prompt_with_rag(
    objective="Turn on AC and set to 72°F",
    component="hvac",
    max_context_length=2000  # Optimized for lightweight models
)
```

### **Task 2: Search Knowledge by Meaning**
```python
# Finds relevant knowledge even with different words
results = pm.retrieve_relevant_prompts(
    query="How do I activate cooling?",  # Uses word "cooling"
    top_k=3
)
# Returns: Info about "AC", "air conditioning", "climate control"
# Because it searches by MEANING, not exact keywords!
```

### **Task 3: Add Learned Solution**
```python
# When agent gets stuck and you provide solution
pm.add_learned_solution(
    problem="Temperature slider not responding",
    solution="Use longer swipe duration (800ms)",
    added_by="Veera Saravanan"
)
# Automatically:
# ✅ Added to learned_solutions.md
# ✅ Embedded in vector DB
# ✅ Retrievable by future queries
```

### **Task 4: Chain-of-Thought Decision**
```python
from langchain_integration import LangChainTestOrchestrator

orchestrator = LangChainTestOrchestrator()

decision = orchestrator.decide_action(
    objective="Turn on AC",
    screen_state="HVAC controls visible",
    previous_actions=[],
    attempt_number=1
)
# Returns structured decision with reasoning:
# - action_type: "tap"
# - target: "AC"
# - method: "ocr"
# - reasoning: "Step 1: Understand... Step 2: Analyze..."
```

### **Task 5: Multi-Step Workflow**
```python
from langgraph_workflow import MultiStepTestWorkflow

workflow = MultiStepTestWorkflow()

# Runs complete test with automatic:
# - Step sequencing
# - 10-attempt retry per step
# - Error recovery
# - Result verification
result = workflow.run_hvac_complete_test()
```

---

## 📊 Performance Tips

### **Speed Up Vector DB Initialization**
```python
# First time (slow - builds DB):
pm = EnhancedPromptManager()
pm.load_and_embed_prompts()  # 30-60s

# Subsequent runs (fast - loads DB):
pm = EnhancedPromptManager()
pm.load_and_embed_prompts()  # 2-5s ⚡
```

### **Use GPU for Acceleration**
```python
# In enhanced_prompt_manager.py, change:
self.embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={'device': 'cuda'},  # ✅ Use GPU
    # model_kwargs={'device': 'cpu'},  # ❌ CPU only
)
# Speedup: 2-5x faster with GPU
```

### **Optimize Chunk Size**
```python
# For better speed (less chunks):
pm.text_splitter.chunk_size = 768

# For better accuracy (more chunks):
pm.text_splitter.chunk_size = 256

# Balanced (default):
pm.text_splitter.chunk_size = 512
```

---

## 🐛 Quick Troubleshooting

### **"Vector DB slow to initialize"**
```bash
# Solution: Pre-build once
python enhanced_prompt_manager.py
# Then it loads in 2-5s in your code
```

### **"Module not found: langchain"**
```bash
pip install langchain langchain-community langchain-core
```

### **"ChromaDB error"**
```bash
pip install chromadb sentence-transformers
```

### **"Ollama not responding"**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### **"Out of memory"**
```python
# Reduce chunk size in enhanced_prompt_manager.py
chunk_size=256  # Instead of 512
```

---

## 📈 Metrics to Monitor

```python
# Get performance statistics
stats = pm.get_prompt_statistics()
print(f"Vector DB initialized: {stats['vector_db_initialized']}")
print(f"Total documents: {stats['total_documents']}")
print(f"Embedding model: {stats['embedding_model']}")

# Monitor retrieval performance
import time
start = time.time()
results = pm.retrieve_relevant_prompts("test query")
print(f"Retrieval time: {(time.time() - start)*1000:.0f}ms")
```

**Target Performance**:
- RAG retrieval: <100ms (CPU), <50ms (GPU)
- Vector DB load: <5s (after first build)
- Decision chain: <2s
- Complete test step: <5s (with OCR success)

---

## 🎓 Key Concepts

### **RAG (Retrieval Augmented Generation)**
Instead of loading ALL prompts (5000+ lines), retrieve only relevant chunks:
```
Query: "How to tap AC?" 
  ↓
Vector DB semantic search
  ↓
Top 5 relevant chunks (~2000 chars)
  ↓
Assemble prompt with only relevant context
  ↓
Lightweight model handles it easily! ✅
```

### **Embeddings**
Convert text to vectors that capture semantic meaning:
```
"AC button" → [0.23, 0.87, -0.45, ...]
"air conditioning" → [0.25, 0.89, -0.43, ...]  # Similar vector!
"hamburger" → [-0.82, 0.12, 0.67, ...]  # Different vector
```

### **Chain-of-Thought**
Force model to think step-by-step:
```
Standard: "Tap AC" → Guesses
Chain-of-Thought:
  "Step 1: Understand task - Need to activate AC
   Step 2: Analyze screen - HVAC controls visible
   Step 3: Plan approach - AC has text 'AC'
   Step 4: Decide method - Use OCR
   Step 5: Execute - Tap coordinates"
→ Systematic reasoning ✅
```

### **State Machine (LangGraph)**
Clean workflow instead of nested if/else:
```
States: capture → analyze → decide → execute → verify
Edges: success → next_step
       failure → retry (if < 10 attempts)
       failure → ask_architect (if = 10 attempts)
```

---

## 💡 Pro Tips

1. **Build vector DB once**: Run `enhanced_prompt_manager.py` standalone first
2. **Use GPU if available**: Set `device='cuda'` in embeddings
3. **Monitor retrieval scores**: If < 0.5, results may not be relevant
4. **Start with small tests**: Test 1-step before running 10-step tests
5. **Check logs**: Enable `logging.DEBUG` to see detailed workflow
6. **Update prompts frequently**: Add new solutions to learned_solutions.md
7. **Use semantic queries**: "How to activate cooling?" instead of exact keywords

---

## 📚 Quick Links

- **Full Documentation**: [README.md](README.md)
- **Implementation Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Integration Guide**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **LangChain Docs**: https://python.langchain.com/docs/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **ChromaDB Docs**: https://docs.trychroma.com/

---

## 🆘 Getting Help

1. Check error message in logs
2. Review relevant documentation file
3. Test component individually (run its standalone test)
4. Check requirements.txt versions match
5. Verify Ollama is running: `ollama list`
6. Rebuild vector DB: `rm -rf vector_db && python enhanced_prompt_manager.py`

---

**Quick Reference v1.0** | Enhanced Automotive AI Agent  
*Keep this handy while developing!* 📌
