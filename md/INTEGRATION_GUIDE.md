# 🔧 Integration Guide: RAG + LangChain into Existing Framework

## 📋 Overview

This guide shows how to integrate the new RAG and LangChain components into your existing `prompt_driven_agent.py` framework **without breaking existing functionality**.

---

## 🎯 Key Integration Points

### **1. Imports**
Add at top of `prompt_driven_agent.py`:
```python
try:
    from rag_prompt_manager import RAGPromptManager
    from langchain_coordinator import LangChainCoordinator
    from langgraph_workflow import LangGraphWorkflowManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
```

### **2. Initialization**
```python
if RAG_AVAILABLE:
    self.rag_manager = RAGPromptManager()
    self.langchain = LangChainCoordinator()
    self.workflow = LangGraphWorkflowManager(self.langchain)
```

### **3. Usage**
Replace traditional prompt loading with RAG:
```python
# OLD: prompts = self.prompt_manager.get_system_prompt(...)
# NEW: prompts = self.rag_manager.get_system_prompt_for_task(...)
```

---

## ✅ Testing

```bash
# Test RAG mode
python prompt_driven_agent.py --test-id "NAID-24430" --use-rag

# Test traditional mode (fallback)
python prompt_driven_agent.py --test-id "NAID-24430" --traditional
```

---

**Full integration code examples in the complete framework files.**
