# 📊 System Architecture & Workflow Visualization

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                USER INTERFACE                               │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │  Command Line  │  │   GUI (PyQt6)    │  │  Voice Interface (TTS/STT) │ │
│  │  --test-id     │  │  Visual Control  │  │  Speak & Listen           │ │
│  └────────┬───────┘  └────────┬─────────┘  └────────┬───────────────────┘ │
└───────────┴──────────────────┴─────────────────────┴──────────────────────┘
            │                   │                     │
            └───────────────────┴─────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROMPT EMBEDDINGS SYSTEM                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Sentence Transformers (all-MiniLM-L6-v2)                            │  │
│  │  • Converts text → 384-dimensional vectors                           │  │
│  │  • Fast: <100ms per query                                           │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  ChromaDB Vector Database                                            │  │
│  │  • 5000+ lines of prompts → ~100 chunks                             │  │
│  │  • Each chunk: 500 chars with 50-char overlap                       │  │
│  │  • Persistent storage in ./vector_db/                               │  │
│  │  • Semantic similarity search by meaning                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG PROMPT MANAGER                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Retrieval Augmented Generation Flow:                                │  │
│  │                                                                       │  │
│  │  1. Receive task: "Turn on AC and set to 72°F"                      │  │
│  │     ↓                                                                │  │
│  │  2. Embed task query (384-dim vector)                               │  │
│  │     ↓                                                                │  │
│  │  3. Cosine similarity search in ChromaDB                            │  │
│  │     ↓                                                                │  │
│  │  4. Retrieve top 5-10 most relevant chunks                          │  │
│  │     • HVAC control methods                                          │  │
│  │     • AC button detection                                           │  │
│  │     • Temperature adjustment                                        │  │
│  │     • Error handling for HVAC                                       │  │
│  │     ↓                                                                │  │
│  │  5. Assemble focused context (~500-600 lines vs 5000+ lines)       │  │
│  │     ↓                                                                │  │
│  │  6. Pass to LangChain                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LANGCHAIN COORDINATOR                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Chain-of-Thought (CoT) Reasoning:                                   │  │
│  │                                                                       │  │
│  │  Input: Task + Relevant Context (600 lines)                         │  │
│  │     ↓                                                                │  │
│  │  CoT Prompt Template:                                               │  │
│  │  "Think step-by-step to solve this task:                           │  │
│  │   1. Understand the objective                                       │  │
│  │   2. Identify UI elements needed                                    │  │
│  │   3. Plan action sequence                                           │  │
│  │   4. Consider error scenarios                                       │  │
│  │   5. Generate structured plan"                                      │  │
│  │     ↓                                                                │  │
│  │  Moondream2 LLM Processing (via Ollama)                            │  │
│  │     ↓                                                                │  │
│  │  Structured Output (Pydantic Model):                               │  │
│  │  {                                                                   │  │
│  │    "objective": "Turn on AC and set to 72°F",                      │  │
│  │    "thought_process": "1. Need to detect HVAC type...              │  │
│  │                        2. Find AC button via OCR...                 │  │
│  │                        3. Verify AC turns on...                     │  │
│  │                        4. Adjust temperature...",                   │  │
│  │    "steps": [                                                        │  │
│  │      {                                                               │  │
│  │        "action": "tap",                                             │  │
│  │        "target": "AC",                                              │  │
│  │        "method": "ocr",                                             │  │
│  │        "reasoning": "AC text visible, OCR fastest",                │  │
│  │        "expected": "AC indicator turns blue"                        │  │
│  │      },                                                              │  │
│  │      {...more steps...}                                             │  │
│  │    ],                                                                │  │
│  │    "component": "hvac",                                             │  │
│  │    "difficulty": "medium"                                           │  │
│  │  }                                                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH WORKFLOW MANAGER                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  State Machine Execution:                                            │  │
│  │                                                                       │  │
│  │  ┌─────────┐      ┌─────────────┐      ┌──────────┐                │  │
│  │  │  PLAN   │─────→│   EXECUTE   │─────→│  VERIFY  │                │  │
│  │  │         │      │   Step 1    │      │  Result  │                │  │
│  │  └─────────┘      └──────┬──────┘      └─────┬────┘                │  │
│  │                           │                    │                      │  │
│  │                           │                    ├───→ Success → Next  │  │
│  │                           │                    │                      │  │
│  │                  ┌────────▼────────┐          ├───→ Failure → Retry │  │
│  │                  │  HANDLE ERROR   │◄─────────┘                      │  │
│  │                  │                 │                                  │  │
│  │                  │ • Retry 1-3:OCR │                                  │  │
│  │                  │ • Retry 4-6:IMG │                                  │  │
│  │                  │ • Retry 7-9:AI  │                                  │  │
│  │                  │ • Retry 10:HELP │                                  │  │
│  │                  └────────┬────────┘                                  │  │
│  │                           │                                           │  │
│  │           ┌───────────────┼───────────────┐                          │  │
│  │           │               │               │                          │  │
│  │      Retry≤10      Human Help      Max Retries                      │  │
│  │           │               │               │                          │  │
│  │           ▼               ▼               ▼                          │  │
│  │      EXECUTE    REQUEST_HUMAN_HELP    FINALIZE                      │  │
│  │                                           │                          │  │
│  │                                           ▼                          │  │
│  │                                      ┌─────────┐                     │  │
│  │                                      │   END   │                     │  │
│  │                                      │ (Report)│                     │  │
│  │                                      └─────────┘                     │  │
│  │                                                                       │  │
│  │  Features:                                                           │  │
│  │  • Checkpointing: Save state, resume later                          │  │
│  │  • Branching: Different paths for success/failure                   │  │
│  │  • Memory: Track all previous attempts                              │  │
│  │  • Human-in-Loop: Request help when stuck                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VISION COORDINATOR                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  10-Attempt Intelligent Retry Strategy:                             │  │
│  │                                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Attempt 1-3: OCR (EasyOCR + PaddleOCR)                         │ │  │
│  │  │ • Find text: "AC", "Temperature", "Fan"                        │ │  │
│  │  │ • 99% accuracy                                                  │ │  │
│  │  │ • <500ms per attempt                                           │ │  │
│  │  │ • Variations: case-insensitive, partial match, synonyms       │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │             ↓ (if failed)                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Attempt 4-6: Image Matching                                    │ │  │
│  │  │ • Load reference icon from ./reference_icons/                  │ │  │
│  │  │ • Multi-scale template matching (80%-120%)                    │ │  │
│  │  │ • Rotation-invariant matching                                  │ │  │
│  │  │ • Confidence threshold: 0.85                                   │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │             ↓ (if failed)                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Attempt 7-9: Vision AI (Moondream2)                           │ │  │
│  │  │ • Semantic understanding of screen                             │ │  │
│  │  │ • RAG: Retrieve relevant error solutions                       │ │  │
│  │  │ • LangChain: Analyze with CoT                                  │ │  │
│  │  │ • Estimate element location by description                     │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │             ↓ (if still failed)                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Attempt 10: Human-in-the-Loop                                  │ │  │
│  │  │ • Voice: "Cannot find AC after 9 attempts, please help"       │ │  │
│  │  │ • GUI: Show screenshot, request guidance                       │ │  │
│  │  │ • Architect provides solution                                  │ │  │
│  │  │ • Solution saved to learned_solutions.md                       │ │  │
│  │  │ • Re-embed into ChromaDB                                       │ │  │
│  │  │ • Future tests auto-use this solution                          │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  AUTOMOTIVE OPERATING SYSTEM (ADB)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Device Control via ADB:                                             │  │
│  │                                                                       │  │
│  │  • Screenshot: adb exec-out screencap -p                            │  │
│  │  • Tap: adb shell input tap X Y                                     │  │
│  │  • Swipe: adb shell input swipe X1 Y1 X2 Y2 DURATION               │  │
│  │  • Double-tap: Two taps with 50ms delay                             │  │
│  │  • Long-press: adb shell input swipe X Y X Y 1000                   │  │
│  │  • Text input: adb shell input text "Hello"                         │  │
│  │  • Key events: adb shell input keyevent KEYCODE_BACK                │  │
│  │                                                                       │  │
│  │  Advanced Features:                                                  │  │
│  │  • Multi-directional swipes (up/down/left/right/diagonal/curved)   │  │
│  │  • Gesture speed control (slow/normal/fast)                         │  │
│  │  • Intent-based app launching (package/activity)                    │  │
│  │  • Screen dimension detection                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANDROID AUTOMOTIVE DEVICE                           │
│                    (Physical Device or Emulator)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  • HVAC Controls: AC, Temperature, Fan, Auto                         │  │
│  │  • Media: FM/AM/SiriusXM/Bluetooth, Play/Pause, Volume              │  │
│  │  • Navigation: Maps, Search, POI, Routes                            │  │
│  │  • Settings: System settings, preferences                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LEARNING & REPORTING                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Auto-Learning System:                                               │  │
│  │  1. When human provides solution                                     │  │
│  │     ↓                                                                │  │
│  │  2. Append to learned_solutions.md                                   │  │
│  │     ↓                                                                │  │
│  │  3. Re-embed into ChromaDB                                           │  │
│  │     ↓                                                                │  │
│  │  4. Future tests retrieve this solution automatically               │  │
│  │                                                                       │  │
│  │  Report Generation:                                                  │  │
│  │  • Excel format (industry standard)                                  │  │
│  │  • Pass/Fail verdicts per step                                       │  │
│  │  • Screenshots as evidence                                           │  │
│  │  • Execution time tracking                                           │  │
│  │  • Jira defect linking                                               │  │
│  │  • Summary statistics                                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Example: "Turn on AC"

```
1. USER INPUT
   "Turn on AC"
      ↓

2. RAG EMBEDDING
   Query → [0.23, -0.45, 0.67, ...] (384-dim vector)
      ↓

3. SEMANTIC SEARCH
   ChromaDB finds similar vectors:
   • "AC button detection" (similarity: 0.92)
   • "HVAC control methods" (similarity: 0.89)
   • "OCR text finding" (similarity: 0.86)
   • "AC troubleshooting" (similarity: 0.84)
      ↓

4. CONTEXT ASSEMBLY
   Retrieved chunks assembled → 600 lines
   (Instead of 5000+ lines!)
      ↓

5. LANGCHAIN CoT REASONING
   Input: "Turn on AC" + 600 lines context
   LLM thinks:
   "1. AC is HVAC component
    2. Need to find AC button
    3. AC button likely has text 'AC' or snowflake icon
    4. OCR is fastest for text detection
    5. After tap, verify AC indicator turns on"
      ↓

6. STRUCTURED PLAN
   {
     "steps": [
       {"action": "tap", "target": "AC", "method": "ocr"},
       {"action": "verify", "target": "AC indicator"}
     ]
   }
      ↓

7. LANGGRAPH EXECUTION
   State: PLAN → EXECUTE → VERIFY
      ↓

8. VISION COORDINATOR
   Attempt 1: EasyOCR finds "AC" at (540, 300)
   ✅ Success!
      ↓

9. ADB EXECUTION
   adb shell input tap 540 300
      ↓

10. VERIFICATION
    Capture screenshot
    LangChain verifies: "AC indicator is blue" ✅
       ↓

11. REPORT
    Step 1: PASS (0.5s)
    Overall: PASS (1.2s)
```

---

## 📊 Performance Comparison

### **Traditional Approach (No RAG)**
```
Load ALL prompts → 5000 lines → 10MB text
   ↓
Pass to Moondream2
   ↓
Processing time: 8-10 seconds
   ↓
Context overflow possible
   ↓
Irrelevant info confuses model
```

### **RAG Approach (With Embeddings)**
```
Semantic search → Relevant 600 lines → 1.2MB text
   ↓
Pass to Moondream2
   ↓
Processing time: 1-2 seconds
   ↓
Focused context
   ↓
Better decision quality
```

**Improvement**: 5x faster, better accuracy, infinite scalability

---

## 🎯 Testing Workflow Stages

```
┌──────────────┐
│ TEST DESIGN  │  Excel files in knowledge_base/ folder
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ TEST LOADING │  Load test case by Test ID
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ RAG RETRIEVAL│  Semantic search for relevant prompts
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ PLANNING     │  LangChain Chain-of-Thought
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ EXECUTION    │  LangGraph workflow with retry
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ VERIFICATION │  LangChain structured validation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LEARNING     │  Save solutions to ChromaDB
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ REPORTING    │  Generate Excel report
└──────────────┘
```

---

**Created by: Veera Saravanan**

**Framework: Neural AI Agent with RAG, LangChain, and LangGraph**
