"""
enhanced_prompt_manager.py - Advanced Prompt Management with Embeddings & RAG

This module implements:
- Semantic embeddings for 5000+ line prompts
- Vector database (ChromaDB) for efficient retrieval
- RAG (Retrieval Augmented Generation)
- Meaning-based prompt loading (not keyword-based)
- Chain-of-Thought prompt engineering

The moondream2 lightweight model can now handle massive prompts efficiently!
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import hashlib

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

logger = logging.getLogger(__name__)


class EnhancedPromptManager:
    """
    Advanced prompt manager with semantic retrieval and RAG.
    
    Features:
    - Embeds all prompts into vector database
    - Retrieves relevant prompts by semantic meaning
    - Supports 5000+ lines without overwhelming lightweight models
    - Chain-of-Thought prompt construction
    - One-shot learning examples
    - Dynamic context assembly
    """
    
    def __init__(
        self, 
        prompts_dir: str = "./prompts",
        vector_db_dir: str = "./vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize enhanced prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt markdown files
            vector_db_dir: Directory for vector database storage
            embedding_model: HuggingFace embedding model name
        """
        self.prompts_dir = Path(prompts_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Original prompts storage (for backward compatibility)
        self.prompts = {
            'base': '',
            'error_handling': '',
            'learned_solutions': '',
            'component_specific': {},
            'custom_commands': ''
        }
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter for chunking large prompts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Optimal for lightweight models
            chunk_overlap=50,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", " "]
        )
        
        # Vector database
        self.vector_db = None
        self.db_initialized = False
        
        logger.info(f"Enhanced Prompt Manager initialized")
        logger.info(f"Prompts: {self.prompts_dir}")
        logger.info(f"Vector DB: {self.vector_db_dir}")
    
    def load_and_embed_prompts(self, force_rebuild: bool = False):
        """
        Load all prompts and create/update vector database.
        
        Args:
            force_rebuild: Force rebuild of vector database even if exists
        """
        try:
            # Check if we need to rebuild
            db_path = self.vector_db_dir / "chroma.sqlite3"
            needs_rebuild = force_rebuild or not db_path.exists()
            
            if needs_rebuild:
                logger.info("Building vector database from prompts...")
                self._build_vector_database()
            else:
                logger.info("Loading existing vector database...")
                self._load_vector_database()
            
            # Also load prompts traditionally for backward compatibility
            self._load_prompts_traditional()
            
            self.db_initialized = True
            logger.info("✅ Prompt system ready (Vector DB + Traditional)")
            
        except Exception as e:
            logger.error(f"Failed to initialize prompt system: {e}")
            # Fallback to traditional loading
            self._load_prompts_traditional()
    
    def _build_vector_database(self):
        """Build vector database from all prompt files."""
        documents = []
        
        # Load all markdown files
        prompt_files = {
            'base_prompts.md': 'core_intelligence',
            'error_handling.md': 'error_solutions',
            'learned_solutions.md': 'learned_knowledge',
            'custom_commands.md': 'adb_commands'
        }
        
        # Component-specific prompts
        component_dir = self.prompts_dir / "component_specific"
        if component_dir.exists():
            for file in component_dir.glob("*.md"):
                prompt_files[f"component_specific/{file.name}"] = f"component_{file.stem}"
        
        # Process each file
        for filename, category in prompt_files.items():
            filepath = self.prompts_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Prompt file not found: {filename}")
                continue
            
            # Read content
            content = filepath.read_text(encoding='utf-8')
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': filename,
                        'category': category,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Processed {filename}: {len(chunks)} chunks")
        
        logger.info(f"Total documents for embedding: {len(documents)}")
        
        # Create vector database
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.vector_db_dir),
            collection_name="automotive_prompts"
        )
        
        # Persist to disk
        self.vector_db.persist()
        
        logger.info(f"✅ Vector database created: {len(documents)} chunks embedded")
    
    def _load_vector_database(self):
        """Load existing vector database."""
        self.vector_db = Chroma(
            persist_directory=str(self.vector_db_dir),
            embedding_function=self.embeddings,
            collection_name="automotive_prompts"
        )
        logger.info("✅ Vector database loaded")
    
    def _load_prompts_traditional(self):
        """Load prompts traditionally (backward compatibility)."""
        try:
            self.prompts['base'] = self._load_prompt_file('base_prompts.md')
            self.prompts['error_handling'] = self._load_prompt_file('error_handling.md')
            self.prompts['learned_solutions'] = self._load_prompt_file('learned_solutions.md')
            self.prompts['custom_commands'] = self._load_prompt_file('custom_commands.md')
            
            # Component-specific
            component_dir = self.prompts_dir / "component_specific"
            if component_dir.exists():
                for file in component_dir.glob("*.md"):
                    self.prompts['component_specific'][file.stem] = file.read_text(encoding='utf-8')
            
            logger.info("✅ Traditional prompts loaded")
        except Exception as e:
            logger.error(f"Traditional prompt loading failed: {e}")
    
    def _load_prompt_file(self, filename: str) -> str:
        """Load single prompt file."""
        filepath = self.prompts_dir / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        return ""
    
    def retrieve_relevant_prompts(
        self,
        query: str,
        component: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant prompt chunks using semantic search.
        
        This is the CORE of RAG - retrieves by MEANING, not keywords!
        
        Args:
            query: Query describing what guidance is needed
            component: Optional component filter (hvac, media, navigation)
            top_k: Number of relevant chunks to retrieve
            min_score: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of (chunk_text, relevance_score) tuples
        """
        if not self.db_initialized or self.vector_db is None:
            logger.warning("Vector DB not initialized, using fallback")
            return []
        
        try:
            # Build filter
            filter_dict = {}
            if component:
                filter_dict['category'] = f"component_{component}"
            
            # Semantic search
            results = self.vector_db.similarity_search_with_score(
                query=query,
                k=top_k * 2,  # Get more, filter later
                filter=filter_dict if filter_dict else None
            )
            
            # Filter by minimum score and format
            relevant = []
            for doc, score in results:
                # Chroma returns distance, convert to similarity
                similarity = 1 / (1 + score)
                
                if similarity >= min_score:
                    relevant.append((doc.page_content, similarity))
            
            # Take top_k after filtering
            relevant = relevant[:top_k]
            
            if relevant:
                logger.info(f"📚 Retrieved {len(relevant)} relevant prompt chunks")
                for i, (_, score) in enumerate(relevant[:3]):
                    logger.debug(f"   Chunk {i+1}: relevance {score:.2f}")
            
            return relevant
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def get_system_prompt_with_rag(
        self,
        objective: str,
        component: Optional[str] = None,
        max_context_length: int = 2000
    ) -> str:
        """
        Generate system prompt using RAG.
        
        Instead of loading ALL prompts (5000+ lines), this:
        1. Understands the test objective
        2. Retrieves ONLY relevant knowledge chunks
        3. Assembles compact, focused system prompt
        
        This allows lightweight models to handle massive knowledge bases!
        
        Args:
            objective: Current test objective
            component: Component being tested
            max_context_length: Maximum total context length
            
        Returns:
            Optimized system prompt with relevant context
        """
        prompt_parts = []
        current_length = 0
        
        # 1. Core intelligence (always include, but compressed)
        core_intelligence = self._get_core_intelligence_cot()
        prompt_parts.append(core_intelligence)
        current_length += len(core_intelligence)
        
        # 2. Retrieve relevant guidance for objective
        objective_guidance = self.retrieve_relevant_prompts(
            query=objective,
            component=component,
            top_k=3,
            min_score=0.6
        )
        
        if objective_guidance:
            prompt_parts.append("\n## Relevant Knowledge for Current Task\n")
            for chunk, score in objective_guidance:
                if current_length + len(chunk) > max_context_length:
                    break
                prompt_parts.append(f"\n{chunk}\n")
                current_length += len(chunk)
        
        # 3. Component-specific knowledge (if specified)
        if component:
            component_guidance = self.retrieve_relevant_prompts(
                query=f"How to test {component} component",
                component=component,
                top_k=2,
                min_score=0.5
            )
            
            if component_guidance:
                prompt_parts.append(f"\n## {component.upper()} Component Guidance\n")
                for chunk, score in component_guidance:
                    if current_length + len(chunk) > max_context_length:
                        break
                    prompt_parts.append(f"\n{chunk}\n")
                    current_length += len(chunk)
        
        # 4. Recent error solutions
        error_guidance = self.retrieve_relevant_prompts(
            query="error solutions and troubleshooting",
            top_k=2,
            min_score=0.5
        )
        
        if error_guidance:
            prompt_parts.append("\n## Error Handling\n")
            for chunk, score in error_guidance[:2]:
                if current_length + len(chunk) > max_context_length:
                    break
                prompt_parts.append(f"\n{chunk}\n")
                current_length += len(chunk)
        
        # 5. Current objective
        prompt_parts.append(f"\n## Current Test Objective\n{objective}\n")
        
        final_prompt = "\n".join(prompt_parts)
        
        logger.info(f"📝 Generated RAG prompt: {len(final_prompt)} chars (from {current_length} total)")
        
        return final_prompt
    
    def _get_core_intelligence_cot(self) -> str:
        """
        Get core intelligence in Chain-of-Thought format.
        
        CoT prompting helps lightweight models reason better.
        """
        cot_prompt = """# AI Agent Core Intelligence (Chain-of-Thought Reasoning)

You are an autonomous automotive testing agent. Follow this reasoning process:

## Step 1: UNDERSTAND the Task
- Read test objective carefully
- Identify: What needs to be tested? Which component? What's the expected result?
- Ask yourself: "Do I understand what success looks like?"

## Step 2: ANALYZE Current Screen
- Capture screenshot via ADB
- Run OCR to detect all visible text
- Identify UI elements (buttons, sliders, indicators)
- Determine: "What is currently visible? What state is the system in?"

## Step 3: PLAN the Action
Think through options in priority order:

Priority 1 - OCR Text Detection (Fastest, 99% accuracy):
- "Can I find the target element by text?"
- If yes → Get coordinates via OCR → Tap

Priority 2 - Image Matching (Fast, 95% accuracy):
- "Is this an icon without text?"
- If yes → Match against reference library → Tap

Priority 3 - Vision AI (Slower, 80% accuracy):
- "Did OCR/Image matching fail?"
- If yes → Use moondream2 to locate element → Tap

Priority 4 - Intent-Based (Last resort):
- "After 9 attempts, still failing?"
- If yes → Ask architect for manual help

## Step 4: EXECUTE with Confidence
- Perform chosen action via ADB
- Log what you're doing (speak via TTS if enabled)
- Wait appropriate time for UI response

## Step 5: VERIFY the Result
- Capture new screenshot
- Compare with expected outcome
- Ask: "Did the action achieve the objective?"
- If NO → Retry with different approach
- If YES → Success! Log and continue

## Step 6: LEARN from Outcome
- If succeeded: Note what worked
- If failed after 10 attempts: Ask architect for guidance
- Save architect's solution to learned_solutions.md
- Apply learning to future similar tasks

## Your Capabilities
- Screen capture/analysis (ADB)
- Advanced OCR (EasyOCR/PaddleOCR) 
- Image matching from reference library
- Vision AI (moondream2)
- All gestures: tap, double_tap, long_press, swipe (all directions)
- Voice interface (TTS/STT)

Remember: Think step-by-step, verify each step, learn from outcomes!
"""
        return cot_prompt
    
    def get_one_shot_example(self, action_type: str) -> str:
        """
        Get one-shot learning example for specific action type.
        
        One-shot prompting helps models learn from examples.
        
        Args:
            action_type: Type of action (tap, swipe, verify, etc.)
            
        Returns:
            Example prompt showing how to perform action
        """
        examples = {
            'tap': """
## One-Shot Example: Finding and Tapping Element

Task: "Tap the AC button"

Step-by-Step Reasoning:
1. UNDERSTAND: Need to activate AC in HVAC system
2. ANALYZE: Capture screenshot, run OCR
3. PLAN: AC button likely has text "AC" or snowflake icon
   - Try Priority 1: OCR search for "AC"
4. EXECUTE: 
   - Run: ocr_tap("AC")
   - Found at (540, 300) with 0.95 confidence
   - Action: tap(540, 300)
5. VERIFY: Capture screenshot, check AC indicator is ON
6. RESULT: ✅ Success! AC activated

Your turn: Apply same reasoning to your task.
""",
            'swipe': """
## One-Shot Example: Adjusting Slider with Swipe

Task: "Increase fan speed"

Step-by-Step Reasoning:
1. UNDERSTAND: Need to move fan speed slider up
2. ANALYZE: Screenshot shows horizontal fan slider
3. PLAN: Horizontal slider → swipe right to increase
   - May need slow swipe (800ms) for smooth control
4. EXECUTE:
   - Action: swipe('right', distance=150, speed='slow', duration=800)
5. VERIFY: Check fan speed indicator increased
6. RESULT: ✅ Fan speed went from 3 → 5

Your turn: Apply same reasoning to your task.
""",
            'double_tap': """
## One-Shot Example: PACCAR Media Double-Tap

Task: "Open FM radio source selection"

Step-by-Step Reasoning:
1. UNDERSTAND: PACCAR system needs double-tap on Media icon
2. ANALYZE: Find Media icon in taskbar via OCR
3. PLAN: Single tap opens player, double-tap opens source menu
   - Use double_tap() with 50ms delay
4. EXECUTE:
   - Find: media_coords = ocr_find("Media")
   - Action: double_tap(media_coords[0], media_coords[1], delay=50)
5. VERIFY: Check if FM/AM/SiriusXM menu appeared
6. RESULT: ✅ Source selection menu opened

Your turn: Apply same reasoning to your task.
"""
        }
        
        return examples.get(action_type, "")
    
    def add_learned_solution(self, problem: str, solution: str, added_by: str = "Veera Saravanan"):
        """
        Add learned solution and update vector database.
        
        Args:
            problem: Problem description
            solution: Solution that worked
            added_by: Architect name
        """
        try:
            # Add to traditional file
            filepath = self.prompts_dir / "learned_solutions.md"
            timestamp = datetime.now().strftime("%Y-%m-%d")
            
            entry = f"\n\n## [{timestamp}] {problem}\n\n"
            entry += f"**Problem**: {problem}\n"
            entry += f"**Solution**: {solution}\n"
            entry += f"**Added by**: {added_by}\n"
            entry += "\n---\n"
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(entry)
            
            # Add to vector database
            if self.vector_db:
                doc = Document(
                    page_content=entry,
                    metadata={
                        'source': 'learned_solutions.md',
                        'category': 'learned_knowledge',
                        'timestamp': timestamp,
                        'added_by': added_by
                    }
                )
                
                self.vector_db.add_documents([doc])
                self.vector_db.persist()
            
            logger.info(f"✅ Learned solution added: {problem[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add learned solution: {e}")
            return False
    
    def get_prompt_statistics(self) -> Dict:
        """Get statistics about prompt system."""
        stats = {
            'vector_db_initialized': self.db_initialized,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 512,
            'total_documents': 0
        }
        
        if self.vector_db:
            try:
                collection = self.vector_db._collection
                stats['total_documents'] = collection.count()
            except:
                pass
        
        return stats


# Backward compatibility wrapper
class PromptManager(EnhancedPromptManager):
    """
    Backward compatible PromptManager that uses enhanced features.
    """
    
    def __init__(self, prompts_dir: str = "./prompts"):
        super().__init__(prompts_dir=prompts_dir)
        self.load_all_prompts()
    
    def load_all_prompts(self):
        """Load prompts (uses enhanced system)."""
        self.load_and_embed_prompts()
    
    def get_system_prompt(
        self,
        component: Optional[str] = None,
        objective: str = ""
    ) -> str:
        """Get system prompt (uses RAG if available)."""
        if self.db_initialized and objective:
            return self.get_system_prompt_with_rag(
                objective=objective,
                component=component
            )
        else:
            # Fallback to traditional
            return self._get_traditional_system_prompt(component, objective)
    
    def _get_traditional_system_prompt(self, component, objective):
        """Traditional prompt assembly (fallback)."""
        parts = []
        
        if self.prompts['base']:
            parts.append(self.prompts['base'])
        
        if component and component in self.prompts['component_specific']:
            parts.append(f"\n\n## Component Knowledge\n")
            parts.append(self.prompts['component_specific'][component])
        
        if self.prompts['error_handling']:
            parts.append("\n\n## Error Solutions\n")
            parts.append(self.prompts['error_handling'])
        
        if objective:
            parts.append(f"\n\n## Current Objective\n{objective}")
        
        return "\n".join(parts)
    
    def reload_prompts(self):
        """Reload prompts."""
        self.load_and_embed_prompts(force_rebuild=True)
    
    def get_prompt_summary(self) -> Dict:
        """Get prompt summary."""
        return self.get_prompt_statistics()


if __name__ == "__main__":
    # Test enhanced prompt manager
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Enhanced Prompt Manager Test")
    print("=" * 80)
    
    pm = PromptManager()
    
    # Test RAG retrieval
    print("\n1. Testing semantic retrieval...")
    results = pm.retrieve_relevant_prompts(
        query="How do I find and tap the AC button in HVAC system?",
        top_k=3
    )
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {i} (score: {score:.2f}):")
        print(chunk[:200] + "...")
    
    # Test RAG prompt generation
    print("\n2. Testing RAG prompt generation...")
    prompt = pm.get_system_prompt_with_rag(
        objective="Turn on AC and set temperature to 72°F",
        component="hvac"
    )
    print(f"Generated prompt length: {len(prompt)} characters")
    
    # Show statistics
    print("\n3. Statistics:")
    stats = pm.get_prompt_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Enhanced prompt manager test completed!")
