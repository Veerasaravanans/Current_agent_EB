"""
rag_prompt_manager.py - RAG-Enhanced Prompt Management

Uses Retrieval Augmented Generation (RAG) to:
- Load only RELEVANT prompts for current task (not all 5000+ lines)
- Semantic retrieval by meaning
- Real-time context assembly
- Dynamic prompt optimization

This replaces traditional prompt_manager.py with intelligent RAG approach.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from prompt_embeddings import PromptEmbeddingsManager

logger = logging.getLogger(__name__)


class RAGPromptManager:
    """
    RAG-based prompt manager with semantic retrieval.
    
    Features:
    - Semantic search: Find prompts by meaning
    - Context-aware: Assemble only relevant prompts
    - Memory efficient: Don't load all prompts
    - Dynamic: Adapts to current task
    - Learning: Auto-updates when prompts change
    """
    
    def __init__(self, prompts_dir: str = "./prompts"):
        """
        Initialize RAG prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt markdown files
        """
        self.prompts_dir = Path(prompts_dir)
        
        # Initialize embeddings manager
        logger.info("Initializing RAG Prompt Manager...")
        self.embeddings = PromptEmbeddingsManager(prompts_dir=str(self.prompts_dir))
        
        # Ensure prompts are embedded
        self._ensure_prompts_embedded()
        
        logger.info("✅ RAG Prompt Manager initialized")
    
    def _ensure_prompts_embedded(self):
        """Ensure all prompts are embedded (run once or when changed)."""
        stats = self.embeddings.get_statistics()
        
        if stats.get('total_chunks', 0) == 0:
            logger.info("No embeddings found. Embedding all prompts...")
            self.embeddings.embed_all_prompts()
        else:
            logger.info(f"Using existing embeddings: {stats.get('total_chunks')} chunks")
            # Check for new/updated files
            self.embeddings.embed_all_prompts(force=False)
    
    def get_system_prompt_for_task(
        self,
        task_description: str,
        component: Optional[str] = None,
        include_base: bool = True,
        max_context_size: int = 3000
    ) -> str:
        """
        Get optimized system prompt for specific task using RAG.
        
        This is the CORE method: Retrieves only relevant prompts!
        
        Args:
            task_description: What the agent needs to do
            component: Optional component (hvac, media, navigation)
            include_base: Include base prompt rules
            max_context_size: Maximum context characters
            
        Returns:
            Optimized system prompt with only relevant context
        """
        prompt_parts = []
        remaining_size = max_context_size
        
        # 1. Base prompt (always include core rules)
        if include_base:
            base_context = self.embeddings.get_context_for_task(
                "core rules decision making process capabilities",
                max_chunks=3
            )
            if base_context:
                prompt_parts.append("# Core Agent Rules\n\n")
                prompt_parts.append(base_context[:1000])  # Limit base
                remaining_size -= len(base_context[:1000])
        
        # 2. Task-specific context (semantic search)
        logger.info(f"🔍 RAG: Retrieving context for task: {task_description}")
        
        task_context = self.embeddings.get_context_for_task(
            task_description,
            component=component,
            max_chunks=8
        )
        
        if task_context:
            # Limit to remaining size
            task_context = task_context[:remaining_size]
            prompt_parts.append("\n\n# Task-Specific Knowledge\n\n")
            prompt_parts.append(task_context)
            remaining_size -= len(task_context)
        
        # 3. Error handling (only if space remains)
        if remaining_size > 500 and "error" in task_description.lower():
            error_context = self.embeddings.get_error_solutions(task_description, n_results=2)
            if error_context:
                prompt_parts.append("\n\n# Relevant Error Solutions\n\n")
                for solution in error_context:
                    text = solution['text'][:300]  # Short snippets
                    prompt_parts.append(f"{text}\n\n")
        
        # Combine
        final_prompt = "".join(prompt_parts)
        
        logger.info(f"✅ RAG: Assembled prompt - {len(final_prompt)} chars (vs {max_context_size} budget)")
        
        return final_prompt
    
    def get_error_solution(self, error_description: str) -> str:
        """
        Get solution for specific error using semantic search.
        
        Args:
            error_description: Description of the error
            
        Returns:
            Relevant solution text
        """
        logger.info(f"🔍 RAG: Searching for solution to: {error_description[:50]}...")
        
        solutions = self.embeddings.get_error_solutions(error_description, n_results=3)
        
        if not solutions:
            logger.warning("No solutions found")
            return ""
        
        # Format solutions
        solution_text = "# Relevant Solutions\n\n"
        
        for i, solution in enumerate(solutions, 1):
            solution_text += f"## Solution {i}\n\n"
            solution_text += f"{solution['text']}\n\n"
            solution_text += "---\n\n"
        
        return solution_text
    
    def get_component_knowledge(self, component: str, specific_task: str = "") -> str:
        """
        Get component-specific knowledge (HVAC, Media, Navigation).
        
        Args:
            component: Component name
            specific_task: Optional specific task
            
        Returns:
            Relevant component knowledge
        """
        logger.info(f"🔍 RAG: Loading {component} knowledge...")
        
        query = f"{component} {specific_task}" if specific_task else component
        
        knowledge = self.embeddings.get_component_knowledge(component, specific_task)
        
        if not knowledge:
            logger.warning(f"No knowledge found for {component}")
            return ""
        
        return knowledge
    
    def add_learned_solution(
        self,
        problem: str,
        solution: str,
        added_by: str = "Veera Saravanan"
    ):
        """
        Add new learned solution and re-embed.
        
        Args:
            problem: Description of problem
            solution: Solution that worked
            added_by: Architect name
        """
        try:
            filepath = self.prompts_dir / "learned_solutions.md"
            
            # Create entry
            timestamp = datetime.now().strftime("%Y-%m-%d")
            entry = f"\n\n## [{timestamp}] {problem}\n\n"
            entry += f"**Problem**: {problem}\n"
            entry += f"**Solution**: {solution}\n"
            entry += f"**Added by**: {added_by}\n"
            entry += "\n---\n"
            
            # Append to file
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(entry)
            
            # Re-embed this file
            logger.info("Re-embedding learned_solutions.md...")
            self.embeddings.embed_prompt_file(filepath, force=True)
            
            logger.info(f"✅ Added learned solution: {problem[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add learned solution: {e}")
            return False
    
    def reload_prompts(self):
        """Reload prompts from files and re-embed if changed."""
        logger.info("🔄 Reloading prompts...")
        results = self.embeddings.embed_all_prompts(force=False)
        
        updated_count = sum(1 for count in results.values() if count > 0)
        logger.info(f"✅ Reloaded: {updated_count} files updated")
    
    def rebuild_index(self):
        """Rebuild entire vector index (use if prompts corrupted)."""
        logger.info("🔄 Rebuilding vector index...")
        self.embeddings.rebuild_index()
        logger.info("✅ Index rebuilt")
    
    def get_prompt_summary(self) -> Dict:
        """Get summary of embeddings database."""
        stats = self.embeddings.get_statistics()
        
        return {
            'rag_enabled': True,
            'vector_database': 'ChromaDB',
            'embedding_model': 'all-MiniLM-L6-v2',
            'total_chunks': stats.get('total_chunks', 0),
            'files_indexed': stats.get('files_indexed', 0),
            'db_path': stats.get('db_path', ''),
            'semantic_search': True,
            'max_context_size': '3000 chars per task (dynamic)'
        }
    
    def search_prompts(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search prompts by natural language query.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            List of relevant chunks
        """
        return self.embeddings.semantic_search(query, n_results=n_results)
    
    def get_context_for_action(self, action_type: str, target: str) -> str:
        """
        Get context for specific action.
        
        Args:
            action_type: Type of action (tap, swipe, verify, etc.)
            target: Target element (AC, Media, Temperature, etc.)
            
        Returns:
            Relevant context string
        """
        query = f"How to {action_type} {target}"
        
        logger.info(f"🔍 RAG: Getting context for action: {query}")
        
        return self.embeddings.get_context_for_task(query, max_chunks=5)


# ==================================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# These provide compatibility with old prompt_manager.py API
# ==================================================================================

# Global RAG manager instance
_rag_manager = None


def get_rag_manager() -> RAGPromptManager:
    """Get or create global RAG manager."""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGPromptManager()
    return _rag_manager


def get_system_prompt(component: Optional[str] = None, objective: str = "") -> str:
    """
    Backward compatible: Get system prompt.
    
    NOW USES RAG instead of loading all prompts!
    """
    manager = get_rag_manager()
    
    # Build task description from objective
    task_description = objective if objective else "general testing task"
    
    return manager.get_system_prompt_for_task(
        task_description=task_description,
        component=component,
        include_base=True
    )


def add_learned_solution(problem: str, solution: str, added_by: str = "Veera Saravanan"):
    """Backward compatible: Add learned solution."""
    manager = get_rag_manager()
    return manager.add_learned_solution(problem, solution, added_by)


def reload_prompts():
    """Backward compatible: Reload prompts."""
    manager = get_rag_manager()
    manager.reload_prompts()


def get_prompt_summary() -> Dict:
    """Backward compatible: Get prompt summary."""
    manager = get_rag_manager()
    return manager.get_prompt_summary()


def get_component_prompt(component: str) -> str:
    """Backward compatible: Get component-specific prompt."""
    manager = get_rag_manager()
    return manager.get_component_knowledge(component)


def main():
    """Test RAG prompt manager."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing RAG Prompt Manager")
    print("=" * 80)
    
    # Initialize
    manager = RAGPromptManager()
    
    # Test 1: Task-specific prompt
    print("\n1. Testing task-specific prompt retrieval...")
    prompt = manager.get_system_prompt_for_task(
        task_description="Turn on AC and set temperature to 72 degrees",
        component="hvac"
    )
    print(f"   Retrieved prompt: {len(prompt)} characters")
    print(f"   Preview: {prompt[:200]}...")
    
    # Test 2: Error solution
    print("\n2. Testing error solution retrieval...")
    solution = manager.get_error_solution("Temperature slider not responding to swipe")
    print(f"   Found solution: {len(solution)} characters")
    print(f"   Preview: {solution[:200]}...")
    
    # Test 3: Component knowledge
    print("\n3. Testing component knowledge retrieval...")
    hvac_knowledge = manager.get_component_knowledge("hvac", "fan speed control")
    print(f"   HVAC knowledge: {len(hvac_knowledge)} characters")
    
    # Test 4: Semantic search
    print("\n4. Testing semantic search...")
    results = manager.search_prompts("PACCAR double tap media", n_results=3)
    print(f"   Found {len(results)} relevant chunks:")
    for i, result in enumerate(results, 1):
        source = result['metadata']['source_file']
        preview = result['text'][:100].replace('\n', ' ')
        print(f"     {i}. [{source}] {preview}...")
    
    # Test 5: Summary
    print("\n5. System summary:")
    summary = manager.get_prompt_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ RAG Prompt Manager test complete!")


if __name__ == "__main__":
    main()
