"""
prompt_manager.py - Dynamic Prompt Management System

Loads, manages, and merges prompt files for the AI agent.
The architect (Veera Saravanan) modifies prompts to improve the agent's intelligence.

NO CODE CHANGES NEEDED - Only edit markdown files in prompts/ folder!
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Try to import embedding system for handling large prompts
try:
    from prompt_embedder import PromptEmbedder
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages system prompts loaded from markdown files.
    
    Features:
    - Load prompts from files dynamically
    - Merge base + learned + component-specific prompts
    - Track prompt versions
    - Reload prompts without restarting agent
    """
    
    def __init__(self, prompts_dir: str = "./prompts", use_embeddings: bool = True):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt markdown files
            use_embeddings: Use embedding-based retrieval for large prompts (default: True)
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {
            'base': '',
            'error_handling': '',
            'learned_solutions': '',
            'component_specific': {},
            'custom_commands': ''
        }
        
        # Ensure prompts directory exists
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        (self.prompts_dir / "component_specific").mkdir(exist_ok=True)
        
        # Initialize embedding system for large prompts
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.embedder = None
        
        if self.use_embeddings:
            try:
                self.embedder = PromptEmbedder(prompts_dir=str(prompts_dir))
                logger.info("✅ Embedding system enabled for large prompt handling")
            except Exception as e:
                logger.warning(f"Embedding system disabled: {e}")
                self.use_embeddings = False
        
        logger.info(f"Prompt Manager initialized: {self.prompts_dir} (embeddings: {self.use_embeddings})")
    
    def load_all_prompts(self):
        """Load all prompt files from the prompts directory."""
        try:
            # Load base prompts
            self.prompts['base'] = self._load_prompt_file('base_prompts.md')
            
            # Load error handling prompts
            self.prompts['error_handling'] = self._load_prompt_file('error_handling.md')
            
            # Load learned solutions
            self.prompts['learned_solutions'] = self._load_prompt_file('learned_solutions.md')
            
            # Load custom commands
            self.prompts['custom_commands'] = self._load_prompt_file('custom_commands.md')
            
            # Load component-specific prompts
            component_dir = self.prompts_dir / "component_specific"
            if component_dir.exists():
                for prompt_file in component_dir.glob("*.md"):
                    component_name = prompt_file.stem
                    self.prompts['component_specific'][component_name] = prompt_file.read_text(encoding='utf-8')
                    logger.info(f"Loaded component prompt: {component_name}")
            
            logger.info("✅ All prompts loaded successfully")
            
            # Build embeddings if enabled
            if self.use_embeddings and self.embedder:
                logger.info("Building embeddings for semantic retrieval...")
                self.embedder.create_embeddings_from_prompts()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return False
    
    def _load_prompt_file(self, filename: str) -> str:
        """Load a single prompt file."""
        filepath = self.prompts_dir / filename
        
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            logger.debug(f"Loaded: {filename} ({len(content)} chars)")
            return content
        else:
            logger.warning(f"Prompt file not found: {filename}")
            return ""
    
    def get_system_prompt(self, component: Optional[str] = None, objective: str = "") -> str:
        """
        Generate complete system prompt by merging:
        - Base prompts
        - Error handling knowledge
        - Learned solutions
        - Component-specific prompts (if specified)
        - Custom commands
        
        If embeddings enabled: Returns ONLY relevant portions based on objective!
        If embeddings disabled: Returns full prompts (may be large!)
        
        Args:
            component: Optional component name (e.g., 'hvac', 'media')
            objective: Current test objective
            
        Returns:
            Merged system prompt string (optimized for lightweight models!)
        """
        # If embeddings enabled and we have an objective, use semantic retrieval
        if self.use_embeddings and self.embedder and objective:
            logger.info(f"📊 Using embedding-based retrieval for objective: '{objective[:50]}...'")
            
            # Get relevant prompts using semantic search
            relevant_prompts = self.embedder.get_relevant_prompts(
                query=objective,
                component=component
            )
            
            # Add objective
            final_prompt = f"# Current Test Objective\n{objective}\n\n"
            final_prompt += "# Relevant Knowledge (Retrieved via Embeddings)\n\n"
            final_prompt += relevant_prompts
            
            logger.info(f"✅ Generated optimized prompt: {len(final_prompt)} chars (vs {self._get_full_prompt_size()} full)")
            return final_prompt
        
        # Fallback: Return full prompts (traditional method)
        logger.debug("Using full prompts (embeddings disabled or no objective)")
        prompt_parts = []
        
        # Add base prompts
        if self.prompts['base']:
            prompt_parts.append(self.prompts['base'])
        
        # Add component-specific knowledge
        if component and component.lower() in self.prompts['component_specific']:
            prompt_parts.append("\n\n## Component-Specific Knowledge\n")
            prompt_parts.append(self.prompts['component_specific'][component.lower()])
        
        # Add error handling knowledge
        if self.prompts['error_handling']:
            prompt_parts.append("\n\n## Known Error Solutions\n")
            prompt_parts.append(self.prompts['error_handling'])
        
        # Add learned solutions
        if self.prompts['learned_solutions']:
            prompt_parts.append("\n\n## Learned Solutions from Previous Runs\n")
            prompt_parts.append(self.prompts['learned_solutions'])
        
        # Add custom commands
        if self.prompts['custom_commands']:
            prompt_parts.append("\n\n## Custom ADB Commands\n")
            prompt_parts.append(self.prompts['custom_commands'])
        
        # Add current objective
        if objective:
            prompt_parts.append(f"\n\n## Current Test Objective\n{objective}")
        
        return "\n".join(prompt_parts)
    
    def _get_full_prompt_size(self) -> int:
        """Calculate total size of all prompts combined."""
        total = len(self.prompts['base'])
        total += len(self.prompts['error_handling'])
        total += len(self.prompts['learned_solutions'])
        total += len(self.prompts['custom_commands'])
        for comp_prompt in self.prompts['component_specific'].values():
            total += len(comp_prompt)
        return total
    
    def add_learned_solution(self, problem: str, solution: str, added_by: str = "Veera Saravanan"):
        """
        Add a new learned solution to the learned_solutions.md file.
        
        Args:
            problem: Description of the problem
            solution: Solution that worked
            added_by: Name of architect who provided solution
        """
        try:
            filepath = self.prompts_dir / "learned_solutions.md"
            
            # Create entry
            timestamp = datetime.now().strftime("%Y-%m-%d")
            entry = f"\n\n## [{timestamp}] New Solution\n\n"
            entry += f"**Problem**: {problem}\n"
            entry += f"**Solution**: {solution}\n"
            entry += f"**Added by**: {added_by}\n"
            entry += "\n---\n"
            
            # Append to file
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(entry)
            
            # Reload prompts
            self.prompts['learned_solutions'] = filepath.read_text(encoding='utf-8')
            
            logger.info(f"✅ Added learned solution: {problem[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add learned solution: {e}")
            return False
    
    def reload_prompts(self):
        """Reload all prompts from files (useful for hot-reloading during agent run)."""
        logger.info("🔄 Reloading prompts...")
        return self.load_all_prompts()
    
    def get_prompt_summary(self) -> Dict:
        """Get summary of loaded prompts for debugging."""
        return {
            'base_loaded': len(self.prompts['base']) > 0,
            'error_handling_loaded': len(self.prompts['error_handling']) > 0,
            'learned_solutions_loaded': len(self.prompts['learned_solutions']) > 0,
            'components_loaded': list(self.prompts['component_specific'].keys()),
            'custom_commands_loaded': len(self.prompts['custom_commands']) > 0
        }


if __name__ == "__main__":
    # Test the prompt manager
    logging.basicConfig(level=logging.INFO)
    
    pm = PromptManager()
    pm.load_all_prompts()
    
    print("\n" + "="*60)
    print("Prompt Manager Test")
    print("="*60)
    print(pm.get_prompt_summary())
    
    # Test system prompt generation
    prompt = pm.get_system_prompt(component='hvac', objective='Test AC button')
    print(f"\nGenerated prompt length: {len(prompt)} characters")
