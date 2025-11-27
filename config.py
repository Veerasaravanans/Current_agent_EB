"""
config.py - Configuration Settings for Enhanced AI Agent

Centralized configuration for RAG, LangChain, and all components.
Edit this file to customize behavior without touching code.
"""

from typing import Dict, Any
from pathlib import Path


class AgentConfig:
    """Configuration for the AI Agent."""
    
    # ==================================================================================
    # RAG CONFIGURATION
    # ==================================================================================
    
    RAG_ENABLED = True  # Set to False to use traditional mode
    
    RAG_SETTINGS = {
        # Embedding model (from Sentence Transformers)
        'embedding_model': 'all-MiniLM-L6-v2',  # Fast, 384 dimensions
        # Alternative: 'all-mpnet-base-v2'  # Slower but more accurate, 768 dimensions
        
        # Vector database path
        'vector_db_path': './vector_db',
        
        # Maximum context size to retrieve (in characters)
        'max_context_size': 3000,
        
        # Chunk settings for prompt embedding
        'chunk_size': 500,  # Characters per chunk
        'chunk_overlap': 50,  # Overlap between chunks
        
        # Number of relevant chunks to retrieve
        'n_results': 10,
        
        # Auto-reload prompts on file changes
        'auto_reload': True
    }
    
    # ==================================================================================
    # LANGCHAIN CONFIGURATION
    # ==================================================================================
    
    LANGCHAIN_ENABLED = True
    
    LANGCHAIN_SETTINGS = {
        # Ollama model for LangChain
        'model': 'llava:7b',
        
        # Temperature (0.0 = deterministic, 1.0 = creative)
        'temperature': 0.3,
        
        # Enable conversation memory
        'enable_memory': True,
        
        # Max retries for LLM calls
        'max_retries': 3,
        
        # Timeout for LLM calls (seconds)
        'timeout': 30
    }
    
    # ==================================================================================
    # LANGGRAPH CONFIGURATION
    # ==================================================================================
    
    LANGGRAPH_ENABLED = True
    
    LANGGRAPH_SETTINGS = {
        # Enable workflow checkpointing
        'checkpoint_enabled': True,
        
        # Maximum workflow steps
        'max_workflow_steps': 20,
        
        # Workflow timeout (seconds)
        'workflow_timeout': 300
    }
    
    # ==================================================================================
    # VISION CONFIGURATION
    # ==================================================================================
    
    VISION_SETTINGS = {
        # OCR engines (priority order)
        'ocr_engines': ['easyocr', 'paddleocr'],
        
        # Image matching confidence threshold
        'image_match_threshold': 0.85,
        
        # Reference icons directory
        'reference_icons_dir': './reference_icons',
        
        # Screenshot compression quality (1-100)
        'screenshot_quality': 85,
        
        # Screenshot max size for AI processing
        'screenshot_max_size': (1280, 720)
    }
    
    # ==================================================================================
    # RETRY STRATEGY
    # ==================================================================================
    
    RETRY_SETTINGS = {
        # Maximum retry attempts
        'max_retries': 10,
        
        # Retry strategy distribution (NEW PRIORITY ORDER)
        'vision_ai_attempts': 3,  # Attempts 1-3 (Priority 1: Most flexible)
        'ocr_attempts': 3,        # Attempts 4-6 (Priority 2: Fast for text)
        'image_attempts': 3,      # Attempts 7-9 (Priority 3: For icons)
        'human_help_attempt': 1,  # Attempt 10
        
        # Delay between retries (seconds)
        'retry_delay': 0.5,
        
        # Timeout for each attempt (seconds)
        'attempt_timeout': 10
    }
    
    # ==================================================================================
    # VOICE INTERFACE
    # ==================================================================================
    
    VOICE_ENABLED = False  # Enable voice narration and listening
    
    VOICE_SETTINGS = {
        # TTS (Text-to-Speech)
        'tts_enabled': True,
        'tts_rate': 150,  # Words per minute
        'tts_volume': 0.9,  # 0.0 to 1.0
        
        # STT (Speech-to-Text)
        'stt_enabled': True,
        'stt_timeout': 5,  # Seconds to wait for speech
        'stt_phrase_limit': 10  # Max seconds per phrase
    }
    
    # ==================================================================================
    # KNOWLEDGE BASE
    # ==================================================================================
    
    KNOWLEDGE_BASE_SETTINGS = {
        # Directory containing Excel test files
        'directory': './knowledge_base',
        
        # Auto-load test cases on startup
        'auto_load': True,
        
        # Supported file formats
        'supported_formats': ['.xlsx', '.xls', '.csv']
    }
    
    # ==================================================================================
    # REPORTING
    # ==================================================================================
    
    REPORT_SETTINGS = {
        # Reports output directory
        'output_dir': './test_reports',
        
        # Report format
        'format': 'excel',  # 'excel' or 'json'
        
        # Include screenshots in reports
        'include_screenshots': True,
        
        # Auto-open report after generation
        'auto_open': False
    }
    
    # ==================================================================================
    # ADB / DEVICE CONTROL
    # ==================================================================================
    
    DEVICE_SETTINGS = {
        # Default device serial (None = auto-detect)
        'default_serial': None,
        
        # ADB command timeout (seconds)
        'adb_timeout': 10,
        
        # Wait time after actions (seconds)
        'action_delay': 1.0,
        
        # Screenshot capture method
        'screenshot_method': 'exec-out',  # 'exec-out' or 'pull'
    }
    
    # ==================================================================================
    # LOGGING
    # ==================================================================================
    
    LOGGING_SETTINGS = {
        # Log level: DEBUG, INFO, WARNING, ERROR
        'level': 'INFO',
        
        # Log to file
        'log_to_file': True,
        'log_file': './logs/agent.log',
        
        # Log format
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    # ==================================================================================
    # PROMPTS
    # ==================================================================================
    
    PROMPT_SETTINGS = {
        # Prompts directory
        'prompts_dir': './prompts',
        
        # Component-specific prompts subdirectory
        'component_dir': 'component_specific',
        
        # Auto-reload prompts on change
        'auto_reload': True
    }
    
    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as dictionary."""
        return {
            'rag': {
                'enabled': cls.RAG_ENABLED,
                'settings': cls.RAG_SETTINGS
            },
            'langchain': {
                'enabled': cls.LANGCHAIN_ENABLED,
                'settings': cls.LANGCHAIN_SETTINGS
            },
            'langgraph': {
                'enabled': cls.LANGGRAPH_ENABLED,
                'settings': cls.LANGGRAPH_SETTINGS
            },
            'vision': cls.VISION_SETTINGS,
            'retry': cls.RETRY_SETTINGS,
            'voice': {
                'enabled': cls.VOICE_ENABLED,
                'settings': cls.VOICE_SETTINGS
            },
            'knowledge_base': cls.KNOWLEDGE_BASE_SETTINGS,
            'reporting': cls.REPORT_SETTINGS,
            'device': cls.DEVICE_SETTINGS,
            'logging': cls.LOGGING_SETTINGS,
            'prompts': cls.PROMPT_SETTINGS
        }
    
    @classmethod
    def is_rag_mode(cls) -> bool:
        """Check if RAG mode is enabled."""
        return cls.RAG_ENABLED and cls.LANGCHAIN_ENABLED
    
    @classmethod
    def get_mode_description(cls) -> str:
        """Get current mode description."""
        if cls.is_rag_mode():
            return "RAG + LangChain (Enhanced)"
        else:
            return "Traditional (Fallback)"
    
    @classmethod
    def create_directories(cls):
        """Create required directories if they don't exist."""
        directories = [
            cls.RAG_SETTINGS['vector_db_path'],
            cls.VISION_SETTINGS['reference_icons_dir'],
            cls.KNOWLEDGE_BASE_SETTINGS['directory'],
            cls.REPORT_SETTINGS['output_dir'],
            cls.PROMPT_SETTINGS['prompts_dir'],
            Path(cls.LOGGING_SETTINGS['log_file']).parent,
            './screenshots'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# ==================================================================================
# PRESETS
# ==================================================================================

class Presets:
    """Pre-configured settings for different use cases."""
    
    @staticmethod
    def fast_mode():
        """Optimize for speed (less accuracy)."""
        AgentConfig.RETRY_SETTINGS['max_retries'] = 5
        AgentConfig.RAG_SETTINGS['max_context_size'] = 2000
        AgentConfig.LANGCHAIN_SETTINGS['temperature'] = 0.1
        AgentConfig.VISION_SETTINGS['screenshot_max_size'] = (800, 600)
    
    @staticmethod
    def accurate_mode():
        """Optimize for accuracy (slower)."""
        AgentConfig.RETRY_SETTINGS['max_retries'] = 15
        AgentConfig.RAG_SETTINGS['max_context_size'] = 5000
        AgentConfig.LANGCHAIN_SETTINGS['temperature'] = 0.5
        AgentConfig.VISION_SETTINGS['screenshot_max_size'] = (1920, 1080)
    
    @staticmethod
    def traditional_mode():
        """Use traditional mode without RAG."""
        AgentConfig.RAG_ENABLED = False
        AgentConfig.LANGCHAIN_ENABLED = False
        AgentConfig.LANGGRAPH_ENABLED = False
    
    @staticmethod
    def rag_only_mode():
        """Use RAG but not LangGraph workflows."""
        AgentConfig.RAG_ENABLED = True
        AgentConfig.LANGCHAIN_ENABLED = True
        AgentConfig.LANGGRAPH_ENABLED = False


# ==================================================================================
# USAGE EXAMPLE
# ==================================================================================

def print_config_summary():
    """Print configuration summary."""
    print("=" * 80)
    print("  Neural AI Agent - Configuration")
    print("=" * 80)
    
    print(f"\nMode: {AgentConfig.get_mode_description()}")
    
    print("\nFeatures Enabled:")
    print(f"  RAG (Semantic Retrieval):     {'✅' if AgentConfig.RAG_ENABLED else '❌'}")
    print(f"  LangChain (CoT Reasoning):    {'✅' if AgentConfig.LANGCHAIN_ENABLED else '❌'}")
    print(f"  LangGraph (Workflows):        {'✅' if AgentConfig.LANGGRAPH_ENABLED else '❌'}")
    print(f"  Voice Interface:              {'✅' if AgentConfig.VOICE_ENABLED else '❌'}")
    
    print("\nKey Settings:")
    print(f"  Max Retries:                  {AgentConfig.RETRY_SETTINGS['max_retries']}")
    print(f"  RAG Context Size:             {AgentConfig.RAG_SETTINGS['max_context_size']} chars")
    print(f"  LLM Temperature:              {AgentConfig.LANGCHAIN_SETTINGS['temperature']}")
    print(f"  Image Match Threshold:        {AgentConfig.VISION_SETTINGS['image_match_threshold']}")
    
    print("\nDirectories:")
    print(f"  Prompts:                      {AgentConfig.PROMPT_SETTINGS['prompts_dir']}")
    print(f"  Knowledge Base:               {AgentConfig.KNOWLEDGE_BASE_SETTINGS['directory']}")
    print(f"  Reports:                      {AgentConfig.REPORT_SETTINGS['output_dir']}")
    print(f"  Vector DB:                    {AgentConfig.RAG_SETTINGS['vector_db_path']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Print configuration summary
    print_config_summary()
    
    # Create directories
    AgentConfig.create_directories()
    print("\n✅ Directories created")
