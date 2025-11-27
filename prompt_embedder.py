"""
prompt_embedder.py - Embedding-Based Prompt Retrieval System

Handles large prompts (5000+ lines) efficiently using embeddings.
Only sends RELEVANT portions of prompts to lightweight moondream2 model.

How it works:
1. Split all prompts into chunks
2. Create embeddings for each chunk using Ollama
3. Store in vector database (FAISS)
4. When needed, retrieve only relevant chunks based on semantic search
5. Moondream2 only processes relevant context instead of all 5000+ lines!

Created by Veera Saravanan
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    from langchain_ollama import OllamaEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Embedding libraries not available. Install: pip install langchain langchain-ollama faiss-cpu")

logger = logging.getLogger(__name__)


class PromptEmbedder:
    """
    Embedding-based prompt management for large prompts (5000+ lines).
    
    Features:
    - Chunk prompts into manageable pieces
    - Create embeddings using Ollama
    - Semantic search for relevant prompts
    - Cache embeddings for fast retrieval
    - Support for 5000+ line prompts without overwhelming lightweight models
    """
    
    def __init__(
        self,
        prompts_dir: str = "./prompts",
        embeddings_dir: str = "./embeddings_cache",
        model_name: str = "nomic-embed-text",  # Lightweight embedding model
        chunk_size: int = 500,  # Characters per chunk
        chunk_overlap: int = 50,
        top_k: int = 5  # Number of relevant chunks to retrieve
    ):
        """
        Initialize the prompt embedder.
        
        Args:
            prompts_dir: Directory containing prompt files
            embeddings_dir: Directory to cache embeddings
            model_name: Ollama embedding model (default: nomic-embed-text)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            top_k: Number of most relevant chunks to retrieve
        """
        self.prompts_dir = Path(prompts_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        
        if EMBEDDINGS_AVAILABLE:
            self._initialize_components()
        else:
            logger.warning("⚠️  Embeddings not available. Falling back to full prompts.")
        
        logger.info(f"Prompt Embedder initialized (top_k={top_k}, chunk_size={chunk_size})")
    
    def _initialize_components(self):
        """Initialize embedding model and text splitter."""
        try:
            # Initialize Ollama embeddings
            logger.info(f"Initializing embeddings with model: {self.model_name}")
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n##", "\n##", "\n\n", "\n", " ", ""],
                length_function=len
            )
            
            logger.info("✅ Embedding components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
            logger.warning(f"And model is pulled: ollama pull {self.model_name}")
            raise
    
    def create_embeddings_from_prompts(self, force_rebuild: bool = False):
        """
        Create embeddings from all prompt files.
        
        Args:
            force_rebuild: Force rebuild even if cache exists
        """
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Embeddings not available. Skipping.")
            return False
        
        cache_file = self.embeddings_dir / "prompt_vectorstore.pkl"
        
        # Check if cache exists and is valid
        if cache_file.exists() and not force_rebuild:
            logger.info("Loading embeddings from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    self.vector_store = pickle.load(f)
                logger.info("✅ Embeddings loaded from cache")
                return True
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Rebuilding...")
        
        # Build embeddings from scratch
        logger.info("Building embeddings from prompt files...")
        
        try:
            all_documents = []
            
            # Load all prompt files
            prompt_files = [
                'base_prompts.md',
                'error_handling.md',
                'learned_solutions.md',
                'custom_commands.md'
            ]
            
            # Add base prompts
            for filename in prompt_files:
                filepath = self.prompts_dir / filename
                if filepath.exists():
                    content = filepath.read_text(encoding='utf-8')
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(content)
                    
                    # Create documents with metadata
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': filename,
                                'chunk_id': i,
                                'type': 'base'
                            }
                        )
                        all_documents.append(doc)
                    
                    logger.info(f"  Processed {filename}: {len(chunks)} chunks")
            
            # Add component-specific prompts
            component_dir = self.prompts_dir / "component_specific"
            if component_dir.exists():
                for prompt_file in component_dir.glob("*.md"):
                    content = prompt_file.read_text(encoding='utf-8')
                    chunks = self.text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': prompt_file.name,
                                'component': prompt_file.stem,
                                'chunk_id': i,
                                'type': 'component'
                            }
                        )
                        all_documents.append(doc)
                    
                    logger.info(f"  Processed {prompt_file.name}: {len(chunks)} chunks")
            
            logger.info(f"Total documents: {len(all_documents)}")
            
            # Create vector store
            logger.info("Creating vector embeddings (this may take a minute)...")
            self.vector_store = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings
            )
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(self.vector_store, f)
            
            logger.info(f"✅ Embeddings created and cached: {len(all_documents)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return False
    
    def get_relevant_prompts(
        self,
        query: str,
        component: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> str:
        """
        Retrieve ONLY relevant prompt chunks for a given query.
        
        This is the KEY function that makes large prompts work with moondream2!
        
        Args:
            query: The search query (e.g., "How to tap AC button")
            component: Optional component filter (e.g., 'hvac')
            filter_type: Optional type filter ('base', 'component', 'error')
            
        Returns:
            Concatenated relevant prompt chunks (much smaller than full prompts)
        """
        if not EMBEDDINGS_AVAILABLE or self.vector_store is None:
            logger.warning("Embeddings not available. Cannot retrieve relevant prompts.")
            return ""
        
        try:
            # Build filter
            search_kwargs = {"k": self.top_k}
            
            if component or filter_type:
                filter_dict = {}
                if component:
                    filter_dict['component'] = component
                if filter_type:
                    filter_dict['type'] = filter_type
                search_kwargs['filter'] = filter_dict
            
            # Semantic search
            logger.debug(f"Searching for: '{query[:50]}...'")
            docs = self.vector_store.similarity_search(query, **search_kwargs)
            
            # Concatenate relevant chunks
            relevant_prompts = []
            for doc in docs:
                source = doc.metadata.get('source', 'unknown')
                relevant_prompts.append(f"# From: {source}\n\n{doc.page_content}\n")
            
            result = "\n---\n\n".join(relevant_prompts)
            
            logger.info(f"✅ Retrieved {len(docs)} relevant chunks ({len(result)} chars) from {len(relevant_prompts)} sources")
            logger.debug(f"  Original full prompts: ~5000+ lines")
            logger.debug(f"  Relevant context: {len(result)} characters")
            logger.debug(f"  Reduction: {100 * (1 - len(result)/50000):.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant prompts: {e}")
            return ""
    
    def add_learned_solution_and_reindex(
        self,
        problem: str,
        solution: str,
        added_by: str = "Veera Saravanan"
    ):
        """
        Add learned solution and update embeddings.
        
        Args:
            problem: Problem description
            solution: Solution that worked
            added_by: Architect name
        """
        try:
            # Add to learned_solutions.md
            filepath = self.prompts_dir / "learned_solutions.md"
            
            timestamp = datetime.now().strftime("%Y-%m-%d")
            entry = f"\n\n## [{timestamp}] New Solution\n\n"
            entry += f"**Problem**: {problem}\n"
            entry += f"**Solution**: {solution}\n"
            entry += f"**Added by**: {added_by}\n"
            entry += "\n---\n"
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(entry)
            
            logger.info(f"✅ Added learned solution to file")
            
            # Rebuild embeddings to include new solution
            logger.info("Rebuilding embeddings to include new solution...")
            self.create_embeddings_from_prompts(force_rebuild=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add learned solution: {e}")
            return False


def test_embeddings():
    """Test the embedding system."""
    logger.info("=" * 70)
    logger.info("Testing Prompt Embedding System")
    logger.info("=" * 70)
    
    # Initialize embedder
    embedder = PromptEmbedder()
    
    # Create embeddings
    logger.info("\n1. Creating embeddings from prompts...")
    success = embedder.create_embeddings_from_prompts()
    
    if not success:
        logger.error("Failed to create embeddings")
        return
    
    # Test retrieval
    logger.info("\n2. Testing semantic search...")
    
    test_queries = [
        "How to tap AC button in HVAC",
        "PACCAR Media requires double tap",
        "What to do when button tap fails",
        "How to swipe temperature control"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        relevant = embedder.get_relevant_prompts(query, component='hvac')
        logger.info(f"Retrieved {len(relevant)} characters of relevant context")
        logger.info(f"Sample: {relevant[:200]}...")
    
    logger.info("\n✅ Embedding system test complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_embeddings()
