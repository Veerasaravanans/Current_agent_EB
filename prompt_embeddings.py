"""
prompt_embeddings.py - Semantic Vector Database for Prompt Management

Uses embeddings to store and retrieve prompts by MEANING, not keywords.
Handles 5000+ lines efficiently with vector similarity search.

Technologies:
- ChromaDB: Vector database
- Sentence Transformers: Embeddings
- Semantic search: Retrieve relevant prompts by context
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not installed. Install with: pip install sentence-transformers")


class PromptEmbeddingsManager:
    """
    Manages prompt embeddings and semantic retrieval.
    
    Features:
    - Embeds all prompts into vector space
    - Stores in ChromaDB for fast retrieval
    - Semantic search: Find prompts by meaning
    - Handles 5000+ lines efficiently
    - Auto-updates when prompts change
    """
    
    def __init__(
        self,
        prompts_dir: str = "./prompts",
        db_dir: str = "./vector_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize embeddings manager.
        
        Args:
            prompts_dir: Directory containing prompt markdown files
            db_dir: Directory for vector database
            embedding_model: Sentence transformer model name
        """
        self.prompts_dir = Path(prompts_dir)
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}...")
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info("✅ Embedding model loaded")
        else:
            self.embedding_model = None
            logger.error("❌ Sentence Transformers not available")
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="automotive_prompts",
                metadata={"description": "Automotive testing prompt embeddings"}
            )
            logger.info("✅ ChromaDB initialized")
        else:
            self.chroma_client = None
            self.collection = None
            logger.error("❌ ChromaDB not available")
        
        # Track prompt versions
        self.prompt_hashes = {}
        self.load_prompt_hashes()
    
    def load_prompt_hashes(self):
        """Load stored prompt hashes to detect changes."""
        hash_file = self.db_dir / "prompt_hashes.json"
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                self.prompt_hashes = json.load(f)
    
    def save_prompt_hashes(self):
        """Save prompt hashes."""
        hash_file = self.db_dir / "prompt_hashes.json"
        with open(hash_file, 'w') as f:
            json.dump(self.prompt_hashes, f, indent=2)
    
    def compute_file_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of file content."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'size': current_size
                })
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) for l in current_chunk)
                chunk_id += 1
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'chunk_id': chunk_id,
                'size': current_size
            })
        
        return chunks
    
    def embed_prompt_file(self, filepath: Path, force: bool = False) -> int:
        """
        Embed a single prompt file into vector database.
        
        Args:
            filepath: Path to markdown file
            force: Force re-embedding even if unchanged
            
        Returns:
            Number of chunks embedded
        """
        if not self.embedding_model or not self.collection:
            logger.error("Embedding infrastructure not available")
            return 0
        
        try:
            # Check if file changed
            current_hash = self.compute_file_hash(filepath)
            file_key = str(filepath)
            
            if not force and file_key in self.prompt_hashes:
                if self.prompt_hashes[file_key] == current_hash:
                    logger.debug(f"Skipping unchanged file: {filepath.name}")
                    return 0
            
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {filepath.name}")
                return 0
            
            # Chunk text
            chunks = self.chunk_text(content, chunk_size=500, overlap=50)
            
            logger.info(f"Embedding {filepath.name}: {len(chunks)} chunks...")
            
            # Embed each chunk
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Generate IDs
            base_id = filepath.stem
            ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Metadata
            metadatas = [
                {
                    'source_file': filepath.name,
                    'chunk_id': chunk['chunk_id'],
                    'category': self._categorize_prompt(filepath),
                    'file_hash': current_hash
                }
                for chunk in chunks
            ]
            
            # Delete old embeddings for this file
            try:
                existing_ids = [f"{base_id}_chunk_{i}" for i in range(100)]
                self.collection.delete(ids=existing_ids)
            except:
                pass
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            # Update hash
            self.prompt_hashes[file_key] = current_hash
            
            logger.info(f"✅ Embedded {filepath.name}: {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to embed {filepath}: {e}")
            return 0
    
    def _categorize_prompt(self, filepath: Path) -> str:
        """Categorize prompt file by name."""
        name = filepath.stem.lower()
        
        if 'base' in name:
            return 'base'
        elif 'hvac' in name:
            return 'hvac'
        elif 'media' in name:
            return 'media'
        elif 'navigation' in name:
            return 'navigation'
        elif 'error' in name:
            return 'error_handling'
        elif 'learned' in name:
            return 'learned_solutions'
        elif 'custom' in name:
            return 'custom_commands'
        else:
            return 'general'
    
    def embed_all_prompts(self, force: bool = False) -> Dict[str, int]:
        """
        Embed all prompt files in the prompts directory.
        
        Args:
            force: Force re-embedding all files
            
        Returns:
            Dict with file names and chunk counts
        """
        if not self.prompts_dir.exists():
            logger.error(f"Prompts directory not found: {self.prompts_dir}")
            return {}
        
        results = {}
        
        # Embed base prompts
        for md_file in self.prompts_dir.glob("*.md"):
            count = self.embed_prompt_file(md_file, force=force)
            results[md_file.name] = count
        
        # Embed component-specific prompts
        component_dir = self.prompts_dir / "component_specific"
        if component_dir.exists():
            for md_file in component_dir.glob("*.md"):
                count = self.embed_prompt_file(md_file, force=force)
                results[f"component/{md_file.name}"] = count
        
        # Save hashes
        self.save_prompt_hashes()
        
        total_chunks = sum(results.values())
        logger.info(f"✅ Embedded {len(results)} files, {total_chunks} total chunks")
        
        return results
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search: Find prompts by meaning.
        
        Args:
            query: Search query (natural language)
            n_results: Number of results to return
            category: Optional category filter
            
        Returns:
            List of relevant prompt chunks with metadata
        """
        if not self.embedding_model or not self.collection:
            logger.error("Semantic search not available")
            return []
        
        try:
            # Embed query
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Build filter
            where = {"category": category} if category else None
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            logger.debug(f"Found {len(formatted_results)} relevant chunks for: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_context_for_task(
        self,
        task_description: str,
        component: Optional[str] = None,
        max_chunks: int = 10
    ) -> str:
        """
        Get relevant context for a specific task.
        
        Args:
            task_description: Description of the task
            component: Optional component (hvac, media, navigation)
            max_chunks: Maximum chunks to retrieve
            
        Returns:
            Combined context string
        """
        # Search for relevant prompts
        category = component if component else None
        results = self.semantic_search(task_description, n_results=max_chunks, category=category)
        
        if not results:
            logger.warning(f"No relevant context found for: {task_description}")
            return ""
        
        # Combine results
        context_parts = []
        for result in results:
            source = result['metadata'].get('source_file', 'unknown')
            text = result['text']
            context_parts.append(f"## From {source}\n\n{text}\n")
        
        combined_context = "\n".join(context_parts)
        
        logger.info(f"Retrieved {len(results)} relevant chunks ({len(combined_context)} chars)")
        return combined_context
    
    def get_error_solutions(self, error_description: str, n_results: int = 3) -> List[Dict]:
        """
        Get relevant error solutions.
        
        Args:
            error_description: Description of the error
            n_results: Number of solutions to return
            
        Returns:
            List of relevant solutions
        """
        return self.semantic_search(
            query=error_description,
            n_results=n_results,
            category='error_handling'
        )
    
    def get_component_knowledge(self, component: str, task: str = "") -> str:
        """
        Get component-specific knowledge.
        
        Args:
            component: Component name (hvac, media, navigation)
            task: Optional specific task
            
        Returns:
            Relevant knowledge string
        """
        query = f"{component} {task}" if task else component
        return self.get_context_for_task(query, component=component, max_chunks=5)
    
    def rebuild_index(self):
        """Rebuild entire vector index from scratch."""
        logger.info("🔄 Rebuilding vector index...")
        
        # Clear collection
        if self.collection:
            try:
                self.chroma_client.delete_collection("automotive_prompts")
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name="automotive_prompts",
                metadata={"description": "Automotive testing prompt embeddings"}
            )
        
        # Clear hashes
        self.prompt_hashes = {}
        
        # Re-embed all
        results = self.embed_all_prompts(force=True)
        
        logger.info(f"✅ Index rebuilt: {sum(results.values())} chunks")
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'files_indexed': len(self.prompt_hashes),
                'embedding_model': self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None,
                'db_path': str(self.db_dir)
            }
        except:
            return {}


def main():
    """Test embeddings system."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing Prompt Embeddings System")
    print("=" * 80)
    
    # Initialize
    manager = PromptEmbeddingsManager()
    
    # Embed all prompts
    print("\n1. Embedding all prompts...")
    results = manager.embed_all_prompts()
    
    print("\nEmbedded files:")
    for file, count in results.items():
        print(f"  - {file}: {count} chunks")
    
    # Test semantic search
    print("\n2. Testing semantic search...")
    
    test_queries = [
        "How do I turn on the AC?",
        "PACCAR media double tap",
        "Temperature slider not responding",
        "Open navigation app"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        results = manager.semantic_search(query, n_results=2)
        
        for i, result in enumerate(results, 1):
            source = result['metadata']['source_file']
            preview = result['text'][:100].replace('\n', ' ')
            print(f"     {i}. [{source}] {preview}...")
    
    # Statistics
    print("\n3. Database statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Embeddings system test complete!")


if __name__ == "__main__":
    main()
