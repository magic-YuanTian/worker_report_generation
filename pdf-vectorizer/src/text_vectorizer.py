# src/text_vectorizer.py
# Text vectorization using various embedding models.
#   vectorize_chunks           – main entry point for vectorization
#   SentenceTransformerVectorizer – local embeddings (sentence-transformers)
#   OpenAIVectorizer           – OpenAI embeddings API
#   save_embeddings            – save embeddings to various formats

from contextlib import nullcontext
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .text_chunker import Chunk
from .utils.cli_utils import print_info, print_warning, spinner, ProgressContext


class VectorizationError(Exception):
    """Raised when vectorization fails."""
    pass


def vectorize_chunks(
    chunks: List[Chunk],
    model_type: str = "sentence_transformers",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    openai_api_key: Optional[str] = None,
) -> np.ndarray:
    """Vectorize text chunks into embeddings.
    
    Args:
        chunks: List of Chunk objects
        model_type: 'sentence_transformers' or 'openai'
        model_name: Model identifier
        batch_size: Batch size for processing
        openai_api_key: OpenAI API key (if using OpenAI)
    
    Returns:
        NumPy array of embeddings (shape: [num_chunks, embedding_dim])
    """
    if not chunks:
        return np.array([])
    
    texts = [chunk.text for chunk in chunks]
    
    if model_type == "sentence_transformers":
        vectorizer = SentenceTransformerVectorizer(
            model_name=model_name,
            batch_size=batch_size,
        )
        return vectorizer.encode(texts)
    
    elif model_type == "openai":
        if not OPENAI_AVAILABLE:
            raise VectorizationError(
                "OpenAI not available. Install: uv add openai"
            )
        vectorizer = OpenAIVectorizer(
            model_name=model_name,
            api_key=openai_api_key,
            batch_size=batch_size,
        )
        return vectorizer.encode(texts)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class SentenceTransformerVectorizer:
    """Local embedding generation using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise VectorizationError(
                "sentence-transformers not available. Install: uv add sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        print_info(f"Loading sentence-transformer model: {model_name}")
        with spinner(f"Loading {model_name}..."):
            self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print_info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
        
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        print_info(f"Encoding {len(texts)} chunks...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        return embeddings


class OpenAIVectorizer:
    """Cloud embedding generation using OpenAI API."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        if not OPENAI_AVAILABLE:
            raise VectorizationError(
                "OpenAI not available. Install: uv add openai"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY env var
            self.client = openai.OpenAI()
        
        # Embedding dimensions for known models
        self.embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.embedding_dim = self.embedding_dims.get(model_name, 1536)
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings using OpenAI API.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
        
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        print_info(f"Encoding {len(texts)} chunks with OpenAI {self.model_name}...")
        
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        ctx = ProgressContext(total=total_batches, desc="Calling OpenAI API") if show_progress else nullcontext()

        with ctx as progress:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                embeddings.extend(item.embedding for item in response.data)
                if progress:
                    progress.update(1)

        return np.array(embeddings)


def save_embeddings(
    embeddings: np.ndarray,
    chunks: List[Chunk],
    output_path: Path,
    format: str = "numpy",
    include_metadata: bool = True,
) -> None:
    """Save embeddings to file.
    
    Args:
        embeddings: NumPy array of embeddings
        chunks: List of Chunk objects
        output_path: Output file path
        format: 'numpy', 'json', or 'jsonl'
        include_metadata: Include chunk metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "numpy":
        # Save as .npz with metadata
        data = {"embeddings": embeddings}
        
        if include_metadata:
            data["texts"] = np.array([chunk.text for chunk in chunks])
            data["chunk_ids"] = np.array([chunk.chunk_id for chunk in chunks])
            data["pages"] = np.array([chunk.source_page or -1 for chunk in chunks])
        
        np.savez(output_path, **data)
        print_info(f"Saved embeddings to {output_path} (numpy format)")
    
    elif format == "json":
        # Save as JSON with full metadata
        data = {
            "embeddings": embeddings.tolist(),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page": chunk.source_page,
                    "metadata": chunk.metadata,
                } if include_metadata else {
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "num_chunks": len(chunks),
        }
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print_info(f"Saved embeddings to {output_path} (JSON format)")
    
    elif format == "jsonl":
        # Save as JSONL (one chunk per line)
        with output_path.open("w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                entry = {
                    "embedding": embeddings[i].tolist(),
                    "chunk_id": chunk.chunk_id,
                }
                
                if include_metadata:
                    entry.update({
                        "text": chunk.text,
                        "page": chunk.source_page,
                        "metadata": chunk.metadata,
                    })
                
                f.write(json.dumps(entry) + "\n")
        
        print_info(f"Saved embeddings to {output_path} (JSONL format)")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_embeddings(path: Path, format: str = "numpy") -> Dict[str, Any]:
    """Load embeddings from file.
    
    Args:
        path: Path to embeddings file
        format: 'numpy', 'json', or 'jsonl'
    
    Returns:
        Dictionary with embeddings and metadata
    """
    if format == "numpy":
        data = np.load(path)
        return {
            "embeddings": data["embeddings"],
            "texts": data.get("texts", []),
            "chunk_ids": data.get("chunk_ids", []),
            "pages": data.get("pages", []),
        }
    
    elif format == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "embeddings": np.array(data["embeddings"]),
            "chunks": data["chunks"],
            "embedding_dim": data["embedding_dim"],
            "num_chunks": data["num_chunks"],
        }
    
    elif format == "jsonl":
        embeddings = []
        chunks = []
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                embeddings.append(entry["embedding"])
                chunks.append({
                    "chunk_id": entry["chunk_id"],
                    "text": entry.get("text", ""),
                    "page": entry.get("page"),
                    "metadata": entry.get("metadata", {}),
                })
        
        return {
            "embeddings": np.array(embeddings),
            "chunks": chunks,
        }
    
    else:
        raise ValueError(f"Unknown format: {format}")
