# src/text_chunker.py
# Text chunking strategies for embedding preparation.
#   chunk_text             – main entry point for text chunking
#   RecursiveChunker       – recursive character splitting (recommended)
#   FixedSizeChunker       – simple fixed-size chunks
#   SemanticChunker        – semantic similarity-based chunking

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .utils.cli_utils import print_warning


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: int
    source_page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_page": self.source_page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata or {},
        }


def chunk_text(
    pages: List[Dict[str, Any]],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[Chunk]:
    """Chunk text from PDF pages into smaller pieces for embedding.
    
    Args:
        pages: List of page dictionaries from PDF extraction
        strategy: Chunking strategy ('recursive', 'fixed', 'semantic')
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
        separators: Custom separators for recursive chunking
    
    Returns:
        List of Chunk objects
    """
    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        ).chunk(pages)
    elif strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).chunk(pages)
    elif strategy == "semantic":
        print_warning("Semantic chunking not yet implemented, using recursive")
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).chunk(pages)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


class RecursiveChunker:
    """Recursive character text splitter (best for most use cases).
    
    Tries to split on paragraph breaks first, then sentences, then words.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators prioritize natural breaks
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "! ",    # Sentences
            "? ",    # Sentences
            "; ",    # Clauses
            ", ",    # Phrases
            " ",     # Words
            "",      # Characters (fallback)
        ]
        
        # Use langchain if available, otherwise custom implementation
        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
                length_function=len,
            )
        else:
            self.splitter = None
    
    def chunk(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk pages into text chunks."""
        chunks = []
        chunk_id = 0
        
        for page in pages:
            page_num = page.get("page_number")
            text = page.get("text", "")
            figures = page.get("figures", [])

            if not text.strip():
                continue

            if self.splitter:
                splits = self.splitter.split_text(text)
            else:
                splits = self._split_text_recursive(text)

            for split in splits:
                meta = {"page": page_num}
                if figures:
                    meta["figures"] = figures
                chunks.append(Chunk(
                    text=split,
                    chunk_id=chunk_id,
                    source_page=page_num,
                    metadata=meta,
                ))
                chunk_id += 1
        
        return chunks
    
    def _split_text_recursive(self, text: str) -> List[str]:
        """Custom recursive splitting implementation."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current = ""
                
                for split in splits:
                    # Add separator back (except for empty string separator)
                    if separator:
                        split = split + separator
                    
                    # If adding this split exceeds size, save current and start new
                    if len(current) + len(split) > self.chunk_size and current:
                        chunks.append(current.strip())
                        # Start new chunk with overlap
                        overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                        current = overlap_text + split
                    else:
                        current += split
                
                # Add remaining text
                if current.strip():
                    chunks.append(current.strip())
                
                return chunks
        
        # Fallback: character-based splitting
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


class FixedSizeChunker:
    """Simple fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk pages into fixed-size chunks."""
        chunks = []
        chunk_id = 0
        
        for page in pages:
            page_num = page.get("page_number")
            text = page.get("text", "")
            figures = page.get("figures", [])

            if not text.strip():
                continue

            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(text), step):
                chunk_text = text[i:i + self.chunk_size].strip()

                if chunk_text:
                    meta = {"page": page_num}
                    if figures:
                        meta["figures"] = figures
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source_page=page_num,
                        char_start=i,
                        char_end=i + len(chunk_text),
                        metadata=meta,
                    ))
                    chunk_id += 1
        
        return chunks


def get_chunk_statistics(chunks: List[Chunk]) -> Dict[str, Any]:
    """Get statistics about the chunks.
    
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
        }
    
    lengths = [len(chunk.text) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_chars": sum(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }
