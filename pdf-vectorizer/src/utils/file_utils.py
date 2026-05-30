# src/utils/file_utils.py
# File handling utilities.
#   list_pdf_files      – find all PDF files in directory
#   ensure_dir          – create directory if it doesn't exist
#   get_output_path     – generate output path for processed files
#   safe_filename       – sanitize filename for filesystem

from pathlib import Path
from typing import List, Optional
import re


def list_pdf_files(directory: Path, recursive: bool = False) -> List[Path]:
    """List all PDF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Search subdirectories
    
    Returns:
        List of PDF file paths
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if recursive:
        pdf_files = list(directory.rglob("*.pdf"))
    else:
        pdf_files = list(directory.glob("*.pdf"))
    
    return sorted(pdf_files)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path.
    
    Args:
        path: Directory path
    
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(
    input_path: Path,
    output_dir: Path,
    suffix: str = "",
    extension: str = None
) -> Path:
    """Generate output path for a processed file.
    
    Args:
        input_path: Original input file path
        output_dir: Output directory
        suffix: Optional suffix to add to filename
        extension: New extension (e.g., '.json', '.txt')
    
    Returns:
        Output file path
    
    Example:
        input: data/raw/document.pdf
        output: data/processed/document_chunks.json
    """
    ensure_dir(output_dir)
    
    # Get base filename without extension
    base_name = input_path.stem
    
    # Add suffix if provided
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    # Use new extension or keep original
    if extension:
        ext = extension if extension.startswith('.') else f'.{extension}'
    else:
        ext = input_path.suffix
    
    return output_dir / f"{base_name}{ext}"


def safe_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize filename by removing/replacing unsafe characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace unsafe chars with
    
    Returns:
        Safe filename
    """
    # Remove unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', replacement, filename)
    
    # Remove leading/trailing spaces and dots
    safe = safe.strip('. ')
    
    # Limit length
    if len(safe) > 255:
        safe = safe[:255]
    
    return safe


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Example:
        format_size(1536) → "1.5 KB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
