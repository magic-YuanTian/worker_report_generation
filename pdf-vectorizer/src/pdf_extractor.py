# src/pdf_extractor.py
# PDF text extraction with support for multiple methods.
#   extract_text_from_pdf  – main entry point, delegates to appropriate method
#   extract_with_pymupdf   – fast extraction using PyMuPDF; saves figures to figures_dir
#   extract_with_pdfplumber – better for tables and structured content

from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .utils.cli_utils import print_warning, progress_iter


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


def extract_text_from_pdf(
    pdf_path: Path,
    method: str = "pymupdf",
    figures_dir: Optional[Path] = None,
    min_figure_px: int = 50,
    extract_tables: bool = False,
) -> List[Dict[str, Any]]:
    """Extract text (and optionally figures) from a PDF.

    Args:
        pdf_path: Path to PDF file
        method: Extraction method ('pymupdf', 'pdfplumber')
        figures_dir: Directory to save extracted images; None to skip
        min_figure_px: Minimum width and height in pixels; smaller images are skipped
        extract_tables: Extract tables separately (pdfplumber only)

    Returns:
        List of page dictionaries with text and metadata

    Raises:
        PDFExtractionError: If extraction fails
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if method == "pymupdf":
        return extract_with_pymupdf(pdf_path, figures_dir=figures_dir, min_figure_px=min_figure_px)
    elif method == "pdfplumber":
        if not PDFPLUMBER_AVAILABLE:
            print_warning("pdfplumber not installed, falling back to pymupdf")
            return extract_with_pymupdf(pdf_path, figures_dir=figures_dir, min_figure_px=min_figure_px)
        return extract_with_pdfplumber(pdf_path, extract_tables=extract_tables)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def extract_with_pymupdf(
    pdf_path: Path,
    figures_dir: Optional[Path] = None,
    min_figure_px: int = 50,
) -> List[Dict[str, Any]]:
    """Extract text using PyMuPDF (fastest method).

    Args:
        pdf_path: Path to PDF file
        figures_dir: Directory to save embedded images; None to skip

    Returns:
        List of page data dictionaries
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Failed to open PDF: {e}")

    pages = []

    for page_num in progress_iter(range(len(doc)), total=len(doc), desc="Extracting pages (pymupdf)"):
        page = doc[page_num]
        text = page.get_text()

        page_data = {
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text),
            "metadata": {
                "width": page.rect.width,
                "height": page.rect.height,
            },
        }

        if figures_dir is not None:
            saved = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image["width"] < min_figure_px or base_image["height"] < min_figure_px:
                        continue
                    ext = base_image["ext"]
                    fname = f"page_{page_num + 1}_fig_{img_index}.{ext}"
                    figures_dir.mkdir(parents=True, exist_ok=True)
                    (figures_dir / fname).write_bytes(base_image["image"])
                    saved.append(fname)
                except Exception as e:
                    print_warning(f"  skipped image xref={xref} page={page_num + 1}: {e}")
            if saved:
                page_data["figures"] = saved

        pages.append(page_data)

    doc.close()
    return pages


def extract_with_pdfplumber(
    pdf_path: Path,
    extract_tables: bool = False,
) -> List[Dict[str, Any]]:
    """Extract text using pdfplumber (better for tables).
    
    Args:
        pdf_path: Path to PDF file
        extract_tables: Extract tables separately
    
    Returns:
        List of page data dictionaries
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []

            for page_num, page in progress_iter(enumerate(pdf.pages), total=len(pdf.pages), desc="Extracting pages (pdfplumber)"):
                text = page.extract_text() or ""
                
                page_data = {
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "width": page.width,
                        "height": page.height,
                    }
                }
                
                # Extract tables if requested
                if extract_tables:
                    tables = page.extract_tables()
                    if tables:
                        page_data["tables"] = tables
                
                pages.append(page_data)
            
            return pages
    except Exception as e:
        raise PDFExtractionError(f"pdfplumber extraction failed: {e}")


def get_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    """Get PDF metadata and information.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        info = {
            "filename": pdf_path.name,
            "pages": len(doc),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }
        
        doc.close()
        return info
    except Exception as e:
        raise PDFExtractionError(f"Failed to get PDF info: {e}")
