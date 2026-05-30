# src/pipeline.py
# Main PDF vectorization pipeline.

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .configuration import (
    load_config,
    get_paths_config,
    get_extraction_config,
    get_chunking_config,
    get_vectorization_config,
)
from .pdf_extractor import extract_text_from_pdf
from .text_chunker import chunk_text, Chunk, get_chunk_statistics
from .text_vectorizer import vectorize_chunks, save_embeddings
from .utils import (
    list_pdf_files,
    print_info,
    print_success,
    print_error,
    print_warning,
    progress_iter,
)


class PDFVectorizationPipeline:

    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_config(config_path)
        self.paths = get_paths_config(self.config)
        self.extraction_config = get_extraction_config(self.config)
        self.chunking_config = get_chunking_config(self.config)
        self.vectorization_config = get_vectorization_config(self.config)

    def _out_dir(self, pdf_path: Path) -> Path:
        """Per-PDF output directory: data/processed/<stem>/"""
        d = self.paths["processed"] / pdf_path.stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        print_info(f"Processing: {pdf_path.name}")
        out_dir = self._out_dir(pdf_path)

        text_path = out_dir / "text.json"
        chunks_path = out_dir / "chunks.json"
        fmt = self.vectorization_config.get("output_format", "numpy")
        embeddings_path = out_dir / ("embeddings.npz" if fmt == "numpy" else "embeddings.json")

        result = {"pdf": str(pdf_path), "status": "processing"}

        try:
            # Step 1: Extract
            if text_path.exists():
                print_info("  Step 1/3: text.json found — skipping extraction")
                pages = json.loads(text_path.read_text(encoding="utf-8"))
            else:
                print_info("  Step 1/3: Extracting text...")
                figures_dir = out_dir / "figures" if self.extraction_config.get("extract_figures", True) else None
                pages = extract_text_from_pdf(
                    pdf_path,
                    method=self.extraction_config.get("method", "pymupdf"),
                    figures_dir=figures_dir,
                    min_figure_px=self.extraction_config.get("min_figure_px", 50),
                    extract_tables=self.extraction_config.get("extract_tables", False),
                )
                text_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")

            total_chars = sum(p.get("char_count", 0) for p in pages)
            print_info(f"    {len(pages)} pages, {total_chars:,} chars → {text_path.name}")
            result["pages"] = len(pages)
            result["text_file"] = str(text_path)

            # Step 2: Chunk
            if chunks_path.exists():
                print_info("  Step 2/3: chunks.json found — skipping chunking")
                chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
            else:
                print_info("  Step 2/3: Chunking text...")
                chunks = chunk_text(
                    pages,
                    strategy=self.chunking_config.get("strategy", "recursive"),
                    chunk_size=self.chunking_config.get("chunk_size", 1000),
                    chunk_overlap=self.chunking_config.get("chunk_overlap", 200),
                    separators=self.chunking_config.get("separators"),
                )
                chunks_path.write_text(
                    json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            stats = get_chunk_statistics(chunks)
            print_info(f"    {stats['total_chunks']} chunks, avg {stats['avg_length']:.0f} chars → {chunks_path.name}")
            result["chunks"] = stats["total_chunks"]
            result["chunks_file"] = str(chunks_path)

            # Step 3: Embed
            print_info("  Step 3/3: Generating embeddings...")
            embeddings = vectorize_chunks(
                chunks,
                model_type=self.vectorization_config.get("model_type", "sentence_transformers"),
                model_name=self.vectorization_config.get("model_name", "all-MiniLM-L6-v2"),
                batch_size=self.vectorization_config.get("batch_size", 32),
                openai_api_key=self.vectorization_config.get("openai_api_key"),
            )
            save_embeddings(
                embeddings,
                chunks,
                embeddings_path,
                format=fmt,
                include_metadata=self.vectorization_config.get("include_metadata", True),
            )
            print_info(f"    {len(embeddings)} embeddings → {embeddings_path.name}")
            result["embeddings"] = len(embeddings)
            result["embeddings_file"] = str(embeddings_path)
            result["status"] = "success"
            print_success(f"Done: {pdf_path.name}")

        except Exception as e:
            print_error(f"Failed: {pdf_path.name} — {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def process_directory(self, input_dir: Path, recursive: bool = False) -> List[Dict[str, Any]]:
        pdf_files = list_pdf_files(input_dir, recursive=recursive)
        if not pdf_files:
            print_warning(f"No PDFs found in {input_dir}")
            return []

        print_info(f"Found {len(pdf_files)} PDFs in {input_dir}")
        results = [self.process_pdf(p) for p in progress_iter(pdf_files, desc="Processing PDFs")]

        ok = sum(1 for r in results if r["status"] == "success")
        print_success(f"{ok}/{len(results)} completed")
        if ok < len(results):
            print_error(f"{len(results) - ok} failed")

        return results
