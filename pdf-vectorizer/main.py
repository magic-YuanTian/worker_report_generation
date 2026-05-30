#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

from src.utils import print_info, print_success, print_error
from extract import run_extract
from chunk import run_chunk
from vect import run_embed


def main():
    parser = argparse.ArgumentParser(
        description="PDF Text Extraction, Chunking, and Vectorization Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("check", help="Verify imports and config")
    p.add_argument("--config", type=Path)

    p = sub.add_parser("extract", help="Extract text from a PDF or directory")
    p.add_argument("input", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("-o", "--output", type=Path, help="Override output path (single file only)")

    p = sub.add_parser("chunk", help="Chunk text.json → chunks.json")
    p.add_argument("input", type=Path, help="PDF file or directory (used to locate text.json)")
    p.add_argument("--config", type=Path)
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("embed", help="Embed chunks.json → embeddings.npz")
    p.add_argument("input", type=Path, help="PDF file or directory (used to locate chunks.json)")
    p.add_argument("--config", type=Path)
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("run", help="Full pipeline: extract → chunk → embed")
    p.add_argument("input", type=Path, help="PDF file or directory")
    p.add_argument("--config", type=Path)
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("info", help="Show current configuration")
    p.add_argument("--config", type=Path)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    try:
        return {
            "check":   run_check,
            "extract": _run_extract_cmd,
            "chunk":   _run_chunk_cmd,
            "embed":   _run_embed_cmd,
            "run":     run_pipeline,
            "info":    run_info,
        }[args.command](args)
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def _run_extract_cmd(args):
    import types
    input_path = args.input
    if input_path.is_file():
        return run_extract(args)
    if input_path.is_dir():
        pdfs = sorted(input_path.rglob("*.pdf") if args.recursive else input_path.glob("*.pdf"))
        if not pdfs:
            print_error(f"No PDFs found in {input_path}")
            return 1
        ok = 0
        for pdf in pdfs:
            print_info(f"\n--- {pdf.name} ---")
            single = types.SimpleNamespace(input=pdf, config=args.config, output=None)
            if run_extract(single) == 0:
                ok += 1
        print_success(f"\nExtracted {ok}/{len(pdfs)} PDFs")
        return 0 if ok == len(pdfs) else 1
    print_error(f"Path not found: {input_path}")
    return 1


def _run_chunk_cmd(args):
    import types
    input_path = args.input
    if input_path.is_file():
        return run_chunk(args)
    if input_path.is_dir():
        pdfs = sorted(input_path.rglob("*.pdf") if args.recursive else input_path.glob("*.pdf"))
        if not pdfs:
            print_error(f"No PDFs found in {input_path}")
            return 1
        ok = 0
        for pdf in pdfs:
            print_info(f"\n--- {pdf.name} ---")
            single = types.SimpleNamespace(input=pdf, config=args.config)
            if run_chunk(single) == 0:
                ok += 1
        print_success(f"\nChunked {ok}/{len(pdfs)} PDFs")
        return 0 if ok == len(pdfs) else 1
    print_error(f"Path not found: {input_path}")
    return 1


def _run_embed_cmd(args):
    import types
    input_path = args.input
    if input_path.is_file():
        return run_embed(args)
    if input_path.is_dir():
        pdfs = sorted(input_path.rglob("*.pdf") if args.recursive else input_path.glob("*.pdf"))
        if not pdfs:
            print_error(f"No PDFs found in {input_path}")
            return 1
        ok = 0
        for pdf in pdfs:
            print_info(f"\n--- {pdf.name} ---")
            single = types.SimpleNamespace(input=pdf, config=args.config)
            if run_embed(single) == 0:
                ok += 1
        print_success(f"\nEmbedded {ok}/{len(pdfs)} PDFs")
        return 0 if ok == len(pdfs) else 1
    print_error(f"Path not found: {input_path}")
    return 1


def run_check(args):
    from src.configuration import load_config, get_paths_config
    print_success("  src.configuration  OK")

    from src.pdf_extractor import extract_text_from_pdf  # noqa: F401
    print_success("  src.pdf_extractor  OK")

    print_info("\nConfig paths:")
    config = load_config(args.config)
    for name, path in get_paths_config(config).items():
        print_info(f"  {name}: {path}")

    print_success("\nAll checks passed.")
    return 0


def run_pipeline(args):
    from src.pipeline import PDFVectorizationPipeline

    input_path = args.input
    if not input_path.exists():
        print_error(f"Path not found: {input_path}")
        return 1

    pipeline = PDFVectorizationPipeline(args.config)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print_error("Input must be a PDF file")
            return 1
        result = pipeline.process_pdf(input_path)
        if result["status"] == "success":
            print_success("Processing complete!")
            print_info(f"Embeddings: {result['embeddings_file']}")
            return 0
        print_error(f"Failed: {result.get('error', 'unknown error')}")
        return 1

    if input_path.is_dir():
        results = pipeline.process_directory(input_path, recursive=args.recursive)
        ok = sum(1 for r in results if r["status"] == "success")
        return 0 if ok == len(results) else 1

    print_error(f"Invalid path: {input_path}")
    return 1


def run_info(args):
    from src.configuration import (
        load_config, get_paths_config, get_extraction_config,
        get_chunking_config, get_vectorization_config,
    )
    config = load_config(args.config)
    sections = {
        "Paths":         get_paths_config(config),
        "Extraction":    get_extraction_config(config),
        "Chunking":      get_chunking_config(config),
        "Vectorization": get_vectorization_config(config),
    }
    for title, data in sections.items():
        print_info(f"\n{title}:")
        for k, v in data.items():
            print_info(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
