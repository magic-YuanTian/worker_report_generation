#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

from src.utils import print_info, print_success, print_error


def run_embed(args):
    print_info("Initialising embed step...")

    from src.configuration import load_config, get_vectorization_config, get_paths_config
    print_info("Config loaded.")

    config = load_config(args.config)
    out_dir = Path(get_paths_config(config).get("processed", "data/processed")) / args.input.stem
    chunks_path = out_dir / "chunks.json"

    if not chunks_path.exists():
        print_error(f"chunks.json not found at {chunks_path} — run chunk first")
        return 1

    chunks_raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    print_info(f"Loaded {len(chunks_raw)} chunks from {chunks_path.name}")

    print_info("Loading ML stack (torch + sentence-transformers) — this takes ~20s on first call...")
    from src.text_chunker import Chunk
    from src.text_vectorizer import vectorize_chunks, save_embeddings
    print_info("ML stack ready.")

    chunks = [Chunk(**d) for d in chunks_raw]

    cfg = get_vectorization_config(config)
    fmt = cfg.get("output_format", "numpy")
    model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
    batch_size = cfg.get("batch_size", 32)
    embeddings_path = out_dir / ("embeddings.npz" if fmt == "numpy" else "embeddings.json")

    print_info(f"Model     : {model_name}")
    print_info(f"Batch size: {batch_size}")
    print_info(f"Output    : {embeddings_path.name}")

    embeddings = vectorize_chunks(
        chunks,
        model_type=cfg.get("model_type", "sentence_transformers"),
        model_name=model_name,
        batch_size=batch_size,
        openai_api_key=cfg.get("openai_api_key"),
    )
    save_embeddings(embeddings, chunks, embeddings_path, format=fmt,
                    include_metadata=cfg.get("include_metadata", True))

    from src.utils import format_size, get_file_size
    print_success(f"Done — {len(embeddings)} embeddings, dim {embeddings.shape[1]}, "
                  f"{format_size(get_file_size(embeddings_path))}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Embed chunks.json → embeddings.npz")
    parser.add_argument("input", type=Path, help="Original PDF path (used to locate chunks.json)")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    return run_embed(args)


if __name__ == "__main__":
    sys.exit(main())
