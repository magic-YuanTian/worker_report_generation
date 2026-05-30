#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

from src.utils import print_info, print_success, print_error


def run_chunk(args):
    from src.configuration import load_config, get_chunking_config, get_paths_config
    from src.text_chunker import chunk_text, get_chunk_statistics

    config = load_config(args.config)
    out_dir = Path(get_paths_config(config).get("processed", "data/processed")) / args.input.stem
    text_path = out_dir / "text.json"
    chunks_path = out_dir / "chunks.json"

    if not text_path.exists():
        print_error(f"text.json not found at {text_path} — run extract first")
        return 1

    pages = json.loads(text_path.read_text(encoding="utf-8"))
    cfg = get_chunking_config(config)
    print_info(f"Chunking {args.input.stem} ({len(pages)} pages)...")

    chunks = chunk_text(
        pages,
        strategy=cfg.get("strategy", "recursive"),
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
    )

    stats = get_chunk_statistics(chunks)
    print_success(f"Created {stats['total_chunks']} chunks, avg {stats['avg_length']:.0f} chars")

    chunks_path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_info(f"Saved to: {chunks_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Chunk text.json → chunks.json")
    parser.add_argument("input", type=Path, help="Original PDF path (used to locate text.json)")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    return run_chunk(args)


if __name__ == "__main__":
    sys.exit(main())
