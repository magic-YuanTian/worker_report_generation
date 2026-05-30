#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

from src.utils import print_info, print_success, print_error


def run_extract(args):
    from src.configuration import load_config, get_extraction_config, get_paths_config
    from src.pdf_extractor import extract_text_from_pdf, get_pdf_info

    pdf_path = args.input
    if not pdf_path.exists():
        print_error(f"File not found: {pdf_path}")
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print_error("Input must be a .pdf file")
        return 1

    config = load_config(args.config)
    ext_cfg = get_extraction_config(config)

    info = get_pdf_info(pdf_path)
    print_info(f"File  : {pdf_path.name}")
    print_info(f"Pages : {info.get('pages', '?')}")
    print_info(f"Method: {ext_cfg.get('method', 'pymupdf')}")

    processed_root = Path(get_paths_config(config).get("processed", "data/processed"))
    figures_dir = processed_root / pdf_path.stem / "figures"

    pages = extract_text_from_pdf(
        pdf_path,
        method=ext_cfg.get("method", "pymupdf"),
        figures_dir=figures_dir,
        min_figure_px=ext_cfg.get("min_figure_px", 50),
    )

    total_chars = sum(len(p.get("text", "")) for p in pages)
    print_success(f"Extracted {len(pages)} pages, {total_chars:,} characters")

    out_path = args.output or (processed_root / pdf_path.stem / "text.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    print_info(f"Saved to: {out_path}")

    fig_count = sum(len(p.get("figures", [])) for p in pages)
    if fig_count:
        print_success(f"Saved {fig_count} figure(s) to: {out_path.parent / 'figures'}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF")
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Override output path")
    args = parser.parse_args()
    return run_extract(args)


if __name__ == "__main__":
    sys.exit(main())
