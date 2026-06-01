"""CLI: structural chunking on a Docling parsed JSON.

Reads:   runtime/parsed_documents_fast/<stem>.parsed.json
Writes:  runtime/chunks/<stem>.chunks.json

Usage:
  .venv/bin/python scripts/chunk.py --parsed "runtime/parsed_documents_fast/Gen AI Q_As (2).parsed.json"
  .venv/bin/python scripts/chunk.py --pdf "data2/Gen AI Q_As (2).pdf"     # infers parsed path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chunkers.structural_chunker import chunk_parsed_doc, save_chunks  # noqa: E402

PARSED_DEFAULT_DIR = PROJECT_ROOT / "runtime" / "parsed_documents_fast"
CHUNKS_OUT_DIR = PROJECT_ROOT / "runtime" / "chunks"


def resolve_parsed_path(args: argparse.Namespace) -> Path:
    if args.parsed:
        return Path(args.parsed).expanduser().resolve()
    if args.pdf:
        stem = Path(args.pdf).stem
        return PARSED_DEFAULT_DIR / f"{stem}.parsed.json"
    raise SystemExit("Provide --parsed <path> or --pdf <pdf path>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, help="Path to <stem>.parsed.json")
    ap.add_argument("--pdf", type=Path, help="Source PDF (infers parsed path)")
    ap.add_argument("--out-dir", type=Path, default=CHUNKS_OUT_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parsed_path = resolve_parsed_path(args)
    if not parsed_path.exists():
        raise SystemExit(f"Parsed JSON not found: {parsed_path}\n"
                         f"  Run `scripts/parse.py` first.")

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    print(f"▶ Structural chunking: {parsed_path.name}")
    print(f"  doc_id    = {parsed.get('doc_id')}")
    print(f"  source    = {parsed.get('source_file')}")
    print(f"  blocks    = {len(parsed.get('blocks', []))}")

    t0 = time.perf_counter()
    chunks = chunk_parsed_doc(parsed)
    elapsed = time.perf_counter() - t0

    out_name = f"{Path(parsed.get('source_file', parsed_path.stem)).stem}.chunks.json"
    out_path = args.out_dir / out_name
    save_chunks(chunks, out_path)

    # Stats
    sizes = sorted(c.token_count for c in chunks)
    total = sum(sizes)
    median = sizes[len(sizes) // 2] if sizes else 0
    mean = total // max(len(sizes), 1)
    under_floor = sum(1 for s in sizes if s < 100)
    over_1k = sum(1 for s in sizes if s > 1000)

    print()
    print(f"✓ Wrote {out_path}")
    print(f"  chunks: {len(chunks)}  |  tokens: {total:,}  |  time: {elapsed:.2f}s")
    print(f"  token dist — min {sizes[0]}  median {median}  mean {mean}  max {sizes[-1]}")
    print(f"  under 100 tok: {under_floor}  |  over 1000 tok: {over_1k}")


if __name__ == "__main__":
    main()
