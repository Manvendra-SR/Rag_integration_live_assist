"""CLI: build a BM25 index over every chunks.json in a source dir.

Reads:
  runtime/chunks_anthropic/*.chunks.json   OR
  runtime/chunks_semantic/*.chunks.json

Writes:
  runtime/bm25/<source>.pkl   (BM25Okapi index)
  runtime/bm25/<source>.docs.json (chunk_id-keyed lookup for search results)

Usage:
  .venv/bin/python scripts/build_bm25.py --pipeline anthropic
  .venv/bin/python scripts/build_bm25.py --pipeline semantic
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import build_bm25_from_dir  # noqa: E402

SOURCE_DIRS = {
    "anthropic": PROJECT_ROOT / "runtime" / "chunks_anthropic",
    "semantic":  PROJECT_ROOT / "runtime" / "chunks_semantic",
}
BM25_OUT = PROJECT_ROOT / "runtime" / "bm25"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=list(SOURCE_DIRS.keys()))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    src = SOURCE_DIRS[args.pipeline]
    if not src.exists():
        raise SystemExit(f"Source dir does not exist: {src}")
    index_path = BM25_OUT / f"{args.pipeline}.pkl"
    docs_path  = BM25_OUT / f"{args.pipeline}.docs.json"
    stats = build_bm25_from_dir(src, index_path, docs_path)
    print(f"✓ BM25 index:  {stats['index_path']}")
    print(f"✓ BM25 docs:   {stats['docs_path']}")
    print(f"  chunks={stats['chunk_count']}  docs={stats['docs']}  "
          f"elapsed={stats['elapsed_s']}s")


if __name__ == "__main__":
    main()
