"""CLI: pure semantic recursive chunking (Pipeline B).

Reads:   runtime/parsed_research_papers/<stem>.parsed.json
         (or runtime/parsed_documents_fast/<stem>.parsed.json)
         OR runtime/chunks/<stem>.chunks.json
Writes:  runtime/chunks_semantic/<stem>.chunks.json
         runtime/chunks_semantic/<stem>.vectors.npz

Token bounds (user spec):
  MIN = 100 tokens  (smaller groups merge into previous)
  MAX = 700 tokens  (anything bigger recursively splits at cosine valley)

Embedding provider follows .env (EMBEDDING_PROVIDER / EMBEDDING_MODEL / KEY)
via clients.model_factory.get_embedder() — same switch as everywhere else.

Usage:
  .venv/bin/python scripts/chunk_semantic.py --pdf "data2/paper2.pdf"
  .venv/bin/python scripts/chunk_semantic.py --parsed "runtime/parsed_research_papers/paper2.parsed.json"
  .venv/bin/python scripts/chunk_semantic.py --chunks "runtime/chunks/paper2.chunks.json"
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

from chunkers.semantic_chunker import (  # noqa: E402
    chunk_parsed_doc_semantic, chunk_structural_chunks_semantic,
    save_semantic_chunks, MIN_TOKENS, MAX_TOKENS,
)
from embedders.local_embed import save_vectors  # noqa: E402
from clients import get_embedder  # noqa: E402

PARSED_DEFAULT_DIRS = [
    PROJECT_ROOT / "runtime" / "parsed_research_papers",
    PROJECT_ROOT / "runtime" / "parsed_documents_fast",
    PROJECT_ROOT / "runtime" / "parsed_documents",
]
OUT_DIR = PROJECT_ROOT / "runtime" / "chunks_semantic"


def _find_parsed(stem: str) -> Path | None:
    for d in PARSED_DEFAULT_DIRS:
        p = d / f"{stem}.parsed.json"
        if p.exists():
            return p
    return None


def _chunks_stem(path: Path) -> str:
    if path.name.endswith(".chunks.json"):
        return path.name.removesuffix(".chunks.json")
    return path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, help="Path to <stem>.parsed.json")
    ap.add_argument("--chunks", type=Path,
                    help="Path to structural <stem>.chunks.json from scripts/chunk.py")
    ap.add_argument("--pdf", type=Path, help="Source PDF (infers parsed path)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-embed", action="store_true",
                    help="Skip the final embedding pass (chunks-only output)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.chunks:
        chunks_path = Path(args.chunks).expanduser().resolve()
        if not chunks_path.exists():
            raise SystemExit(f"Chunks JSON not found: {chunks_path}")
        chunks_blob = json.loads(chunks_path.read_text(encoding="utf-8"))
        print(f"▶ Semantic chunking from structural chunks "
              f"(recursive, MIN={MIN_TOKENS} MAX={MAX_TOKENS}): {chunks_path.name}")
        print(f"  source chunks = {len(chunks_blob.get('chunks', []))}")

        t0 = time.perf_counter()
        chunks = chunk_structural_chunks_semantic(chunks_blob)
        dt = time.perf_counter() - t0
        stem = _chunks_stem(chunks_path)
    elif args.parsed:
        parsed_path = Path(args.parsed).expanduser().resolve()
        if not parsed_path.exists():
            raise SystemExit(f"Parsed JSON not found: {parsed_path}")
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        print(f"▶ Semantic chunking from parsed document "
              f"(recursive, MIN={MIN_TOKENS} MAX={MAX_TOKENS}): {parsed_path.name}")
        print(f"  blocks = {len(parsed.get('blocks', []))}")

        t0 = time.perf_counter()
        chunks = chunk_parsed_doc_semantic(parsed)
        dt = time.perf_counter() - t0
        stem = Path(parsed.get("source_file", parsed_path.stem)).stem
    elif args.pdf:
        stem = Path(args.pdf).stem
        parsed_path = _find_parsed(stem)
        if parsed_path is None:
            raise SystemExit(f"No parsed.json found for stem {stem!r} under any of: "
                             f"{[str(d) for d in PARSED_DEFAULT_DIRS]}\n"
                             f"Run scripts/parse.py first.")
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        print(f"▶ Semantic chunking from parsed document "
              f"(recursive, MIN={MIN_TOKENS} MAX={MAX_TOKENS}): {parsed_path.name}")
        print(f"  blocks = {len(parsed.get('blocks', []))}")

        t0 = time.perf_counter()
        chunks = chunk_parsed_doc_semantic(parsed)
        dt = time.perf_counter() - t0
        stem = Path(parsed.get("source_file", parsed_path.stem)).stem
    else:
        raise SystemExit("Provide --chunks <path>, --parsed <path>, or --pdf <pdf path>")

    out_path = args.out_dir / f"{stem}.chunks.json"
    save_semantic_chunks(chunks, out_path)

    sizes = sorted(c.token_count for c in chunks)
    if not sizes:
        print("(no chunks produced)")
        return
    print()
    print(f"✓ Wrote {out_path}")
    print(f"  chunks  : {len(chunks)} | total tokens: {sum(sizes):,} | time: {dt:.2f}s")
    print(f"  tok dist: min {sizes[0]}  median {sizes[len(sizes)//2]}  "
          f"mean {sum(sizes)//len(sizes)}  max {sizes[-1]}")
    print(f"  under {MIN_TOKENS}: {sum(1 for s in sizes if s < MIN_TOKENS)}  |  "
          f"over {MAX_TOKENS}: {sum(1 for s in sizes if s > MAX_TOKENS)}")
    coh = [c.coherence_score for c in chunks]
    if coh:
        print(f"  avg coherence: {sum(coh)/len(coh):.3f}")

    if args.no_embed:
        return

    # ---- Embedding pass ----
    print()
    print(f"▶ Embedding {len(chunks)} chunks via factory…")
    emb = get_embedder()
    print(f"  embedder: {emb.name}  (dim={emb.dim})")
    t0 = time.perf_counter()
    vecs_list = emb.embed([c.text for c in chunks])
    import numpy as np
    vecs = np.asarray(vecs_list, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    vec_path = args.out_dir / f"{stem}.vectors.npz"
    save_vectors(vecs, [c.chunk_id for c in chunks], vec_path, model_name=emb.name)
    print(f"  embedded in {time.perf_counter()-t0:.1f}s  →  {vec_path}")


if __name__ == "__main__":
    main()
