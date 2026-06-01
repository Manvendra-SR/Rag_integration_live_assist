"""CLI: enrich structural chunks with LLM contextual prefix + local embedding.

Reads:
  runtime/parsed_documents_fast/<stem>.parsed.json   (for parent sections)
  runtime/chunks/<stem>.chunks.json                  (atomic structural chunks)

Writes:
  runtime/chunks_enriched/<stem>.chunks.json          (chunks with text_with_context + embedding metadata)
  runtime/chunks_enriched/<stem>.vectors.npz          (bge-m3 vectors, L2-normalized)

Usage:
  .venv/bin/python scripts/enrich.py --pdf "data2/Gen AI Q_As (2).pdf"
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

import os  # noqa: E402

from chunkers.contextual_prefix import (  # noqa: E402
    build_section_lookup, enrich_chunks, save_enriched,
    OLLAMA_MODEL, GROQ_MODEL, GROQ_RPM,
)
from chunkers.semantic_refine import refine_chunks, REFINE_THRESHOLD  # noqa: E402
from embedders.local_embed import embed_texts, save_vectors, DEFAULT_MODEL  # noqa: E402


PARSED_DIR = PROJECT_ROOT / "runtime" / "parsed_documents_fast"
CHUNKS_DIR = PROJECT_ROOT / "runtime" / "chunks"
# Pipeline A (Anthropic): structural chunks → semantic refine for oversize →
# LLM contextual prefix → embedding. Output lives in chunks_anthropic to
# distinguish from Pipeline B (pure semantic, in chunks_semantic).
ENRICHED_DIR = PROJECT_ROOT / "runtime" / "chunks_anthropic"


def resolve_paths(args):
    if args.pdf:
        stem = Path(args.pdf).stem
        return PARSED_DIR / f"{stem}.parsed.json", CHUNKS_DIR / f"{stem}.chunks.json"
    if args.parsed and args.chunks:
        return Path(args.parsed), Path(args.chunks)
    raise SystemExit("Need --pdf <pdf> OR (--parsed <p> + --chunks <c>)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--parsed", type=Path)
    ap.add_argument("--chunks", type=Path)
    ap.add_argument("--out-dir", type=Path, default=ENRICHED_DIR)
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "ollama"),
                    choices=["ollama", "groq"],
                    help="LLM provider: 'ollama' (local) or 'groq' (cloud)")
    ap.add_argument("--model", default=None,
                    help="Model name. Defaults to provider's default (ollama: llama3.1, groq: llama-3.3-70b-versatile)")
    ap.add_argument("--embedder", default=DEFAULT_MODEL,
                    help=f"Embedder model (default: {DEFAULT_MODEL})")
    ap.add_argument("--parallelism", type=int, default=None,
                    help="Concurrent workers. Auto: 6 for ollama, 4 for groq (RPM-limiter handles pacing)")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip semantic refinement of oversized chunks (>600 tokens)")
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    # Provider-specific defaults
    os.environ["LLM_PROVIDER"] = args.provider
    if args.model is None:
        args.model = GROQ_MODEL if args.provider == "groq" else OLLAMA_MODEL
    if args.parallelism is None:
        args.parallelism = 4 if args.provider == "groq" else 6

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parsed_path, chunks_path = resolve_paths(args)
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    chunks_blob = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = chunks_blob["chunks"]

    print("=" * 70)
    print(f"Enriching {len(chunks)} chunks from {chunks_path.name}")
    print(f"  parent doc: {parsed.get('source_file')}")
    print(f"  provider:   {args.provider}")
    print(f"  LLM model:  {args.model}")
    if args.provider == "groq":
        print(f"  RPM cap:    {GROQ_RPM}")
    print(f"  embedder:   {args.embedder}")
    print(f"  workers:    {args.parallelism}")
    print(f"  refinement: {'OFF' if args.no_refine else f'ON (threshold {REFINE_THRESHOLD} tok)'}")
    print("=" * 70)

    # --- Semantic refinement of oversized chunks ---
    if not args.no_refine:
        print(f"\n▶ Semantic refinement of chunks > {REFINE_THRESHOLD} tokens ...")
        t_refine = time.perf_counter()
        # bge-m3 wrapper that returns numpy array
        def _refine_embedder(sents):
            return embed_texts(sents, model_name=args.embedder, batch_size=16)
        chunks = refine_chunks(chunks, _refine_embedder)
        refine_time = time.perf_counter() - t_refine
        print(f"  refinement done in {refine_time:.1f}s — new chunk count: {len(chunks)}")
    else:
        refine_time = 0.0

    # Build section lookup once
    section_lookup = build_section_lookup(parsed)
    print(f"\nSection lookup built: {len(section_lookup)} first-level sections")

    # --- Enrich with LLM prefix ---
    print("\n▶ Generating contextual prefixes via local LLM ...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(
        chunks=chunks,
        section_lookup=section_lookup,
        doc_title=parsed.get("source_file", ""),
        model=args.model,
        parallelism=args.parallelism,
    )
    enrich_time = time.perf_counter() - t0

    out_stem = Path(parsed.get("source_file", parsed_path.stem)).stem
    enriched_path = args.out_dir / f"{out_stem}.chunks.json"
    save_enriched(enriched, enriched_path)
    print(f"\n✓ Wrote enriched chunks: {enriched_path}")

    # --- Embed ---
    if args.skip_embed:
        print("Skipping embedding (--skip-embed)")
        return

    print("\n▶ Embedding prefixed chunks with bge-m3 (local, MPS) ...")
    texts = [c["text_with_context"] for c in enriched]
    chunk_ids = [c["chunk_id"] for c in enriched]
    t1 = time.perf_counter()
    vectors = embed_texts(texts, model_name=args.embedder)
    embed_time = time.perf_counter() - t1

    vec_path = args.out_dir / f"{out_stem}.vectors.npz"
    save_vectors(vectors, chunk_ids, vec_path, model_name=args.embedder)
    print(f"\n✓ Wrote vectors: {vec_path}  shape={vectors.shape}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("DONE")
    print(f"  refinement time: {refine_time:.1f}s")
    print(f"  enrichment time: {enrich_time:.1f}s")
    print(f"  embedding time:  {embed_time:.1f}s")
    print(f"  chunks enriched: {len(enriched)}")
    print(f"  vector shape:    {vectors.shape}")
    print(f"  outputs:         {enriched_path}")
    print(f"                   {vec_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
