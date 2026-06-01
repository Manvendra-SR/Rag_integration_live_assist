"""Full end-to-end Anthropic-style pipeline run, with checkpointed outputs.

Saves at every stage so you can inspect each handoff:
  1_parsed.json              ← Docling parser output
  2_structural_chunks.json   ← after structural chunking (merge upward, no refinement yet)
  3_chunks_pre_llm.json      ← after semantic refinement (with overlap), ready for LLM
  4_chunks_final.json        ← after LLM contextual prefix (qwen2.5:14b)
  5_vectors.npz              ← bge-m3 embeddings of text_with_context

Run:
  .venv/bin/python scripts/run_anthropic_pipeline.py --pdf "data2/Gen AI Q_As (2).pdf"
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "anthropic_pipeline_results"


def stage_banner(num: int, name: str) -> None:
    print()
    print("=" * 78)
    print(f"STAGE {num}: {name}")
    print("=" * 78)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ saved {path.name}  ({size_kb:,.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--ollama-model", default="qwen2.5:14b",
                    help="Local Ollama model for contextual prefix")
    ap.add_argument("--parallelism", type=int, default=2,
                    help="Concurrent workers (2 recommended for qwen2.5:14b on M3 16GB)")
    ap.add_argument("--skip-parse", action="store_true",
                    help="Reuse existing 1_parsed.json if present")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.pdf.expanduser().resolve()
    pdf_name = pdf_path.stem

    # Use Ollama provider explicitly
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_MODEL"] = args.ollama_model

    print()
    print("=" * 78)
    print(f"ANTHROPIC-STYLE PIPELINE — {pdf_path.name}")
    print(f"  output dir:       {args.out_dir}")
    print(f"  Docling profile:  FAST (no VLM)")
    print(f"  floor:            100 tokens (MIN_TOKENS)")
    print(f"  refinement:       chunks > 600 tokens, sub-chunk min 100, overlap 60 tok")
    print(f"  enrichment LLM:   {args.ollama_model} (local, parallel={args.parallelism})")
    print(f"  embedder:         BAAI/bge-m3 (local, MPS)")
    print("=" * 78)

    timings = {}

    # -------------- STAGE 1: PARSE --------------
    stage_banner(1, "Docling parse → 1_parsed.json")
    parsed_path = args.out_dir / "1_parsed.json"
    if args.skip_parse and parsed_path.exists():
        print(f"  reusing existing {parsed_path.name}")
        parsed = json.loads(parsed_path.read_text())
    else:
        from parsers.docling_parser import parse_pdf
        t0 = time.perf_counter()
        # parse_pdf returns path; we read+resave to OUT_DIR
        tmp_path = parse_pdf(pdf_path, args.out_dir)
        parsed = json.loads(tmp_path.read_text())
        tmp_path.unlink()  # remove the auto-named file; we save canonically below
        save_json(parsed_path, parsed)
        timings["parse"] = time.perf_counter() - t0
        print(f"  pages: {parsed.get('doc_profile', {}).get('page_count')}  "
              f"blocks: {len(parsed.get('blocks', []))}  "
              f"time: {timings['parse']:.1f}s")

    # -------------- STAGE 2: STRUCTURAL CHUNKING --------------
    stage_banner(2, "Structural chunking → 2_structural_chunks.json")
    from chunkers.structural_chunker import chunk_parsed_doc, MIN_TOKENS
    t0 = time.perf_counter()
    structural = chunk_parsed_doc(parsed)
    # Convert dataclasses to dicts for JSON
    structural_dicts = [dataclasses.asdict(c) for c in structural]
    timings["structural"] = time.perf_counter() - t0
    save_json(args.out_dir / "2_structural_chunks.json", {
        "chunk_count": len(structural_dicts),
        "total_tokens": sum(c["token_count"] for c in structural_dicts),
        "min_tokens_floor": MIN_TOKENS,
        "chunks": structural_dicts,
    })
    n_with_merge = sum(1 for c in structural_dicts if c.get("merged_sections"))
    print(f"  chunks: {len(structural_dicts)}  "
          f"with merged_sections: {n_with_merge}  "
          f"time: {timings['structural']:.2f}s")
    print(f"  token dist: "
          f"min={min(c['token_count'] for c in structural_dicts)}  "
          f"median={sorted(c['token_count'] for c in structural_dicts)[len(structural_dicts)//2]}  "
          f"max={max(c['token_count'] for c in structural_dicts)}")

    # -------------- STAGE 3: SEMANTIC REFINEMENT --------------
    stage_banner(3, "Semantic refinement (>600 tok) + overlap → 3_chunks_pre_llm.json")
    from chunkers.semantic_refine import refine_chunks, REFINE_THRESHOLD, REFINE_OVERLAP_TOKENS
    from embedders.local_embed import embed_texts, DEFAULT_MODEL as EMBEDDER

    def _refine_embedder(sents):
        return embed_texts(sents, model_name=EMBEDDER, batch_size=16)

    t0 = time.perf_counter()
    refined_dicts = refine_chunks(structural_dicts, _refine_embedder)
    timings["refine"] = time.perf_counter() - t0
    save_json(args.out_dir / "3_chunks_pre_llm.json", {
        "chunk_count": len(refined_dicts),
        "total_tokens": sum(c["token_count"] for c in refined_dicts),
        "refine_threshold": REFINE_THRESHOLD,
        "refine_overlap_tokens": REFINE_OVERLAP_TOKENS,
        "chunks": refined_dicts,
    })
    n_refined_subchunks = sum(1 for c in refined_dicts if c.get("refined_from"))
    n_with_overlap = sum(1 for c in refined_dicts if c.get("overlap_with_prev_tokens", 0) > 0)
    print(f"  chunks: {len(refined_dicts)} "
          f"(refined sub-chunks: {n_refined_subchunks}, with overlap: {n_with_overlap})  "
          f"time: {timings['refine']:.1f}s")

    # -------------- STAGE 4: LLM CONTEXTUAL PREFIX --------------
    stage_banner(4, f"LLM contextual prefix ({args.ollama_model}) → 4_chunks_final.json")
    from chunkers.contextual_prefix import build_section_lookup, enrich_chunks
    section_lookup = build_section_lookup(parsed)
    print(f"  section lookup built: {len(section_lookup)} first-level sections")
    print(f"  starting LLM pass (this may take a while for {len(refined_dicts)} chunks)...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(
        chunks=refined_dicts,
        section_lookup=section_lookup,
        doc_title=parsed.get("source_file", ""),
        model=args.ollama_model,
        parallelism=args.parallelism,
    )
    timings["enrich"] = time.perf_counter() - t0
    save_json(args.out_dir / "4_chunks_final.json", {
        "chunk_count": len(enriched),
        "enrichment_model": args.ollama_model,
        "chunks": enriched,
    })
    print(f"  enriched: {len(enriched)}  time: {timings['enrich']:.1f}s "
          f"({timings['enrich']/len(enriched):.1f}s/chunk avg)")

    # -------------- STAGE 5: EMBEDDING --------------
    stage_banner(5, "bge-m3 embedding → 5_vectors.npz")
    from embedders.local_embed import save_vectors
    t0 = time.perf_counter()
    texts = [c["text_with_context"] for c in enriched]
    chunk_ids = [c["chunk_id"] for c in enriched]
    vectors = embed_texts(texts, model_name=EMBEDDER)
    timings["embed"] = time.perf_counter() - t0
    vec_path = args.out_dir / "5_vectors.npz"
    save_vectors(vectors, chunk_ids, vec_path, model_name=EMBEDDER)
    print(f"  ✓ saved {vec_path.name}  shape={vectors.shape}  "
          f"time: {timings['embed']:.1f}s")

    # -------------- SUMMARY --------------
    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"  pdf:           {pdf_path.name}")
    print(f"  output dir:    {args.out_dir}")
    print(f"  files written:")
    for f in sorted(args.out_dir.iterdir()):
        print(f"    {f.name:<30}  {f.stat().st_size/1024:>8,.0f} KB")
    print()
    print(f"  stage timings:")
    for k, v in timings.items():
        print(f"    {k:<12} {v:>8.1f}s")
    print(f"    {'TOTAL':<12} {sum(timings.values()):>8.1f}s ({sum(timings.values())/60:.1f} min)")
    print("=" * 78)


if __name__ == "__main__":
    main()
