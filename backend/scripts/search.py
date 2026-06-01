"""CLI: query the indexed corpus via semantic / BM25 / hybrid / reranked.

Examples
--------
  # Pure cosine (Weaviate) over all docs
  .venv/bin/python scripts/search.py --query "what is DCGAN used for?" --mode semantic

  # BM25 only, restricted to one doc
  .venv/bin/python scripts/search.py --query "ResNet accuracy" --mode bm25 \
      --pipeline anthropic --doc-id paper2

  # Hybrid (semantic + BM25 → RRF fusion)
  .venv/bin/python scripts/search.py --query "Comfort Plan SIP" --mode hybrid --pipeline anthropic

  # Hybrid + cross-encoder rerank (final top-k after rerank)
  .venv/bin/python scripts/search.py --query "Comfort Plan SIP" --mode reranked --top-k 5

`--pipeline` picks which BM25 index to use (anthropic vs semantic) — the
Weaviate vector store is shared across both pipelines (each chunk carries
chunking_strategy=anthropic|semantic), and the `--pipeline` flag is applied as
a Weaviate property filter too so each mode stays in its lane.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.semantic_search import search_semantic    # noqa: E402
from retrieval.bm25_index import search_bm25             # noqa: E402
from retrieval.hybrid_search import search_hybrid        # noqa: E402
from retrieval.reranker import build_reranker, RERANK_CANDIDATE_LIMIT, RERANK_FINAL_LIMIT  # noqa: E402
from retrieval.context_assembler import assemble_context  # noqa: E402
from generation.prompt_builder import build_prompt_package  # noqa: E402
from generation.generator import generate_answer  # noqa: E402

BM25_OUT = PROJECT_ROOT / "runtime" / "bm25"
SEARCH_RUNS_DIR = PROJECT_ROOT / "runtime" / "search_runs"


def _show(results: list[dict], mode: str, top_k: int) -> None:
    print(f"\n=== {mode.upper()}  top {min(top_k, len(results))} of {len(results)} ===")
    for i, r in enumerate(results[:top_k], 1):
        src   = r.get("source_filename") or r.get("doc_id")
        sec   = " > ".join(r.get("section_path") or []) or "—"
        pgs   = f"p{r.get('page_start')}-{r.get('page_end')}"
        score = (r.get("rerank_score") or r.get("rrf_score")
                 or r.get("semantic_score") or r.get("bm25_score") or 0)
        snippet = (r.get("text") or "").strip().replace("\n", " ")[:140]
        print(f"  [{i:>2}] score={score:.4f}  {src}  {pgs}  {sec}")
        if r.get("sources"):
            print(f"        sources={r['sources']}  ranks={r.get('source_ranks')}")
        print(f"        text: {snippet}...")


def _slug(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return (value or "query")[:max_len].strip("_") or "query"


def _default_run_dir(query: str, pipeline: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SEARCH_RUNS_DIR / f"{stamp}_{pipeline}_{_slug(query)}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_answer(query: str, results: list[dict]) -> dict:
    context_payload = assemble_context({"query": query, "results": results})
    prompt_package = build_prompt_package(context_payload)
    result = generate_answer(
        system_prompt=prompt_package["system_prompt"],
        user_prompt=prompt_package["user_prompt"],
    )
    return {
        "provider": result.provider,
        "model": result.model,
        "answer": result.answer_payload.get("answer", ""),
        "answer_payload": result.answer_payload,
        "raw_text": result.raw_text,
        "fallback_reason": result.fallback_reason,
        "context": context_payload,
        "prompt_package": prompt_package,
        "prompt": (
            f"SYSTEM:\n{prompt_package['system_prompt']}\n\n"
            f"USER:\n{prompt_package['user_prompt']}"
        ),
    }


def _save_run_artifacts(
    *,
    run_dir: Path,
    query: str,
    mode: str,
    pipeline: str,
    filters: dict,
    bm25_results: list[dict] | None = None,
    semantic_results: list[dict] | None = None,
    rrf_results: list[dict] | None = None,
    reranker_results: list[dict] | None = None,
    final_results: list[dict] | None = None,
    generation: dict | None = None,
) -> None:
    meta = {
        "query": query,
        "mode": mode,
        "pipeline": pipeline,
        "filters": filters,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json(run_dir / "meta.json", meta)
    if bm25_results is not None:
        _write_json(run_dir / "bm25" / "results.json", {**meta, "results": bm25_results})
    if semantic_results is not None:
        _write_json(run_dir / "semantic" / "results.json", {**meta, "results": semantic_results})
    if rrf_results is not None:
        _write_json(run_dir / "rrf" / "results.json", {**meta, "results": rrf_results})
    if reranker_results is not None:
        _write_json(run_dir / "reranker" / "results.json", {**meta, "results": reranker_results})
    if final_results is not None:
        _write_json(run_dir / "final" / "results.json", {**meta, "results": final_results})
    if generation is not None:
        gen_dir = run_dir / "generation"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "prompt.txt").write_text(generation["prompt"], encoding="utf-8")
        _write_json(gen_dir / "context.json", generation.get("context", {}))
        _write_json(gen_dir / "prompt_package.json", generation.get("prompt_package", {}))
        _write_json(gen_dir / "answer.json", {
            **meta,
            "provider": generation.get("provider"),
            "model": generation.get("model"),
            "answer": generation.get("answer", ""),
            "answer_payload": generation.get("answer_payload"),
            "raw_text": generation.get("raw_text"),
            "fallback_reason": generation.get("fallback_reason"),
            "error": generation.get("error"),
            "prompt_path": str(gen_dir / "prompt.txt"),
            "context_path": str(gen_dir / "context.json"),
            "prompt_package_path": str(gen_dir / "prompt_package.json"),
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--mode", default="hybrid",
                    choices=["semantic", "bm25", "hybrid", "reranked"])
    ap.add_argument("--pipeline", default="anthropic",
                    choices=["anthropic", "semantic"],
                    help="Which chunking pipeline to query")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--pipeline-filename", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to dump the results JSON")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Directory for stage artefacts. Default: runtime/search_runs/<timestamp>_<query>")
    ap.add_argument("--no-generate", action="store_true",
                    help="Skip final answer generation. Reranked mode generates by default.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    filters: dict = {"chunking_strategy": args.pipeline}
    if args.doc_id:
        filters["doc_id"] = args.doc_id
    if args.pipeline_filename:
        filters["source_filename"] = args.pipeline_filename

    bm25_index = BM25_OUT / f"{args.pipeline}.pkl"
    bm25_docs  = BM25_OUT / f"{args.pipeline}.docs.json"
    run_dir = args.run_dir or _default_run_dir(args.query, args.pipeline)
    bm25_stage = None
    semantic_stage = None
    rrf_stage = None
    reranker_stage = None
    generation = None

    if args.mode == "semantic":
        results = search_semantic(args.query, limit=args.top_k,
                                  filters=filters,
                                  collection_name=args.collection)
        semantic_stage = results

    elif args.mode == "bm25":
        if not bm25_index.exists():
            raise SystemExit(f"BM25 index missing: {bm25_index} "
                             f"— run scripts/build_bm25.py --pipeline {args.pipeline}")
        results = search_bm25(args.query, bm25_index, bm25_docs,
                              limit=args.top_k, filters=filters)
        bm25_stage = results

    elif args.mode == "hybrid":
        if not bm25_index.exists():
            raise SystemExit(f"BM25 index missing: {bm25_index} "
                             f"— run scripts/build_bm25.py --pipeline {args.pipeline}")
        out = search_hybrid(args.query, bm25_index, bm25_docs,
                            limit=args.top_k, filters=filters,
                            collection_name=args.collection)
        bm25_stage = out.get("bm25_results", [])
        semantic_stage = out.get("semantic_results", [])
        rrf_stage = out["results"]
        results = out["results"]
        print(f"  bm25={out['bm25_count']}  semantic={out['semantic_count']}  "
              f"fused_top={len(results)}")

    elif args.mode == "reranked":
        if not bm25_index.exists():
            raise SystemExit(f"BM25 index missing: {bm25_index} "
                             f"— run scripts/build_bm25.py --pipeline {args.pipeline}")
        candidate_limit = max(RERANK_CANDIDATE_LIMIT, args.top_k * 4)
        out = search_hybrid(args.query, bm25_index, bm25_docs,
                            limit=candidate_limit, filters=filters,
                            collection_name=args.collection)
        bm25_stage = out.get("bm25_results", [])
        semantic_stage = out.get("semantic_results", [])
        rrf_stage = [dict(r) for r in out["results"]]
        print(f"  hybrid candidates: {len(out['results'])} → reranking…")
        rr = build_reranker()
        results = rr.rerank(args.query, out["results"], top_k=args.top_k)
        reranker_stage = results
        if not args.no_generate:
            print("  generating answer from reranked chunks…")
            try:
                generation = _generate_answer(args.query, results)
            except Exception as exc:
                generation = {
                    "model": None,
                    "prompt": "",
                    "answer": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logging.exception("generation failed")
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    _show(results, args.mode, args.top_k)

    if generation and generation.get("answer"):
        print("\n=== ANSWER ===")
        print(generation["answer"])
        citations = (generation.get("answer_payload") or {}).get("citations") or []
        if citations:
            print("\nCitations:")
            for i, c in enumerate(citations, start=1):
                print(f"  [{i}] {c.get('source_filename')} p{c.get('page')} — {c.get('section')}")

    if args.mode == "reranked":
        _save_run_artifacts(
            run_dir=run_dir,
            query=args.query,
            mode=args.mode,
            pipeline=args.pipeline,
            filters=filters,
            bm25_results=bm25_stage,
            semantic_results=semantic_stage,
            rrf_results=rrf_stage,
            reranker_results=reranker_stage,
            final_results=results,
            generation=generation,
        )
        print(f"\nSaved search run: {run_dir}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"query": args.query, "mode": args.mode,
                        "source": args.pipeline, "filters": filters,
                        "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
