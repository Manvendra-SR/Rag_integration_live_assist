"""
pipeline.py — Importable RAG Pipeline API
==========================================

Drop this file into the Advanced_RAG_pipeline project root.
Your backend imports two functions:

    from pipeline import ingest_pdf, query

INGEST (run once per PDF):
    result = ingest_pdf("path/to/document.pdf")
    # result = {
    #   "status": "ok",
    #   "stem": "document",
    #   "stages": { "parse": {...}, "chunk": {...}, "enrich": {...},
    #               "index": {...}, "bm25": {...} },
    #   "chunk_count": 42,
    #   "total_tokens": 18500,
    #   "elapsed_seconds": 94.2
    # }

QUERY (run anytime after ingest):
    result = query("What is DCGAN used for?")
    # result = {
    #   "answer": "DCGAN is used for...",
    #   "confidence": 0.94,
    #   "citations": [ {"source_filename": "...", "page": 3, ...} ],
    #   "mode": "reranked",
    #   "provider": "groq",
    #   "model": "llama-3.3-70b-versatile"
    # }

PROGRESS CALLBACKS:
    def my_progress(stage: str, message: str):
        print(f"[{stage}] {message}")   # or push to websocket, log, etc.

    ingest_pdf("doc.pdf", on_progress=my_progress)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ── Make sure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ── Runtime directories ───────────────────────────────────────────────────────
_PARSED_DIR   = PROJECT_ROOT / "runtime" / "parsed_documents_fast"
_CHUNKS_DIR   = PROJECT_ROOT / "runtime" / "chunks"
_ENRICHED_DIR = PROJECT_ROOT / "runtime" / "chunks_anthropic"
_BM25_DIR     = PROJECT_ROOT / "runtime" / "bm25"

_PIPELINE_NAME = "anthropic"   # the enriched pipeline used for search

# ── Type alias for the progress callback ─────────────────────────────────────
ProgressFn = Callable[[str, str], None]   # (stage_name, message) -> None


def _noop(stage: str, msg: str) -> None:    # default: do nothing
    pass


# =============================================================================
# PUBLIC API
# =============================================================================

def ingest_pdf(
    pdf_path: str | Path,
    *,
    llm_provider: str = "groq",           # "groq" | "ollama"
    embedder_model: str = "BAAI/bge-m3",
    llm_parallelism: int | None = None,   # None = auto (4 groq, 6 ollama)
    semantic_refine: bool = True,         # split oversized chunks (>600 tok)
    on_progress: ProgressFn = _noop,      # optional callback for live updates
) -> dict[str, Any]:
    """
    Run the full 5-stage ingestion pipeline on a single PDF.

    Stages
    ------
    1. Parse      — Docling PDF → structured JSON with blocks/sections
    2. Chunk      — Structural chunking (prose / table / figure chunks)
    3. Enrich     — LLM contextual prefix + bge-m3 embedding
    5. BM25       — Rebuild keyword index over all enriched chunks

    Parameters
    ----------
    pdf_path        : Path to the PDF file (absolute or relative to cwd)
    llm_provider    : LLM backend for prefix generation ("groq" or "ollama")
    embedder_model  : HuggingFace model name for embeddings
    llm_parallelism : Worker threads for prefix generation (None = auto)
    semantic_refine : Whether to semantically split oversized chunks (>600 tok)
    on_progress     : Callback fn(stage, message) called at each stage start/end.
                      Use this to push updates to a WebSocket or log stream.

    Returns
    -------
    dict with keys: status, stem, stages, chunk_count, total_tokens,
                    elapsed_seconds, error (only on failure)
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        return {"status": "error", "error": f"PDF not found: {pdf_path}"}

    stem = pdf_path.stem
    t_total = time.perf_counter()
    stages: dict[str, Any] = {}

    try:
        # ── Stage 1: Parse ────────────────────────────────────────────────────
        on_progress("parse", f"Parsing PDF: {pdf_path.name} ...")
        t0 = time.perf_counter()

        from parsers.docling_parser import parse_pdf
        parsed_path = parse_pdf(pdf_path, _PARSED_DIR)
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))

        parse_time = round(time.perf_counter() - t0, 2)
        block_count = len(parsed.get("blocks", []))
        stages["parse"] = {
            "elapsed_s": parse_time,
            "output": str(parsed_path),
            "block_count": block_count,
        }
        on_progress("parse", f"✓ Parsed {block_count} blocks in {parse_time}s → {parsed_path.name}")

        # ── Stage 2: Structural Chunk ─────────────────────────────────────────
        on_progress("chunk", "Structural chunking ...")
        t0 = time.perf_counter()

        from chunkers.structural_chunker import chunk_parsed_doc, save_chunks
        chunks = chunk_parsed_doc(parsed)
        out_name = f"{stem}.chunks.json"
        chunk_path = _CHUNKS_DIR / out_name
        save_chunks(chunks, chunk_path)

        chunk_time = round(time.perf_counter() - t0, 2)
        raw_chunks_list = [c.__dict__ if hasattr(c, "__dict__") else dict(c) for c in chunks]
        total_tokens = sum(c.token_count if hasattr(c, "token_count") else c.get("token_count", 0) for c in chunks)
        stages["chunk"] = {
            "elapsed_s": chunk_time,
            "output": str(chunk_path),
            "chunk_count": len(chunks),
            "total_tokens": total_tokens,
        }
        on_progress("chunk", f"✓ {len(chunks)} chunks ({total_tokens:,} tokens) in {chunk_time}s")

        # ── Stage 3: Enrich (LLM prefix + embed) ─────────────────────────────
        on_progress("enrich", f"Generating LLM prefixes via {llm_provider} ...")
        t0 = time.perf_counter()

        import os
        os.environ["LLM_PROVIDER"] = llm_provider

        from chunkers.contextual_prefix import (
            build_section_lookup, enrich_chunks, save_enriched,
            OLLAMA_MODEL, GROQ_MODEL,
        )
        from chunkers.semantic_refine import refine_chunks, REFINE_THRESHOLD
        from embedders.local_embed import embed_texts, save_vectors

        # Determine model + parallelism
        model = GROQ_MODEL if llm_provider == "groq" else OLLAMA_MODEL
        parallelism = llm_parallelism or (4 if llm_provider == "groq" else 6)

        # Load raw chunks as plain dicts
        chunks_blob = json.loads(chunk_path.read_text(encoding="utf-8"))
        raw_chunks = chunks_blob["chunks"]

        # Optional semantic refinement for oversized chunks
        if semantic_refine:
            on_progress("enrich", f"Semantic refinement of chunks >{REFINE_THRESHOLD} tokens ...")

            def _refine_embedder(sents):
                return embed_texts(sents, model_name=embedder_model, batch_size=16)

            raw_chunks = refine_chunks(raw_chunks, _refine_embedder)
            on_progress("enrich", f"  Refinement done → {len(raw_chunks)} chunks")

        # Build section context lookup from parsed doc
        section_lookup = build_section_lookup(parsed)

        # Generate LLM contextual prefixes (parallelised)
        on_progress("enrich", f"Generating contextual prefixes ({parallelism} workers) ...")
        enriched = enrich_chunks(
            chunks=raw_chunks,
            section_lookup=section_lookup,
            doc_title=parsed.get("source_file", ""),
            model=model,
            parallelism=parallelism,
        )

        # Save enriched chunks JSON
        _ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
        enriched_path = _ENRICHED_DIR / f"{stem}.chunks.json"
        save_enriched(enriched, enriched_path)

        # Embed with bge-m3 (or cloud provider from .env)
        on_progress("enrich", f"Embedding {len(enriched)} chunks with {embedder_model} ...")
        texts = [c["text_with_context"] for c in enriched]
        chunk_ids = [c["chunk_id"] for c in enriched]
        vectors = embed_texts(texts, model_name=embedder_model)
        vec_path = _ENRICHED_DIR / f"{stem}.vectors.npz"
        save_vectors(vectors, chunk_ids, vec_path, model_name=embedder_model)

        enrich_time = round(time.perf_counter() - t0, 2)
        stages["enrich"] = {
            "elapsed_s": enrich_time,
            "chunks_output": str(enriched_path),
            "vectors_output": str(vec_path),
            "enriched_count": len(enriched),
            "vector_shape": list(vectors.shape),
        }
        on_progress("enrich", f"✓ {len(enriched)} chunks enriched + embedded in {enrich_time}s")

        # ── Stage 4: Index into ChromaDB ─────────────────────────────────────
        on_progress("index", "Indexing chunks + vectors into ChromaDB ...")
        t0 = time.perf_counter()

        import numpy as np
        from retrieval.chroma_store import ChromaChunkStore

        # Load enriched chunks + align vectors
        payload = json.loads(enriched_path.read_text(encoding="utf-8"))
        e_chunks = payload.get("chunks", [])
        npz = np.load(vec_path, allow_pickle=True)
        raw_vecs = npz["vectors"].astype(float).tolist()
        ids_order = list(npz["chunk_ids"])
        idx_by_id = {cid: i for i, cid in enumerate(ids_order)}
        ordered_vecs = [raw_vecs[idx_by_id[c["chunk_id"]]] for c in e_chunks]

        store = ChromaChunkStore()
        index_result = store.index_chunks(
            chunks=e_chunks,
            vectors=ordered_vecs,
            source_filename=f"{stem}.pdf",
            chunking_strategy=_PIPELINE_NAME,
        )

        index_time = round(time.perf_counter() - t0, 2)
        stages["index"] = {
            "elapsed_s": index_time,
            "collection": index_result["collection"],
            "indexed": index_result["indexed"],
            "failed": index_result["failed"],
        }
        on_progress("index", f"✓ {index_result['indexed']} chunks indexed into ChromaDB in {index_time}s")

        # ── Stage 5: Rebuild BM25 Index ───────────────────────────────────────
        on_progress("bm25", "Rebuilding BM25 keyword index ...")
        t0 = time.perf_counter()

        from retrieval.bm25_index import build_bm25_from_dir

        _BM25_DIR.mkdir(parents=True, exist_ok=True)
        bm25_stats = build_bm25_from_dir(
            chunks_dir=_ENRICHED_DIR,
            index_out=_BM25_DIR / f"{_PIPELINE_NAME}.pkl",
            docs_out=_BM25_DIR / f"{_PIPELINE_NAME}.docs.json",
        )

        bm25_time = round(time.perf_counter() - t0, 2)
        stages["bm25"] = {
            "elapsed_s": bm25_time,
            "total_chunks_in_index": bm25_stats["chunk_count"],
            "docs_in_index": bm25_stats["docs"],
        }
        on_progress("bm25", f"✓ BM25 index rebuilt ({bm25_stats['chunk_count']} total chunks) in {bm25_time}s")

        # ── Final summary ─────────────────────────────────────────────────────
        total_elapsed = round(time.perf_counter() - t_total, 2)
        on_progress("done", f"✅ Ingestion complete in {total_elapsed}s — ready to query!")

        return {
            "status": "ok",
            "stem": stem,
            "source_file": str(pdf_path),
            "stages": stages,
            "chunk_count": len(enriched),
            "total_tokens": total_tokens,
            "elapsed_seconds": total_elapsed,
        }

    except Exception as exc:
        log.exception(f"Ingestion failed for {pdf_path.name}")
        return {
            "status": "error",
            "stem": stem,
            "stages": stages,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - t_total, 2),
        }


# =============================================================================

def query(
    question: str,
    *,
    mode: str = "reranked",              # "semantic" | "bm25" | "hybrid" | "reranked"
    pipeline: str = _PIPELINE_NAME,      # "anthropic" | "semantic"
    top_k: int = 5,
    doc_filter: str | None = None,       # restrict to one doc stem e.g. "paper2"
    on_progress: ProgressFn = _noop,
) -> dict[str, Any]:
    """
    Run the full retrieval + generation pipeline on a question.

    Stages (in reranked mode)
    -------------------------
    1. Embed query with bge-m3
    2. Semantic search in Weaviate  (top candidates)
    3. BM25 keyword search          (top candidates)
    4. RRF fusion of both result lists
    5. Cross-encoder reranking      (top_k final)
    6. Context assembly + dedupe + token budget
    7. Prompt construction (system + user JSON)
    8. LLM answer generation (Groq → Ollama fallback)

    Parameters
    ----------
    question    : The user's natural-language question
    mode        : Search strategy (default "reranked" = best quality)
    pipeline    : Which chunking pipeline to search ("anthropic" or "semantic")
    top_k       : Number of final chunks to retrieve
    doc_filter  : Optional — restrict search to one document by stem name
    on_progress : Callback fn(stage, message)

    Returns
    -------
    dict with keys: answer, confidence, citations, mode, provider, model,
                    retrieval_count, elapsed_seconds, error (only on failure)
    """
    if not question or not question.strip():
        return {"status": "error", "error": "Question cannot be empty"}

    t_total = time.perf_counter()

    try:
        bm25_index = _BM25_DIR / f"{pipeline}.pkl"
        bm25_docs  = _BM25_DIR / f"{pipeline}.docs.json"
        filters: dict[str, Any] = {"chunking_strategy": pipeline}
        if doc_filter:
            filters["source_filename"] = f"{doc_filter}.pdf"

        from retrieval.semantic_search import search_semantic
        from retrieval.bm25_index import search_bm25
        from retrieval.hybrid_search import search_hybrid
        from retrieval.reranker import build_reranker, RERANK_CANDIDATE_LIMIT
        from retrieval.context_assembler import assemble_context
        from generation.prompt_builder import build_prompt_package
        from generation.generator import generate_answer

        results: list[dict] = []

        # ── Retrieval ─────────────────────────────────────────────────────────
        if mode == "semantic":
            on_progress("retrieve", "Semantic vector search ...")
            results = search_semantic(question, limit=top_k, filters=filters)

        elif mode == "bm25":
            if not bm25_index.exists():
                return {
                    "status": "error",
                    "error": f"BM25 index not found: {bm25_index}. Run ingest_pdf() first.",
                }
            on_progress("retrieve", "BM25 keyword search ...")
            results = search_bm25(question, bm25_index, bm25_docs,
                                  limit=top_k, filters=filters)

        elif mode in ("hybrid", "reranked"):
            if not bm25_index.exists():
                return {
                    "status": "error",
                    "error": f"BM25 index not found: {bm25_index}. Run ingest_pdf() first.",
                }
            candidate_limit = max(RERANK_CANDIDATE_LIMIT, top_k * 4) if mode == "reranked" else top_k
            on_progress("retrieve", f"Hybrid search (semantic + BM25, {candidate_limit} candidates) ...")
            fused = search_hybrid(
                question, bm25_index, bm25_docs,
                limit=candidate_limit, filters=filters,
            )
            results = fused["results"]
            on_progress("retrieve",
                f"  semantic={fused['semantic_count']}  bm25={fused['bm25_count']}  "
                f"fused={len(results)}")

            if mode == "reranked":
                on_progress("rerank", f"Cross-encoder reranking {len(results)} candidates → top {top_k} ...")
                rr = build_reranker()
                results = rr.rerank(question, results, top_k=top_k)
                on_progress("rerank", f"✓ Reranked. Top score: {results[0].get('rerank_score', 0):.3f}" if results else "No results")

        else:
            return {"status": "error", "error": f"Unknown mode: {mode!r}"}

        if not results:
            return {
                "status": "ok",
                "answer": "No relevant content found in the indexed documents for this question.",
                "confidence": 0.0,
                "citations": [],
                "mode": mode,
                "retrieval_count": 0,
                "elapsed_seconds": round(time.perf_counter() - t_total, 2),
            }

        # ── Context assembly ──────────────────────────────────────────────────
        on_progress("generate", "Assembling context and building prompt ...")
        context_payload = assemble_context({"query": question, "results": results})
        prompt_package = build_prompt_package(context_payload)

        on_progress("generate",
            f"  context: {context_payload['selected_chunk_count']} chunks / "
            f"~{context_payload['estimated_tokens']} tokens")

        # ── LLM generation ────────────────────────────────────────────────────
        on_progress("generate", "Calling LLM for answer generation ...")
        gen = generate_answer(
            system_prompt=prompt_package["system_prompt"],
            user_prompt=prompt_package["user_prompt"],
        )

        elapsed = round(time.perf_counter() - t_total, 2)
        on_progress("done", f"✅ Answer ready in {elapsed}s (provider={gen.provider})")

        return {
            "status": "ok",
            "answer": gen.answer_payload.get("answer", ""),
            "confidence": gen.answer_payload.get("confidence", 0.0),
            "citations": gen.answer_payload.get("citations", []),
            "mode": mode,
            "provider": gen.provider,
            "model": gen.model,
            "retrieval_count": len(results),
            "context_chunks": context_payload["selected_chunk_count"],
            "estimated_context_tokens": context_payload["estimated_tokens"],
            "fallback_reason": gen.fallback_reason,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        log.exception(f"Query failed: {question!r}")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - t_total, 2),
        }
