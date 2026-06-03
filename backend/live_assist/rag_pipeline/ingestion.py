from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

from live_assist.core.config import get_settings
from live_assist.rag_pipeline.paths import (
    BM25_DIR, CHUNKS_DIR, ENRICHED_DIR, PARSED_DIR,
)

settings = get_settings()
_PIPELINE_NAME = settings.rag_pipeline_name

ProgressFn = Callable[[str, str], None]

def _noop(stage: str, msg: str) -> None:
    pass

def ingest_pdf(
    pdf_path: str | Path,
    user_id: str,
    document_id: str,
    *,
    llm_provider: str = "groq",
    embedder_model: str = "BAAI/bge-m3",
    llm_parallelism: int | None = None,
    semantic_refine: bool = True,
    on_progress: ProgressFn = _noop,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        return {"status": "error", "error": f"PDF not found: {pdf_path}"}

    stem = f"{user_id}_{document_id}"
    t_total = time.perf_counter()
    stages: dict[str, Any] = {}

    try:
        # Stage 1: Parse
        on_progress("parse", f"Parsing PDF: {pdf_path.name} ...")
        t0 = time.perf_counter()

        from live_assist.parsers.docling_parser import parse_pdf
        
        # NOTE: Docling might need unique filename logic, but we can pass the output dir and parse
        parsed_path = parse_pdf(pdf_path, PARSED_DIR)
        
        # Rename parsed path to include user_id and document_id so it's unique
        unique_parsed_path = PARSED_DIR / f"{stem}.json"
        if parsed_path.exists():
            parsed_path.rename(unique_parsed_path)
            parsed_path = unique_parsed_path

        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))

        parse_time = round(time.perf_counter() - t0, 2)
        block_count = len(parsed.get("blocks", []))
        stages["parse"] = {
            "elapsed_s": parse_time,
            "output": str(parsed_path),
            "block_count": block_count,
        }
        on_progress("parse", f"✓ Parsed {block_count} blocks in {parse_time}s → {parsed_path.name}")

        # Stage 2: Structural Chunk
        on_progress("chunk", "Structural chunking ...")
        t0 = time.perf_counter()

        from live_assist.chunkers.structural_chunker import chunk_parsed_doc, save_chunks
        chunks = chunk_parsed_doc(parsed)
        out_name = f"{stem}.chunks.json"
        chunk_path = CHUNKS_DIR / out_name
        save_chunks(chunks, chunk_path)

        chunk_time = round(time.perf_counter() - t0, 2)
        total_tokens = sum(c.token_count if hasattr(c, "token_count") else c.get("token_count", 0) for c in chunks)
        stages["chunk"] = {
            "elapsed_s": chunk_time,
            "output": str(chunk_path),
            "chunk_count": len(chunks),
            "total_tokens": total_tokens,
        }
        on_progress("chunk", f"✓ {len(chunks)} chunks ({total_tokens:,} tokens) in {chunk_time}s")

        # Stage 3: Enrich
        on_progress("enrich", f"Generating LLM prefixes via {llm_provider} ...")
        t0 = time.perf_counter()

        os.environ["LLM_PROVIDER"] = llm_provider

        from live_assist.chunkers.contextual_prefix import (
            GROQ_MODEL, OLLAMA_MODEL, build_section_lookup, enrich_chunks,
            save_enriched,
        )
        from live_assist.chunkers.semantic_refine import (
            REFINE_THRESHOLD, refine_chunks,
        )
        from live_assist.embedders.local_embed import embed_texts, save_vectors

        model = GROQ_MODEL if llm_provider == "groq" else OLLAMA_MODEL
        parallelism = llm_parallelism or (4 if llm_provider == "groq" else 6)

        chunks_blob = json.loads(chunk_path.read_text(encoding="utf-8"))
        raw_chunks = chunks_blob["chunks"]

        # Tag chunks with metadata BEFORE enrichment/embedding
        for c in raw_chunks:
            c["user_id"] = user_id
            c["document_id"] = document_id
            c["source_filename"] = pdf_path.name
            c["chunking_strategy"] = _PIPELINE_NAME

        if semantic_refine:
            on_progress("enrich", f"Semantic refinement of chunks >{REFINE_THRESHOLD} tokens ...")

            def _refine_embedder(sents):
                return embed_texts(sents, model_name=embedder_model, batch_size=16)

            raw_chunks = refine_chunks(raw_chunks, _refine_embedder)
            on_progress("enrich", f"  Refinement done → {len(raw_chunks)} chunks")

        section_lookup = build_section_lookup(parsed)

        on_progress("enrich", f"Generating contextual prefixes ({parallelism} workers) ...")
        enriched = enrich_chunks(
            chunks=raw_chunks,
            section_lookup=section_lookup,
            doc_title=pdf_path.name,
            model=model,
            parallelism=parallelism,
        )

        ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
        enriched_path = ENRICHED_DIR / f"{stem}.chunks.json"
        save_enriched(enriched, enriched_path)

        on_progress("enrich", f"Embedding {len(enriched)} chunks with {embedder_model} ...")
        texts = [c["text_with_context"] for c in enriched]
        chunk_ids = [c["chunk_id"] for c in enriched]
        vectors = embed_texts(texts, model_name=embedder_model)
        vec_path = ENRICHED_DIR / f"{stem}.vectors.npz"
        save_vectors(vectors, chunk_ids, vec_path, model_name=embedder_model)

        enrich_time = round(time.perf_counter() - t0, 2)
        stages["enrich"] = {
            "elapsed_s": enrich_time,
            "chunks_output": str(enriched_path),
            "vectors_output": str(vec_path),
            "enriched_count": len(enriched),
            "vector_shape": list(vectors.shape) if hasattr(vectors, "shape") else [],
        }
        on_progress("enrich", f"✓ {len(enriched)} chunks enriched + embedded in {enrich_time}s")

        # Stage 4: Index into ChromaDB
        on_progress("index", "Indexing chunks + vectors into ChromaDB ...")
        t0 = time.perf_counter()

        import numpy as np
        
        # Instead of directly using ChromaChunkStore, we'll adapt it here to respect scoping
        from live_assist.retrieval.chroma_store import ChromaChunkStore

        payload = json.loads(enriched_path.read_text(encoding="utf-8"))
        e_chunks = payload.get("chunks", [])
        npz = np.load(vec_path, allow_pickle=True)
        raw_vecs = npz["vectors"].astype(float).tolist()
        ids_order = list(npz["chunk_ids"])
        idx_by_id = {cid: i for i, cid in enumerate(ids_order)}
        ordered_vecs = [raw_vecs[idx_by_id[c["chunk_id"]]] for c in e_chunks]
        
        # Prefix chunk_id with document_id to make them globally unique across the vector DB
        for c in e_chunks:
            c["chunk_id"] = f"{document_id}_{c['chunk_id']}"

        # Initialize ChromaChunkStore with collection name from settings (not os.environ)
        store = ChromaChunkStore(collection_name=settings.rag_chroma_collection)
        index_result = store.index_chunks(
            chunks=e_chunks,
            vectors=ordered_vecs,
            source_filename=pdf_path.name,
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

        # Stage 5: Rebuild BM25 Index
        # Since we use user scoping, we need the BM25 index to either be per-user OR built across all.
        # The existing `build_bm25_from_dir` rebuilds across ALL enriched chunks in `ENRICHED_DIR`.
        # That's fine because we can filter by `user_id` when retrieving.
        on_progress("bm25", "Rebuilding BM25 keyword index ...")
        t0 = time.perf_counter()

        from live_assist.retrieval.bm25_index import build_bm25_from_dir

        BM25_DIR.mkdir(parents=True, exist_ok=True)
        bm25_stats = build_bm25_from_dir(
            chunks_dir=ENRICHED_DIR,
            index_out=BM25_DIR / f"{_PIPELINE_NAME}.pkl",
            docs_out=BM25_DIR / f"{_PIPELINE_NAME}.docs.json",
        )

        bm25_time = round(time.perf_counter() - t0, 2)
        stages["bm25"] = {
            "elapsed_s": bm25_time,
            "total_chunks_in_index": bm25_stats["chunk_count"],
            "docs_in_index": bm25_stats["docs"],
        }
        on_progress("bm25", f"✓ BM25 index rebuilt ({bm25_stats['chunk_count']} total chunks) in {bm25_time}s")

        total_elapsed = round(time.perf_counter() - t_total, 2)
        on_progress("done", f"✅ Ingestion complete in {total_elapsed}s")

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
