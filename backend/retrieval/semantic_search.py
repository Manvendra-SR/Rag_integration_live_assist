"""Semantic (Weaviate cosine) search.

Embeds the query via `clients.get_embedder()` and runs near_vector on the
shared `DocumentChunk` collection. Penalises reference-section chunks.
"""
from __future__ import annotations

import logging
from typing import Any

from .ranking_utils import is_reference_like, penalize_reference_score
from .chroma_store import ChromaChunkStore

log = logging.getLogger(__name__)


def search_semantic(query: str, limit: int = 10,
                    filters: dict | None = None,
                    oversample_factor: int = 5,
                    collection_name: str | None = None) -> list[dict]:
    from clients import get_embedder
    emb = get_embedder()
    q_vec = emb.embed_one(query)
    # L2-normalise the query vector if the embedder didn't already
    import math
    n = math.sqrt(sum(v * v for v in q_vec))
    if n and abs(n - 1.0) > 1e-3:
        q_vec = [v / n for v in q_vec]

    store = ChromaChunkStore(collection_name=collection_name)
    raw = store.search(query_vector=q_vec,
                       limit=max(limit * oversample_factor, limit),
                       filters=filters)
    out: list[dict] = []
    for item in raw:
        p = item["properties"]
        dist = item.get("distance")
        base = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
        sec_for_ref = " > ".join(p.get("section_path") or []) or None
        ref = is_reference_like(sec_for_ref, p.get("text"))
        score = penalize_reference_score(base, ref)
        out.append({
            "chunk_id":        p.get("chunk_id"),
            "doc_id":          p.get("doc_id"),
            "source_filename": p.get("source_filename"),
            "section_path":    list(p.get("section_path") or []),
            "text":            p.get("text"),
            "page_start":      p.get("page_start"),
            "page_end":        p.get("page_end"),
            "token_count":     p.get("token_count"),
            "chunking_strategy": p.get("chunking_strategy"),
            "distance":        dist,
            "semantic_score":  round(score, 6),
            "reference_like":  ref,
        })
    out.sort(key=lambda x: (x["semantic_score"],
                            -(x["distance"] or 0.0)), reverse=True)
    return out[:limit]
