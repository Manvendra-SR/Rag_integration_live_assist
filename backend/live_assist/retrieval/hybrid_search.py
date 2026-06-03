"""Hybrid retrieval — semantic + BM25 fused by Reciprocal Rank Fusion (RRF).

RRF score for a chunk = sum over retrievers of  1 / (k + rank_in_retriever),
where rank is 1-based and k=60 is the canonical Microsoft/PyTorch default.
Reference-like chunks get the same penalty as in semantic_search.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from .bm25_index import search_bm25
from .ranking_utils import is_reference_like, penalize_reference_score
from .semantic_search import search_semantic

log = logging.getLogger(__name__)

RRF_K_DEFAULT = 60


def _dynamic_limit(query: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    n = len(query.split())
    if n <= 3:
        return 8
    if n <= 8:
        return 10
    return 12


def reciprocal_rank_fusion(ranked_lists: dict[str, list[dict]],
                           limit: int,
                           rrf_k: int = RRF_K_DEFAULT) -> list[dict]:
    fused_scores: dict[str, float] = defaultdict(float)
    fused_payloads: dict[str, dict] = {}
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for source, results in ranked_lists.items():
        for rank, r in enumerate(results, start=1):
            cid = r["chunk_id"]
            fused_scores[cid] += 1.0 / (rrf_k + rank)
            source_ranks[cid][source] = rank
            if cid not in fused_payloads:
                fused_payloads[cid] = {**r, "sources": set()}
            fused_payloads[cid]["sources"].add(source)

    out: list[dict] = []
    rescored: list[tuple[str, float]] = []
    for cid, base in fused_scores.items():
        p = dict(fused_payloads[cid])
        sec_for_ref = " > ".join(p.get("section_path") or []) or None
        ref = is_reference_like(sec_for_ref, p.get("text"))
        adj = penalize_reference_score(base, ref)
        p["rrf_score"] = round(adj, 6)
        p["reference_like"] = ref
        fused_payloads[cid] = p
        rescored.append((cid, adj))
    ranked_ids = [cid for cid, _ in sorted(rescored, key=lambda x: x[1], reverse=True)[:limit]]
    for rank, cid in enumerate(ranked_ids, start=1):
        p = dict(fused_payloads[cid])
        p["source_ranks"] = source_ranks[cid]
        p["sources"] = sorted(p["sources"])
        p["final_rank"] = rank
        out.append(p)
    return out


def search_hybrid(query: str,
                  bm25_index_path: Path,
                  bm25_docs_path: Path,
                  limit: int | None = None,
                  filters: dict | None = None,
                  collection_name: str | None = None,
                  rrf_k: int = RRF_K_DEFAULT) -> dict:
    final_limit = _dynamic_limit(query, limit)
    candidate = max(final_limit * 4, 20)

    sem = search_semantic(query=query, limit=candidate,
                          filters=filters, collection_name=collection_name)
    bm = search_bm25(query=query,
                     index_path=bm25_index_path, docs_path=bm25_docs_path,
                     limit=candidate, filters=filters)
    fused = reciprocal_rank_fusion(
        ranked_lists={"semantic": sem, "bm25": bm},
        limit=final_limit, rrf_k=rrf_k,
    )
    return {
        "query": query, "limit": final_limit,
        "candidate_pool": candidate, "rrf_k": rrf_k,
        "bm25_count": len(bm), "semantic_count": len(sem),
        "bm25_results": bm,
        "semantic_results": sem,
        "results": fused,
    }
