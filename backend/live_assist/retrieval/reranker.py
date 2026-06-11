"""Cross-encoder reranker — local (sentence-transformers) or Cohere.

Provider + model are read from .env (RERANK_PROVIDER, RERANK_MODEL,
COHERE_API_KEY). Same cloud→local fallback pattern as `clients.model_factory`:
if RERANK_PROVIDER is a cloud option but the key/SDK is missing/broken, fall
back to the local CrossEncoder. Local default:
    cross-encoder/ms-marco-MiniLM-L-6-v2  (≈90 MB, ~10 ms per pair on M3)

Usage:
    rr = build_reranker()
    ranked = rr.rerank(query, candidates, top_k=5)   # adds .rerank_score, sorts
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Sequence

from langfuse import observe

log = logging.getLogger(__name__)

RERANK_PROVIDER = (os.environ.get("RERANK_PROVIDER", "local") or "local").lower()
RERANK_MODEL = os.environ.get("RERANK_MODEL") or ("rerank-v3.5"
    if RERANK_PROVIDER == "cohere" else "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATE_LIMIT = int(os.environ.get("RERANK_CANDIDATE_LIMIT", "20"))
RERANK_FINAL_LIMIT = int(os.environ.get("RERANK_FINAL_LIMIT", "5"))


class BaseReranker:
    name = "base"
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise NotImplementedError
    @observe(name="rerank_documents")
    def rerank(self, query: str, candidates: list[dict],
               top_k: int = RERANK_FINAL_LIMIT,
               text_field: str = "text") -> list[dict]:
        if not candidates:
            return []
        scores = self.score(query, [c.get(text_field, "") for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 6)
        candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        for r, c in enumerate(candidates[:top_k], start=1):
            c["rerank_rank"] = r
        return candidates[:top_k]


class LocalCrossEncoder(BaseReranker):
    name = "local_cross_encoder"
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    def score(self, query, documents):
        if not documents:
            return []
        pairs = [[query, d] for d in documents]
        return [float(s) for s in self.model.predict(pairs)]


class CohereReranker(BaseReranker):
    name = "cohere"
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        import cohere
        key = (os.environ.get("COHERE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("COHERE_API_KEY missing")
        self.client = cohere.ClientV2(api_key=key)
    def score(self, query, documents):
        if not documents:
            return []
        resp = self.client.rerank(model=self.model_name, query=query,
                                  documents=list(documents))
        indexed = {item.index: float(item.relevance_score) for item in resp.results}
        return [indexed.get(i, 0.0) for i in range(len(documents))]


@observe(name="build_reranker")
@lru_cache(maxsize=1)
def build_reranker() -> BaseReranker:
    """Return the configured reranker. Cloud first; local fallback on failure."""
    if RERANK_PROVIDER == "cohere":
        try:
            log.info(f"reranker → cohere:{RERANK_MODEL}")
            return CohereReranker(RERANK_MODEL)
        except Exception as exc:
            log.warning(f"cohere reranker failed ({exc}); falling back to local")
    # default local
    model = RERANK_MODEL if RERANK_PROVIDER == "local" else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    log.info(f"reranker → local:{model}")
    return LocalCrossEncoder(model)
