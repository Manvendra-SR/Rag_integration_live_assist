from __future__ import annotations

import logging
from typing import Any

from langfuse import observe

from live_assist.core.config import get_settings
from live_assist.rag_pipeline.paths import BM25_DIR
from live_assist.retrieval.bm25_index import search_bm25
from live_assist.retrieval.context_assembler import assemble_context
from live_assist.retrieval.hybrid_search import search_hybrid
from live_assist.retrieval.reranker import RERANK_CANDIDATE_LIMIT, build_reranker
from live_assist.retrieval.semantic_search import search_semantic

log = logging.getLogger(__name__)

class AdvancedRetriever:
    def __init__(self, settings: Any):
        self.settings = settings
        self.mode = settings.rag_retrieval_mode
        self.pipeline = settings.rag_pipeline_name
        self.top_k = settings.rag_top_k
        self.collection_name = settings.rag_chroma_collection
        self.bm25_index = BM25_DIR / f"{self.pipeline}.pkl"
        self.bm25_docs = BM25_DIR / f"{self.pipeline}.docs.json"

    @observe(name="retrieve_documents")
    def retrieve(self, query: str, user_id: str, doc_filter: str | None = None,
                 mode: str | None = None) -> dict[str, Any]:
        effective_mode = mode or self.mode

        # Build filters: MUST scope by user_id
        filters: dict[str, Any] = {
            "chunking_strategy": self.pipeline,
        }
        if self.settings.rag_user_scope_enabled and user_id:
            filters["user_id"] = user_id

        if doc_filter:
            import re
            # Check if doc_filter is a UUID (sent by the frontend UI)
            if re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', doc_filter):
                filters["doc_id"] = doc_filter
            else:
                # Accept a full filename stem for manual queries
                if doc_filter.endswith(".pdf"):
                    filters["source_filename"] = doc_filter
                else:
                    filters["source_filename"] = f"{doc_filter}.pdf"

        results: list[dict] = []

        try:
            if effective_mode == "semantic":
                results = search_semantic(query, limit=self.top_k, filters=filters, collection_name=self.collection_name)

            elif effective_mode == "bm25":
                if not self.bm25_index.exists():
                    log.warning(f"BM25 index not found: {self.bm25_index}")
                    return self._empty_result()
                results = search_bm25(query, self.bm25_index, self.bm25_docs, limit=self.top_k, filters=filters)

            elif effective_mode in ("hybrid", "reranked"):
                if not self.bm25_index.exists():
                    log.warning(f"BM25 index not found: {self.bm25_index}")
                    # Fall back to semantic-only when BM25 index missing
                    log.info("Falling back to semantic-only retrieval (no BM25 index)")
                    results = search_semantic(query, limit=self.top_k, filters=filters, collection_name=self.collection_name)
                else:
                    candidate_limit = max(RERANK_CANDIDATE_LIMIT, self.top_k * 4) if effective_mode == "reranked" else self.top_k
                    fused = search_hybrid(query, self.bm25_index, self.bm25_docs, limit=candidate_limit, filters=filters, collection_name=self.collection_name)
                    results = fused["results"]

                    if effective_mode == "reranked" and results:
                        rr = build_reranker()
                        results = rr.rerank(query, results, top_k=self.top_k)

            if not results:
                return self._empty_result()

            context_payload = assemble_context({"query": query, "results": results})
            
            # Format context string as expected by live_assist graph
            context_parts = []
            rag_top_chunks = []
            
            for c in results:
                excerpt = c.get("text", "")
                if "text_with_context" in c:
                    excerpt = c["text_with_context"]
                    
                source = c.get("source_filename", "")
                page = c.get("page", "")
                
                if excerpt:
                    header = f"[{source} | p{page}]"
                    context_parts.append(f"{header}\n{excerpt}")
                    if len(rag_top_chunks) < 3:
                        rag_top_chunks.append(excerpt[:300])
                        
            context = "\n\n".join(context_parts)
            
            return {
                "context": context,
                "rag_top_chunks": rag_top_chunks,
                "rag_raw_chunks": results,
            }

        except Exception as exc:
            log.exception("Retrieval failed")
            return self._empty_result()

    def _empty_result(self) -> dict[str, Any]:
        return {
            "context": "",
            "rag_top_chunks": [],
        }
