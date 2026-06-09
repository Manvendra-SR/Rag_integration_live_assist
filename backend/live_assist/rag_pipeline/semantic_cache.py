"""Semantic query cache.

Stores and retrieves cached query results using enriched-query embedding similarity.
Cache entries are scoped strictly to a single conversation (session_id directory).

Key design decisions:
  - Conversation-scoped: runtime/cache/<session_id>/ — never shared across conversations.
  - Purged on conversation end via SemanticCache.purge() called from the call_end route.
  - Uses the ENRICHED query (post-LLM rewrite) as the vector key, not the raw query.
  - No doc_filter / retrieval_mode / index_version in scope — those are server-controlled.
  - Activated for manual questions when rag_cache_enabled=True.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langfuse import observe

log = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        cache_dir: Path,
        similarity_threshold: float = 0.97,
        max_age_days: int = 7,
    ) -> None:
        """
        Args:
            cache_dir: The per-conversation cache directory
                       (runtime/cache/<session_id>/).
            similarity_threshold: Cosine similarity threshold for a cache hit.
            max_age_days: Safety eviction guard; 0 disables time-based eviction.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_age_seconds = max_age_days * 86400

    # ── Embed ──────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray | None:
        """Embed a single query string using the configured embedder."""
        try:
            from live_assist.clients.model_factory import get_embedder
            emb = get_embedder()
            q_vec = emb.embed_one(query)
            # L2-normalise (mirrors semantic_search.py)
            n = math.sqrt(sum(v * v for v in q_vec))
            if n and abs(n - 1.0) > 1e-3:
                q_vec = [v / n for v in q_vec]
            return np.asarray(q_vec, dtype=np.float32)
        except Exception as exc:
            log.warning(f"[SemanticCache] embed failed: {exc}")
            return None

    # ── Lookup ─────────────────────────────────────────────────────────────────

    @observe(name="cache_lookup")
    def lookup(self, enriched_query: str) -> dict[str, Any]:
        """
        Attempt a semantic cache lookup using the enriched (rewritten) query.

        Returns a dict with:
          - 'result': 'HIT' | 'MISS'
          - 'similarity': float | None
          - 'workflow_response': dict  (only on HIT)
          - 'cached_query': str        (only on HIT)
        """
        entries = list(self.cache_dir.glob("*.json"))
        if not entries:
            return {"result": "MISS", "similarity": None}

        q_vec = self._embed_query(enriched_query)
        if q_vec is None:
            return {"result": "MISS", "similarity": None}

        now = time.time()
        best_score = -1.0
        best_entry: dict | None = None

        for path in entries:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Safety eviction (conversations are short-lived so this rarely fires)
            if self.max_age_seconds > 0:
                age = now - data.get("created_at", 0)
                if age > self.max_age_seconds:
                    continue

            stored_vec = np.asarray(data.get("query_vector", []), dtype=np.float32)
            if stored_vec.shape != q_vec.shape:
                continue

            score = float(np.dot(q_vec, stored_vec))
            if score > best_score:
                best_score = score
                best_entry = data

        if best_entry is not None and best_score >= self.similarity_threshold:
            log.info(
                f"[SemanticCache] HIT | similarity={best_score:.4f} | "
                f"conversation={self.cache_dir.name}"
            )
            return {
                "workflow_response": best_entry["workflow_response"],
                "similarity": best_score,
                "result": "HIT",
                "cached_query": best_entry.get("enriched_query", ""),
            }

        log.info(
            f"[SemanticCache] MISS | best_sim={best_score:.4f} "
            f"(threshold={self.similarity_threshold}) | conversation={self.cache_dir.name}"
        )
        return {"result": "MISS", "similarity": best_score if best_score >= 0 else None}

    # ── Write ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_serialise(obj: Any) -> Any:
        """Recursively convert non-JSON-serialisable objects to safe types."""
        # LangChain message objects — serialise to a plain dict
        try:
            from langchain_core.messages import BaseMessage
            if isinstance(obj, BaseMessage):
                return {"role": obj.type, "content": str(obj.content)}
        except ImportError:
            pass
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            # Skip the 'messages' key — it holds full conversation history
            # which is not needed for cache replay and contains BaseMessage objects
            return {
                k: SemanticCache._safe_serialise(v)
                for k, v in obj.items()
                if k != "messages"
            }
        if isinstance(obj, (list, tuple)):
            return [SemanticCache._safe_serialise(v) for v in obj]
        return obj

    @observe(name="cache_store")
    def write(
        self,
        enriched_query: str,
        workflow_response: dict[str, Any],
    ) -> None:
        """Persist a cache entry keyed by the enriched query vector."""
        q_vec = self._embed_query(enriched_query)
        if q_vec is None:
            log.warning("[SemanticCache] skipping write — embed returned None")
            return

        # Strip internal _cache_* keys and make response JSON-safe
        safe_response = self._safe_serialise({
            k: v for k, v in workflow_response.items()
            if not k.startswith("_cache")
        })

        entry = {
            "enriched_query": enriched_query,
            "query_vector": q_vec.tolist(),
            "created_at": time.time(),
            "workflow_response": safe_response,
        }
        entry_file = self.cache_dir / f"{uuid.uuid4().hex}.json"
        try:
            entry_file.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            log.info(f"[SemanticCache] entry written → {entry_file.name}")
        except Exception as exc:
            import traceback
            log.error(
                f"[SemanticCache] failed to write entry: {exc}\n{traceback.format_exc()}"
            )

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def purge(self) -> int:
        """
        Delete the entire conversation cache directory.
        Called by the call_end route when a conversation ends.
        Returns the number of cache entries removed.
        """
        removed = 0
        if self.cache_dir.exists():
            try:
                files = list(self.cache_dir.glob("*.json"))
                removed = len(files)
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                log.info(
                    f"[SemanticCache] purged {removed} entries — "
                    f"conversation={self.cache_dir.name}"
                )
            except Exception as exc:
                log.warning(f"[SemanticCache] purge failed: {exc}")
        return removed

    def evict_expired(self) -> int:
        """
        Delete cache entries older than max_age_days.
        Rarely needed since conversations are purged on end; kept as a safety guard.
        Returns count removed.
        """
        if self.max_age_seconds <= 0:
            return 0
        now = time.time()
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - data.get("created_at", 0) > self.max_age_seconds:
                    path.unlink()
                    removed += 1
            except Exception:
                pass
        return removed
