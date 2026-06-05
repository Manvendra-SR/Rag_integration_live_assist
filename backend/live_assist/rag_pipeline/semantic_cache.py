"""Semantic query cache.

Stores and retrieves cached query results using raw-query embedding similarity.
Cache entries are scoped strictly by (doc_filter, retrieval_mode, index_version)
so semantic matching never crosses different retrieval configurations.

Only activated when a specific doc_filter is present (not "All Documents").
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        cache_dir: Path,
        index_version_file: Path,
        similarity_threshold: float = 0.97,
        max_age_days: int = 7,
    ) -> None:
        self.cache_dir = cache_dir
        self.index_version_file = index_version_file
        self.similarity_threshold = similarity_threshold
        self.max_age_seconds = max_age_days * 86400

    # ── Index Version ──────────────────────────────────────────────────────────

    def read_index_version(self) -> str:
        """Return the current index version string (a timestamp)."""
        try:
            if self.index_version_file.exists():
                return self.index_version_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return "0"

    def bump_index_version(self) -> str:
        """Write a new index version timestamp. Call after successful ingestion."""
        version = str(time.time())
        try:
            self.index_version_file.write_text(version, encoding="utf-8")
            log.info(f"[SemanticCache] index version bumped → {version}")
        except Exception as exc:
            log.warning(f"[SemanticCache] failed to bump index version: {exc}")
        return version

    # ── Condition Group ────────────────────────────────────────────────────────

    def _condition_hash(self, doc_filter: str, retrieval_mode: str, index_version: str) -> str:
        """Return a stable hash identifying a retrieval configuration group."""
        key = f"{doc_filter}|{retrieval_mode}|{index_version}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _group_dir(self, doc_filter: str, retrieval_mode: str, index_version: str) -> Path:
        cond = self._condition_hash(doc_filter, retrieval_mode, index_version)
        d = self.cache_dir / cond
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Embed ──────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray | None:
        """Embed a single query string using the same client as semantic_search.py."""
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

    def lookup(
        self,
        raw_query: str,
        doc_filter: str,
        retrieval_mode: str,
    ) -> dict[str, Any] | None:
        """
        Attempt a semantic cache lookup.

        Returns the cached workflow_response dict on a hit, or None on a miss.
        Only valid when doc_filter is non-empty.
        """
        if not doc_filter:
            return {"result": "SKIPPED", "similarity": None}

        index_version = self.read_index_version()
        group = self._group_dir(doc_filter, retrieval_mode, index_version)
        entries = list(group.glob("*.json"))
        if not entries:
            return {"result": "MISS", "similarity": None}  # no entries yet for this config

        q_vec = self._embed_query(raw_query)
        if q_vec is None:
            return {"result": "MISS", "similarity": None}  # embed failed

        now = time.time()
        best_score = -1.0
        best_entry: dict | None = None

        for path in entries:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Skip expired entries
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
                f"doc_filter={doc_filter} | retrieval_mode={retrieval_mode}"
            )
            return {
                "workflow_response": best_entry["workflow_response"],
                "similarity": best_score,
                "result": "HIT",
                "cached_query": best_entry.get("raw_query", ""),
            }

        log.info(
            f"[SemanticCache] MISS | best_sim={best_score:.4f} "
            f"(threshold={self.similarity_threshold}) | doc_filter={doc_filter}"
        )
        # Return None to signal MISS, but carry best_score so caller can log it
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

    def write(
        self,
        raw_query: str,
        doc_filter: str,
        retrieval_mode: str,
        workflow_response: dict[str, Any],
    ) -> None:
        """Persist a cache entry for a successful query response."""
        if not doc_filter:
            return  # Never cache "All Documents" queries

        index_version = self.read_index_version()
        q_vec = self._embed_query(raw_query)
        if q_vec is None:
            log.warning("[SemanticCache] skipping write — embed returned None")
            return

        # Strip internal _cache_* keys and make response JSON-safe
        safe_response = self._safe_serialise({
            k: v for k, v in workflow_response.items()
            if not k.startswith("_cache")
        })

        group = self._group_dir(doc_filter, retrieval_mode, index_version)
        entry = {
            "raw_query": raw_query,
            "query_vector": q_vec.tolist(),
            "doc_filter": doc_filter,
            "retrieval_mode": retrieval_mode,
            "index_version": index_version,
            "created_at": time.time(),
            "workflow_response": safe_response,
        }
        entry_file = group / f"{uuid.uuid4().hex}.json"
        try:
            entry_file.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            log.info(f"[SemanticCache] entry written → {entry_file.name}")
        except Exception as exc:
            import traceback
            log.error(f"[SemanticCache] failed to write entry: {exc}\n{traceback.format_exc()}")

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Delete cache entries older than max_age_days. Returns count removed."""
        now = time.time()
        removed = 0
        for path in self.cache_dir.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - data.get("created_at", 0) > self.max_age_seconds:
                    path.unlink()
                    removed += 1
            except Exception:
                pass
        return removed
