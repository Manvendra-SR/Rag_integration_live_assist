"""ChromaDB vector store — drop-in replacement for weaviate_store.py.

Same external interface as WeaviateChunkStore:
    store = ChromaChunkStore()
    store.index_chunks(chunks, vectors, source_filename, chunking_strategy)
    results = store.search(query_vector, limit, filters)
    store.delete_doc(doc_id)

Uses a persistent local ChromaDB directory configured by:
    CHROMA_PERSIST_DIR   (default: ./runtime/chroma_db)
    CHROMA_COLLECTION    (default: DocumentChunk)
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

# Load .env from backend/ (same pattern as model_factory.py)
try:
    from clients import model_factory as _mf  # noqa: F401 — triggers dotenv load
except Exception:
    pass

log = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parents[1] / "runtime" / "chroma_db"),
)
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "DocumentChunk")


def _stable_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class ChromaChunkStore:
    def __init__(
        self,
        collection_name: str | None = None,
        client_id: str | None = None,
    ) -> None:
        base = collection_name or CHROMA_COLLECTION
        if client_id:
            safe = "".join(c if c.isalnum() else "_" for c in client_id).strip("_")
            self.collection_name = f"{base}__{safe}" if safe else base
        else:
            self.collection_name = base
        self._client = None
        self._collection = None

    # ── connection ────────────────────────────────────────────────────────────
    def _get_collection(self):
        import chromadb
        if self._client is None:
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            log.info(f"[chroma] connected, persist_dir={CHROMA_PERSIST_DIR}")
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"[chroma] using collection {self.collection_name!r}")
        return self._collection

    # ── indexing ──────────────────────────────────────────────────────────────
    def index_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
        source_filename: str,
        chunking_strategy: str,
    ) -> dict:
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        coll = self._get_collection()

        ids, embeddings, documents, metadatas = [], [], [], []
        for c, vec in zip(chunks, vectors):
            section_path = c.get("section_path") or []
            if isinstance(section_path, str):
                section_path = [section_path]

            ids.append(_stable_uuid(c["chunk_id"]))
            embeddings.append(list(map(float, vec)))
            documents.append(c.get("text") or c.get("content") or "")
            metadatas.append({
                "chunk_id":          c["chunk_id"],
                "doc_id":            c.get("doc_id") or "",
                "source_filename":   source_filename,
                "section_path":      " > ".join(str(s) for s in section_path),
                "page_start":        int(c.get("page_start") or 0),
                "page_end":          int(c.get("page_end") or 0),
                "token_count":       int(c.get("token_count") or 0),
                "chunking_strategy": chunking_strategy,
            })

        # Upsert in batches of 500 to avoid memory spikes
        BATCH = 500
        n_indexed = 0
        for i in range(0, len(ids), BATCH):
            coll.upsert(
                ids=ids[i:i + BATCH],
                embeddings=embeddings[i:i + BATCH],
                documents=documents[i:i + BATCH],
                metadatas=metadatas[i:i + BATCH],
            )
            n_indexed += len(ids[i:i + BATCH])

        log.info(f"[chroma] indexed {n_indexed} chunks into {self.collection_name!r}")
        return {
            "collection": self.collection_name,
            "indexed": n_indexed,
            "failed": 0,
        }

    # ── query ─────────────────────────────────────────────────────────────────
    def _build_where(self, filters: dict | None) -> dict | None:
        if not filters:
            return None
        clauses = []
        for key in ("doc_id", "source_filename", "chunking_strategy"):
            if filters.get(key):
                clauses.append({key: {"$eq": filters[key]}})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        coll = self._get_collection()
        where = self._build_where(filters)
        n = limit or 50

        kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, query_vector))],
            "n_results": min(n, coll.count() or 1),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        resp = coll.query(**kwargs)

        out = []
        for meta, doc, dist in zip(
            resp["metadatas"][0],
            resp["documents"][0],
            resp["distances"][0],
        ):
            out.append({
                "uuid": None,
                "properties": {
                    "chunk_id":          meta.get("chunk_id"),
                    "doc_id":            meta.get("doc_id"),
                    "source_filename":   meta.get("source_filename"),
                    "section_path":      [s for s in (meta.get("section_path") or "").split(" > ") if s],
                    "text":              doc,
                    "page_start":        meta.get("page_start"),
                    "page_end":          meta.get("page_end"),
                    "token_count":       meta.get("token_count"),
                    "chunking_strategy": meta.get("chunking_strategy"),
                },
                "distance": float(dist),
            })
        return out

    # ── delete ────────────────────────────────────────────────────────────────
    def delete_doc(self, doc_id: str) -> int:
        coll = self._get_collection()
        results = coll.get(where={"doc_id": {"$eq": doc_id}}, include=[])
        ids = results.get("ids") or []
        if ids:
            coll.delete(ids=ids)
        log.info(f"[chroma] deleted {len(ids)} chunks for doc_id={doc_id!r}")
        return len(ids)
