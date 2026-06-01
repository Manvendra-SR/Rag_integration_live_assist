"""Weaviate wrapper — single-collection, multi-document, self-provided vectors.

Schema (collection `DocumentChunk` by default):

    chunk_id            TEXT      (filterable)  ← stable UUID derives from this
    doc_id              TEXT      (filterable)
    source_filename     TEXT      (filterable)
    section_path        TEXT_ARRAY(filterable)
    text                TEXT      (searchable)
    page_start          INT       (filterable)
    page_end            INT       (filterable)
    token_count         INT       (filterable)
    chunking_strategy   TEXT      (filterable)  ← 'anthropic' or 'semantic'
    ingestion_timestamp DATE      (filterable)

Many PDFs share one collection. Per-document scope is a `doc_id` filter, NOT a
separate collection. Optional `client_id` argument creates a per-tenant
collection suffix (`DocumentChunk__<client>`) for hard isolation.

UUID for each chunk = uuid5(NAMESPACE_URL, chunk_id) → idempotent re-indexing.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

# Make sure .env is loaded before reading WEAVIATE_* envs at module-import time.
# `clients.model_factory` already has its own dotenv loader; importing it here
# triggers the same load path without needing python-dotenv as a hard dep.
try:
    from clients import model_factory as _model_factory  # noqa: F401
except Exception:
    pass

log = logging.getLogger(__name__)

WEAVIATE_PROVIDER = (os.environ.get("WEAVIATE_PROVIDER", "cloud") or "cloud").lower()
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "").strip()
WEAVIATE_COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "DocumentChunk")
WEAVIATE_LOCAL_HOST = os.environ.get("WEAVIATE_LOCAL_HOST", "localhost").strip()
WEAVIATE_LOCAL_HTTP_PORT = int(os.environ.get("WEAVIATE_LOCAL_HTTP_PORT", "8080"))
WEAVIATE_LOCAL_GRPC_PORT = int(os.environ.get("WEAVIATE_LOCAL_GRPC_PORT", "50051"))
WEAVIATE_HNSW_EF_CONSTRUCTION = int(os.environ.get("WEAVIATE_HNSW_EF_CONSTRUCTION", "128"))
WEAVIATE_HNSW_M = int(os.environ.get("WEAVIATE_HNSW_M", "32"))
WEAVIATE_QUERY_LIMIT = int(os.environ.get("WEAVIATE_QUERY_LIMIT", "50"))


def _stable_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _normalize_collection_suffix(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WeaviateChunkStore:
    def __init__(self, collection_name: str | None = None,
                 client_id: str | None = None) -> None:
        base = collection_name or WEAVIATE_COLLECTION
        if client_id:
            suf = _normalize_collection_suffix(client_id)
            self.collection_name = f"{base}__{suf}" if suf else base
        else:
            self.collection_name = base

    # ---------- connection ----------
    def connect(self):
        """Connect according to WEAVIATE_PROVIDER (.env).

        - `cloud` (default): connect_to_weaviate_cloud(WEAVIATE_URL,
          api_key=WEAVIATE_API_KEY). If the URL is unreachable, auto-falls
          back to the local entry below so a dev machine without a cloud
          cluster still works.
        - `local`: connect_to_local(host=WEAVIATE_LOCAL_HOST, port=…).
          Assumes a Weaviate container running on those ports.
        """
        import weaviate
        from weaviate.classes.init import Auth

        def _connect_local():
            log.info(f"[weaviate] connecting local "
                     f"{WEAVIATE_LOCAL_HOST}:{WEAVIATE_LOCAL_HTTP_PORT}")
            return weaviate.connect_to_local(
                host=WEAVIATE_LOCAL_HOST,
                port=WEAVIATE_LOCAL_HTTP_PORT,
                grpc_port=WEAVIATE_LOCAL_GRPC_PORT,
            )

        def _connect_cloud():
            if not WEAVIATE_URL:
                raise RuntimeError("WEAVIATE_PROVIDER=cloud but WEAVIATE_URL is empty")
            log.info(f"[weaviate] connecting cloud {WEAVIATE_URL[:40]}...")
            if WEAVIATE_API_KEY:
                return weaviate.connect_to_weaviate_cloud(
                    cluster_url=WEAVIATE_URL,
                    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                )
            return weaviate.connect_to_custom(http_host=WEAVIATE_URL)

        if WEAVIATE_PROVIDER == "local":
            return _connect_local()
        # cloud (default) — try, then fall back to local on failure
        try:
            return _connect_cloud()
        except Exception as exc:
            log.warning(f"[weaviate] cloud connect failed ({type(exc).__name__}: "
                        f"{exc}); falling back to local "
                        f"{WEAVIATE_LOCAL_HOST}:{WEAVIATE_LOCAL_HTTP_PORT}")
            return _connect_local()

    # ---------- schema ----------
    def ensure_collection(self, client) -> None:
        from weaviate.classes.config import (
            Configure, Property, DataType, VectorDistances,
        )
        existing_response = client.collections.list_all()
        if isinstance(existing_response, dict):
            existing = list(existing_response.keys())
        else:
            existing = [getattr(c, "name", str(c)) for c in existing_response]
        if self.collection_name in existing:
            return
        client.collections.create(
            self.collection_name,
            vector_config=Configure.Vectors.self_provided(
                name="default",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=WEAVIATE_HNSW_EF_CONSTRUCTION,
                    max_connections=WEAVIATE_HNSW_M,
                ),
            ),
            properties=[
                Property(name="chunk_id",            data_type=DataType.TEXT,       index_filterable=True),
                Property(name="doc_id",              data_type=DataType.TEXT,       index_filterable=True),
                Property(name="source_filename",     data_type=DataType.TEXT,       index_filterable=True),
                Property(name="section_path",        data_type=DataType.TEXT_ARRAY, index_filterable=True),
                Property(name="text",                data_type=DataType.TEXT,       index_searchable=True),
                Property(name="page_start",          data_type=DataType.INT,        index_filterable=True),
                Property(name="page_end",            data_type=DataType.INT,        index_filterable=True),
                Property(name="token_count",         data_type=DataType.INT,        index_filterable=True),
                Property(name="chunking_strategy",   data_type=DataType.TEXT,       index_filterable=True),
                Property(name="ingestion_timestamp", data_type=DataType.DATE,       index_filterable=True),
            ],
        )
        log.info(f"Created Weaviate collection {self.collection_name!r}")

    # ---------- indexing ----------
    def index_chunks(self, chunks: list[dict], vectors: list[list[float]],
                     source_filename: str, chunking_strategy: str) -> dict:
        """Index a batch of chunks + vectors into the collection.

        Args:
          chunks: list of chunk dicts with keys chunk_id, doc_id, text,
                  section_path, page_start, page_end, token_count
          vectors: parallel list of float vectors (same length as chunks)
          source_filename: e.g. "paper2.pdf" — stored on every chunk
          chunking_strategy: 'anthropic' or 'semantic'
        """
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        client = self.connect()
        n_indexed = 0
        n_failed = 0
        try:
            self.ensure_collection(client)
            coll = client.collections.use(self.collection_name)
            ts = _iso_now()
            with coll.batch.dynamic() as batch:
                for c, vec in zip(chunks, vectors):
                    section_path = c.get("section_path") or []
                    if isinstance(section_path, str):
                        section_path = [section_path]
                    batch.add_object(
                        properties={
                            "chunk_id":            c["chunk_id"],
                            "doc_id":              c.get("doc_id") or "",
                            "source_filename":     source_filename,
                            "section_path":        [str(s) for s in section_path],
                            "text":                c.get("text") or c.get("content") or "",
                            "page_start":          int(c.get("page_start") or 0),
                            "page_end":            int(c.get("page_end") or 0),
                            "token_count":         int(c.get("token_count") or 0),
                            "chunking_strategy":   chunking_strategy,
                            "ingestion_timestamp": ts,
                        },
                        vector=list(vec),
                        uuid=_stable_uuid(c["chunk_id"]),
                    )
                    n_indexed += 1
            failed = getattr(coll.batch, "failed_objects", []) or []
            n_failed = len(failed)
            log.info(f"Indexed {n_indexed} chunks into {self.collection_name} "
                     f"(failed={n_failed})")
            return {"collection": self.collection_name,
                    "indexed": n_indexed, "failed": n_failed}
        finally:
            client.close()

    # ---------- query ----------
    def _build_filters(self, filters):
        if not filters:
            return None
        from weaviate.classes.query import Filter
        clauses = []
        if filters.get("doc_id"):
            clauses.append(Filter.by_property("doc_id").equal(filters["doc_id"]))
        if filters.get("source_filename"):
            clauses.append(Filter.by_property("source_filename").equal(filters["source_filename"]))
        if filters.get("chunking_strategy"):
            clauses.append(Filter.by_property("chunking_strategy").equal(filters["chunking_strategy"]))
        if filters.get("page_start") is not None:
            clauses.append(Filter.by_property("page_end").greater_or_equal(int(filters["page_start"])))
        if filters.get("page_end") is not None:
            clauses.append(Filter.by_property("page_start").less_or_equal(int(filters["page_end"])))
        if not clauses:
            return None
        combined = clauses[0]
        for cl in clauses[1:]:
            combined = combined & cl
        return combined

    def search(self, query_vector: list[float], limit: int | None = None,
               filters: dict | None = None) -> list[dict]:
        client = self.connect()
        try:
            self.ensure_collection(client)
            coll = client.collections.use(self.collection_name)
            wfilters = self._build_filters(filters)
            resp = coll.query.near_vector(
                near_vector=list(query_vector),
                limit=limit or WEAVIATE_QUERY_LIMIT,
                filters=wfilters,
                return_metadata=["distance"],
            )
            out = []
            for item in resp.objects:
                out.append({
                    "uuid": str(item.uuid),
                    "properties": dict(item.properties or {}),
                    "distance": getattr(item.metadata, "distance", None),
                })
            return out
        finally:
            client.close()

    def delete_doc(self, doc_id: str) -> int:
        """Remove all chunks of a single doc_id (handy for re-ingestion)."""
        client = self.connect()
        try:
            self.ensure_collection(client)
            coll = client.collections.use(self.collection_name)
            from weaviate.classes.query import Filter
            res = coll.data.delete_many(
                where=Filter.by_property("doc_id").equal(doc_id),
            )
            return getattr(res, "successful", 0) or 0
        finally:
            client.close()
