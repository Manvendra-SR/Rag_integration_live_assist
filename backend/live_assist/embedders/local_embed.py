"""Embedding shim — delegates to `clients.get_embedder()`.

Backward-compatible: existing call sites that do
    from live_assist.embedders.local_embed import embed_texts, get_embedder
still work, but the actual provider+model is now selected by .env
(EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBED_LOCAL_MODEL / *_API_KEY) with
cloud→local fallback handled centrally in `clients.model_factory`.

Outputs remain L2-normalized float32 vectors compatible with the prior
pipeline. Default local fallback is BAAI/bge-m3 (1024-dim).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"

_LEGACY_MODEL = None  # cached sentence-transformers model for the legacy path


def get_embedder(model_name: str = DEFAULT_MODEL):
    """Backward-compat: returns the underlying SentenceTransformer model.

    New code should call `clients.get_embedder()` directly to get the
    unified client that handles cloud + local in one place. This shim
    preserves the old behaviour for any caller that depends on receiving
    the raw `SentenceTransformer` object.
    """
    global _LEGACY_MODEL
    if _LEGACY_MODEL is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading legacy local embedder {model_name} ...")
        t0 = time.perf_counter()
        _LEGACY_MODEL = SentenceTransformer(model_name)
        log.info(f"  legacy embedder loaded in {time.perf_counter()-t0:.1f}s")
    return _LEGACY_MODEL


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL,
                batch_size: int = 8, normalize: bool = True) -> np.ndarray:
    """Embed a list of texts via the unified client factory.

    Returns ndarray shape (n_texts, dim) of float32 L2-normalized vectors.
    The provider (cohere / openai / gemini / local sentence-transformers) is
    chosen by .env at `clients.get_embedder()` time. `model_name` is only
    honoured on the legacy fallback path; otherwise EMBEDDING_MODEL wins.
    """
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    t0 = time.perf_counter()
    # Try unified factory first (reads .env: EMBEDDING_PROVIDER / MODEL / KEY)
    try:
        from live_assist.clients import get_embedder as _factory_get_embedder
        emb = _factory_get_embedder()
        # Some providers may need batched chunking; cohere/openai accept
        # large arrays directly so a single call is fine for typical sizes.
        vecs_list = emb.embed(list(texts))
        vecs = np.asarray(vecs_list, dtype=np.float32)
        # L2-normalize for cosine retrieval (cloud APIs return un-normalized)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        log.info(f"  embedded {len(texts)} texts via {emb.name} in "
                 f"{time.perf_counter()-t0:.1f}s (shape={vecs.shape})")
        return vecs
    except Exception as exc:
        log.warning(f"factory embed failed ({exc}); falling back to legacy "
                    f"local sentence-transformers path")
    # Legacy local path — bge-m3 via sentence-transformers
    model = get_embedder(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    log.info(f"  embedded {len(texts)} texts via legacy local in "
             f"{time.perf_counter()-t0:.1f}s (shape={vecs.shape})")
    return vecs.astype(np.float32)


def save_vectors(vectors: np.ndarray, chunk_ids: list[str], out_path: Path,
                 model_name: str = DEFAULT_MODEL) -> Path:
    """Persist as .npz so it can be reloaded fast."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        vectors=vectors,
        chunk_ids=np.array(chunk_ids, dtype=object),
        model=np.array([model_name], dtype=object),
    )
    return out_path
