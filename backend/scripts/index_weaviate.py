"""CLI: load chunks + vectors from one source dir into Weaviate.

Reads:
  runtime/chunks_anthropic/<stem>.chunks.json + .vectors.npz   (Pipeline A)
  runtime/chunks_semantic/<stem>.chunks.json + .vectors.npz    (Pipeline B)

Writes:
  Into the shared `DocumentChunk` collection in Weaviate Cloud
  (URL + API key from .env).

Many PDFs share one collection. Re-indexing the same doc is idempotent
because the UUID is uuid5(NAMESPACE_URL, chunk_id).

Usage:
  .venv/bin/python scripts/index_weaviate.py --pipeline anthropic --stem paper2
  .venv/bin/python scripts/index_weaviate.py --pipeline semantic  --stem paper2
  .venv/bin/python scripts/index_weaviate.py --pipeline anthropic --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from retrieval.weaviate_store import WeaviateChunkStore  # noqa: E402

SOURCE_DIRS = {
    "anthropic": PROJECT_ROOT / "runtime" / "chunks_anthropic",
    "semantic":  PROJECT_ROOT / "runtime" / "chunks_semantic",
}


def _load_stem(src_dir: Path, stem: str) -> tuple[dict, list[list[float]], str]:
    chunks_path = src_dir / f"{stem}.chunks.json"
    vec_path    = src_dir / f"{stem}.vectors.npz"
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if vec_path.exists():
        npz = np.load(vec_path, allow_pickle=True)
        vecs = npz["vectors"].astype(float).tolist()
        ids = list(npz["chunk_ids"])
        # Reorder vectors so they line up with chunks list
        idx_by_id = {cid: i for i, cid in enumerate(ids)}
        ordered = []
        for c in chunks:
            i = idx_by_id.get(c["chunk_id"])
            if i is None:
                raise KeyError(f"vector missing for chunk_id={c['chunk_id']}")
            ordered.append(vecs[i])
        vectors = ordered
    else:
        raise FileNotFoundError(f"{vec_path} not found — run the embedding stage first")
    source_filename = f"{stem}.pdf"
    return {"chunks": chunks, "payload": payload}, vectors, source_filename


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=list(SOURCE_DIRS.keys()),
                    help="Which chunk pipeline to index from")
    ap.add_argument("--stem", help="Single doc stem (e.g. paper2)")
    ap.add_argument("--all", action="store_true",
                    help="Index every *.chunks.json in the source dir")
    ap.add_argument("--collection", default=None,
                    help="Override WEAVIATE_COLLECTION from .env")
    ap.add_argument("--client-id", default=None,
                    help="Optional per-tenant collection suffix")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    src_dir = SOURCE_DIRS[args.pipeline]
    if not src_dir.exists():
        raise SystemExit(f"Source dir does not exist: {src_dir}")
    if args.all:
        stems = [p.name.replace(".chunks.json", "")
                 for p in sorted(src_dir.glob("*.chunks.json"))]
    elif args.stem:
        stems = [args.stem]
    else:
        raise SystemExit("Provide --stem <name> or --all")

    store = WeaviateChunkStore(collection_name=args.collection,
                               client_id=args.client_id)
    print(f"▶ Weaviate collection: {store.collection_name}")
    print(f"  source dir: {src_dir}  ({args.pipeline} pipeline)")
    print(f"  stems     : {stems}")

    totals = {"indexed": 0, "failed": 0, "elapsed_s": 0.0}
    for stem in stems:
        try:
            chunks_payload, vectors, source_filename = _load_stem(src_dir, stem)
        except FileNotFoundError as e:
            print(f"  skip {stem}: {e}")
            continue
        t0 = time.perf_counter()
        result = store.index_chunks(
            chunks=chunks_payload["chunks"], vectors=vectors,
            source_filename=source_filename, chunking_strategy=args.pipeline,
        )
        dt = time.perf_counter() - t0
        totals["indexed"] += result["indexed"]
        totals["failed"]  += result["failed"]
        totals["elapsed_s"] += dt
        print(f"  ✓ {stem}: indexed={result['indexed']} "
              f"failed={result['failed']} in {dt:.2f}s")

    print()
    print(f"DONE — indexed {totals['indexed']} chunks "
          f"({totals['failed']} failed) into {store.collection_name} "
          f"in {totals['elapsed_s']:.2f}s total")


if __name__ == "__main__":
    main()
