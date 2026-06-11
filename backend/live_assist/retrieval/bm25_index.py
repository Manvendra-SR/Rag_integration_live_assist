"""BM25 index + search (Okapi BM25).

Persists a single pickle per chunk-source dir under runtime/bm25/. Re-build
with `scripts/build_bm25.py` whenever new chunks land in chunks_anthropic or
chunks_semantic. Search is a plain ranked list — no Weaviate involvement.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from math import log as math_log
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # tiny fallback
        def __init__(self, corpus, k1: float = 1.5, b: float = 0.75):
            self.corpus = corpus
            self.k1, self.b = k1, b
            self.doc_count = len(corpus)
            self.doc_freqs, self.doc_len = [], []
            self.idf = {}
            self.avgdl = (sum(len(d) for d in corpus) / max(len(corpus), 1)) if corpus else 0
            nd = Counter()
            for doc in corpus:
                f = Counter(doc)
                self.doc_freqs.append(f)
                self.doc_len.append(len(doc))
                for w in f:
                    nd[w] += 1
            for w, freq in nd.items():
                self.idf[w] = math_log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))

        def get_scores(self, q_tokens):
            scores = []
            for f, dl in zip(self.doc_freqs, self.doc_len):
                s = 0.0
                for t in q_tokens:
                    if t not in f:
                        continue
                    tf = f[t]
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                    s += self.idf.get(t, 0.0) * (num / den)
                scores.append(s)
            return scores


TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*|\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?|[A-Z]{2,}\d+[A-Z0-9-]*",
    re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text or "")]


@dataclass
class BM25Doc:
    chunk_id: str
    doc_id: str
    source_filename: str
    section_path: list[str]
    text: str
    page_start: int
    page_end: int
    user_id: str
    text_with_context: str | None = None


def _coerce_chunk(c: dict, source_filename: str) -> BM25Doc:
    sp = c.get("section_path") or []
    if isinstance(sp, str):
        sp = [sp]
    return BM25Doc(
        chunk_id=c["chunk_id"],
        doc_id=c.get("document_id") or c.get("doc_id") or "",
        source_filename=c.get("source_filename") or source_filename,
        section_path=[str(s) for s in sp],
        text=c.get("text") or c.get("content") or "",
        page_start=int(c.get("page_start") or 0),
        page_end=int(c.get("page_end") or 0),
        user_id=c.get("user_id") or "",
        text_with_context=c.get("text_with_context"),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_bm25_from_dir(chunks_dir: Path, index_out: Path,
                        docs_out: Path) -> dict:
    """Walk all *.chunks.json in chunks_dir, build a BM25 index across them."""
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if not files:
        raise FileNotFoundError(f"No chunks files in {chunks_dir}")
    all_docs: list[BM25Doc] = []
    for cp in files:
        payload = json.loads(cp.read_text(encoding="utf-8"))
        stem = cp.name.replace(".chunks.json", "")
        source_filename = f"{stem}.pdf"
        for c in payload.get("chunks", []):
            all_docs.append(_coerce_chunk(c, source_filename))
    if not all_docs:
        raise ValueError(f"No chunks found in {chunks_dir}")
    # Include section_path words in the BM25 token stream — boost topical match
    corpus = [
        tokenize(" ".join(d.section_path) + " " + (d.text_with_context or d.text)) for d in all_docs
    ]
    t0 = time.perf_counter()
    bm25 = BM25Okapi(corpus)
    elapsed = time.perf_counter() - t0
    index_out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    with index_out.open("wb") as f:
        pickle.dump(bm25, f)
    docs_out.write_text(
        json.dumps([asdict(d) for d in all_docs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"BM25 indexed {len(all_docs)} chunks from {len(files)} docs "
             f"in {elapsed:.2f}s → {index_out.name}")
    return {"chunk_count": len(all_docs), "docs": len(files),
            "elapsed_s": round(elapsed, 3),
            "index_path": str(index_out), "docs_path": str(docs_out)}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _load(index_path: Path, docs_path: Path) -> tuple[Any, list[BM25Doc]]:
    with index_path.open("rb") as f:
        bm25 = pickle.load(f)
    docs_raw = json.loads(docs_path.read_text(encoding="utf-8"))
    docs = [BM25Doc(**d) for d in docs_raw]
    return bm25, docs


def search_bm25(query: str, index_path: Path, docs_path: Path,
                limit: int = 10, filters: dict | None = None) -> list[dict]:
    bm25, docs = _load(index_path, docs_path)
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    # Apply property-style filters (doc_id / source_filename) post-hoc
    def passes(d: BM25Doc) -> bool:
        if not filters:
            return True
        if filters.get("doc_id") and d.doc_id != filters["doc_id"]:
            return False
        if filters.get("source_filename") and d.source_filename != filters["source_filename"]:
            return False
        if filters.get("page_start") is not None and d.page_end < int(filters["page_start"]):
            return False
        if filters.get("page_end") is not None and d.page_start > int(filters["page_end"]):
            return False
        if filters.get("user_id") and d.user_id != filters["user_id"]:
            return False
        if filters.get("chunking_strategy"):
            # BM25Doc doesn't carry chunking_strategy, but we can check
            # source_filename or other available fields. Since BM25 index
            # is built per-pipeline (one .pkl per pipeline name), this
            # filter is mostly a safety net.
            pass
        return True

    paired = [
        (s, d) for s, d in zip(scores, docs) if s > 0 and passes(d)
    ]
    paired.sort(key=lambda x: x[0], reverse=True)
    paired = paired[: max(limit, 1)]
    out = []
    for s, d in paired:
        out.append({
            "chunk_id":        d.chunk_id,
            "doc_id":          d.doc_id,
            "source_filename": d.source_filename,
            "section_path":    d.section_path,
            "text":            d.text,
            "text_with_context": d.text_with_context,
            "page_start":      d.page_start,
            "page_end":        d.page_end,
            "bm25_score":      round(float(s), 6),
        })
    return out
