"""Context assembly for generation.

Adapted from the older RAG pipeline: dedupe reranked chunks, order them by
retrieval strength, cap the final context, and build a source manifest for
strict citation-aware generation.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict, defaultdict

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2800"))
CONTEXT_MAX_CHUNKS = int(os.getenv("CONTEXT_MAX_CHUNKS", "6"))
CONTEXT_MAX_CHUNKS_PER_DOC = int(os.getenv("CONTEXT_MAX_CHUNKS_PER_DOC", "6"))
CONTEXT_ORDERING = os.getenv("CONTEXT_ORDERING", "ranked")


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_signature(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()[:400]


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", (left or "").lower()))
    right_tokens = set(re.findall(r"\w+", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _score(chunk: dict) -> float:
    return float(
        chunk.get("rerank_score")
        or chunk.get("rrf_score")
        or chunk.get("semantic_score")
        or chunk.get("bm25_score")
        or 0.0
    )


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_ids = set()
    seen_signatures = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id in seen_ids:
            continue
        text = chunk.get("text") or chunk.get("content") or ""
        signature = _content_signature(text)
        if signature and signature in seen_signatures:
            continue
        if any(
            existing.get("doc_id") == chunk.get("doc_id")
            and abs(int(existing.get("page_start") or 0) - int(chunk.get("page_start") or 0)) <= 1
            and _overlap_ratio(existing.get("text") or "", text) > 0.82
            for existing in deduped
        ):
            continue
        if chunk_id:
            seen_ids.add(chunk_id)
        if signature:
            seen_signatures.add(signature)
        deduped.append(chunk)
    return deduped


def _sort_chunks(chunks: list[dict], ordering: str) -> list[dict]:
    if ordering == "document":
        return sorted(
            chunks,
            key=lambda item: (
                item.get("source_filename") or "",
                int(item.get("page_start") or 0),
                -_score(item),
            ),
        )
    return sorted(
        chunks,
        key=lambda item: (
            -_score(item),
            item.get("source_filename") or "",
            int(item.get("page_start") or 0),
        ),
    )


def _section(chunk: dict) -> str:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        return section_path
    return " > ".join(str(s) for s in section_path if s) or "Unknown"


def _retrieval_metrics(chunk: dict) -> dict:
    source_ranks = chunk.get("source_ranks") or {}
    return {
        "bm25_rank": source_ranks.get("bm25"),
        "semantic_rank": source_ranks.get("semantic"),
        "rrf_score": chunk.get("rrf_score"),
        "reranker_score": chunk.get("rerank_score"),
        "final_rank": chunk.get("rerank_rank") or chunk.get("final_rank"),
    }


def _format_chunk_block(chunk: dict, ordinal: int) -> str:
    score = _score(chunk)
    header = (
        f"[Chunk {ordinal} | prose score={score:.3f}] "
        f"file={chunk.get('source_filename') or chunk.get('doc_id')} "
        f"pages={chunk.get('page_start')}-{chunk.get('page_end')} "
        f"section={_section(chunk)}"
    )
    return f"{header}\n{_normalize_text(chunk.get('text') or chunk.get('content') or '')}"


def _excerpt_from_chunk(chunk: dict, max_chars: int = 520) -> str:
    text = _normalize_text(chunk.get("text") or chunk.get("content") or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rsplit(' ', 1)[0].strip()}..."


def assemble_context(
    reranked_payload: dict,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
    ordering: str = CONTEXT_ORDERING,
    max_chunks: int = CONTEXT_MAX_CHUNKS,
) -> dict:
    chunks = deduplicate_chunks(reranked_payload.get("results", []))
    ordered = _sort_chunks(chunks, ordering=ordering)

    selected: list[dict] = []
    source_map = OrderedDict()
    per_doc_counts: dict[str, int] = defaultdict(int)
    used_tokens = 0

    for chunk in ordered:
        if len(selected) >= max_chunks:
            break
        doc_id = chunk.get("doc_id") or chunk.get("source_filename") or ""
        if per_doc_counts[doc_id] >= CONTEXT_MAX_CHUNKS_PER_DOC:
            continue
        block = _format_chunk_block(chunk, len(selected) + 1)
        block_tokens = estimate_tokens(block)
        if selected and used_tokens + block_tokens > token_budget:
            continue
        if not selected and block_tokens > token_budget:
            tokens = TOKEN_PATTERN.findall(block)[:token_budget]
            block = " ".join(tokens)
            block_tokens = estimate_tokens(block)

        payload = dict(chunk)
        payload.update(_retrieval_metrics(chunk))
        payload["context_text"] = block
        payload["context_tokens"] = block_tokens
        selected.append(payload)
        used_tokens += block_tokens
        per_doc_counts[doc_id] += 1

        source_map[chunk["chunk_id"]] = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk.get("doc_id") or "",
            "source_filename": chunk.get("source_filename") or "",
            "page": int(chunk.get("page_start") or 0),
            "page_start": int(chunk.get("page_start") or 0),
            "page_end": int(chunk.get("page_end") or 0),
            "chunk_index": int(chunk.get("final_rank") or chunk.get("rerank_rank") or len(selected)),
            "section": _section(chunk),
            "excerpt": _excerpt_from_chunk(chunk),
            **_retrieval_metrics(chunk),
            "reference_like": chunk.get("reference_like", False),
        }

    return {
        "query": reranked_payload["query"],
        "ordering": ordering,
        "token_budget": token_budget,
        "estimated_tokens": used_tokens,
        "selected_chunk_count": len(selected),
        "context": "\n\n".join(chunk["context_text"] for chunk in selected),
        "selected_chunks": selected,
        "sources": list(source_map.values()),
    }
