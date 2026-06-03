"""Section-aware recursive semantic chunker (T-08, Pipeline B).

Reads a parsed.json (output of `parsers/docling_parser.py`) and produces
chunks by:

  1. Grouping blocks by section_path (every section_heading starts a new
     section bucket — exactly like the structural chunker).
  2. Inside each section, splitting the joined text into sentences.
  3. Embedding every sentence via `clients.get_embedder()` (one call, batched).
  4. Walking sentences in order — start a new chunk when:
       - token-count is over MAX_TOKENS (700), OR
       - we've passed MIN_TOKENS (100) AND the cosine similarity between this
         sentence and the previous one is below SIM_BREAK_THRESHOLD.
  5. Recursively splitting any group that still exceeds MAX_TOKENS at its
     lowest-similarity valley until all groups fit.
  6. Merging any tail group smaller than MIN_TOKENS into its predecessor.

This is the port of RAG_project_old/Parsing_Agent/semantic_chunker.py,
simplified to:
  - use token counts (tiktoken) instead of word counts
  - consume the flat-blocks parsed.json schema
  - call clients.get_embedder() so the embedding provider follows .env
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

import tiktoken

log = logging.getLogger(__name__)

ENC = tiktoken.encoding_for_model("gpt-4")

# Token-bounds — per user spec
MIN_TOKENS = 100
MAX_TOKENS = 700

# Similarity drop that signals a topic break (only fires after MIN_TOKENS)
SIM_BREAK_THRESHOLD = 0.58
# Stronger break — fires earlier (before MIN_TOKENS) only on clear topic shifts
STRONG_BREAK_THRESHOLD = 0.42


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(ENC.encode(text, disallowed_special=()))


def _split_sentences(text: str) -> list[str]:
    """Tokenize text into sentences. NLTK if available, regex fallback."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        out = [s.strip() for s in sent_tokenize(t) if s.strip()]
        if out:
            return out
    except Exception:
        pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SemanticChunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str]
    page_start: int
    page_end: int
    token_count: int
    sentence_count: int
    source_block_ids: list[str] = field(default_factory=list)
    source_chunk_id: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunking_strategy: str = "semantic_recursive"
    coherence_score: float = 0.0
    chunk_type: str = "prose"   # 'prose' | 'figure' (tables are excluded entirely)
    caption: str | None = None
    image_path: str | None = None


def _build_table_chunks_semantic(blocks: list[dict], doc_id: str) -> list[SemanticChunk]:
    """One SemanticChunk per table — atomic. Markdown is not split by sentences."""
    out: list[SemanticChunk] = []
    for b in blocks:
        if b.get("type") != "table":
            continue
        md = b.get("metadata") or {}
        markdown = (md.get("markdown") or "").strip()
        cap = (md.get("caption") or "").strip()
        if not markdown and not cap:
            continue
        page = b.get("page") or 0
        parts = []
        if cap:
            parts.append(f"{cap} (page {page})")
        else:
            parts.append(f"[TABLE page {page}]")
        if markdown:
            parts.append(markdown)
        text = "\n\n".join(parts).strip()
        if not text:
            continue
        out.append(SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=b.get("section_path") or [],
            page_start=page, page_end=page,
            token_count=count_tokens(text),
            sentence_count=1,   # atomic; markdown isn't sentence-split
            source_block_ids=[b.get("block_id", "")],
            chunking_strategy="table_atomic",
            coherence_score=1.0,
            chunk_type="table",
            caption=cap or None,
        ))
    return out


def _build_figure_chunks_semantic(blocks: list[dict], doc_id: str) -> list[SemanticChunk]:
    """One SemanticChunk per figure that has a VLM description and/or caption.

    Mirrors `structural_chunker._build_figure_chunks` so both pipelines emit
    figure chunks the same way. Tables are intentionally skipped here too —
    they're excluded from chunking entirely per the new spec.
    """
    out: list[SemanticChunk] = []
    for b in blocks:
        if b.get("type") != "figure":
            continue
        md = b.get("metadata") or {}
        desc = (md.get("vlm_description") or "").strip()
        cap = (md.get("caption") or "").strip()
        path = md.get("image_path") or ""
        if not desc and not cap:
            continue
        page = b.get("page") or 0
        parts = []
        if cap:
            parts.append(f"{cap} (page {page})")
        else:
            parts.append(f"[FIGURE page {page}]")
        if desc:
            parts.append(desc)
        text = "\n\n".join(parts).strip()
        if not text:
            continue
        out.append(SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=b.get("section_path") or [],
            page_start=page, page_end=page,
            token_count=count_tokens(text),
            sentence_count=max(1, len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])),
            source_block_ids=[b.get("block_id", "")],
            chunking_strategy="figure",
            coherence_score=1.0,
            chunk_type="figure",
            caption=cap or None,
            image_path=path or None,
        ))
    return out


# ---------------------------------------------------------------------------
# Section grouping (same boundary rule as structural_chunker)
# ---------------------------------------------------------------------------
def _group_blocks_by_section(blocks: list[dict]) -> list[dict]:
    """Return a list of sections; each section is {
        section_path, page_start, page_end, text, source_block_ids
    }."""
    sections: list[dict] = []
    cur_path: list[str] = []
    cur: dict | None = None

    def _open(path: list[str]) -> dict:
        return {"section_path": list(path), "page_start": None,
                "page_end": None, "parts": [], "source_block_ids": []}

    for b in blocks:
        t = b.get("type")
        if t == "section_heading":
            if cur:
                sections.append(cur)
            cur_path = (b.get("section_path") or [])[:]
            if not cur_path:
                cur_path = [b.get("text") or ""]
            cur = _open(cur_path)
            # include heading in text so retrieval sees it
            cur["parts"].append(f"## {b.get('text','').strip()}")
            cur["source_block_ids"].append(b.get("block_id"))
            pg = b.get("page")
            if pg is not None:
                cur["page_start"] = pg
                cur["page_end"] = pg
            continue
        if t in ("paragraph", "list_item", "caption"):
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if cur is None:
                cur = _open(cur_path or ["(unsectioned)"])
            prefix = "- " if t == "list_item" else ""
            cur["parts"].append(f"{prefix}{text}")
            cur["source_block_ids"].append(b.get("block_id"))
            pg = b.get("page")
            if pg is not None:
                cur["page_start"] = cur["page_start"] or pg
                cur["page_end"] = pg
        elif t == "equation":
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if cur is None:
                cur = _open(cur_path or ["(unsectioned)"])
            cur["parts"].append(f"$$ {text} $$")
            cur["source_block_ids"].append(b.get("block_id"))
        # figures + tables intentionally skipped from prose chunking — they
        # live as separate blocks with their own VLM-extracted content and
        # are referenced via source_block_ids when needed.
    if cur:
        sections.append(cur)

    out = []
    for s in sections:
        text = "\n\n".join(s["parts"]).strip()
        if not text:
            continue
        out.append({
            "section_path": s["section_path"],
            "page_start": s["page_start"] or 1,
            "page_end": s["page_end"] or s["page_start"] or 1,
            "text": text,
            "source_block_ids": s["source_block_ids"],
        })
    return out


# ---------------------------------------------------------------------------
# Recursive semantic split
# ---------------------------------------------------------------------------
def _initial_grouping(sentences: list[str], sims: list[float],
                      tokens_per_sent: list[int]) -> list[list[int]]:
    """Greedy left-to-right grouping with similarity-based breaks."""
    if not sentences:
        return []
    groups: list[list[int]] = []
    cur: list[int] = [0]
    cur_tok = tokens_per_sent[0]
    for i in range(1, len(sentences)):
        sim = sims[i - 1]  # similarity between i-1 and i
        next_tok = tokens_per_sent[i]
        should_break = False
        if cur_tok + next_tok > MAX_TOKENS:
            should_break = True
        elif cur_tok >= MIN_TOKENS and sim < SIM_BREAK_THRESHOLD:
            should_break = True
        elif sim < STRONG_BREAK_THRESHOLD and cur_tok >= MIN_TOKENS // 2:
            should_break = True
        if should_break:
            groups.append(cur)
            cur = [i]
            cur_tok = next_tok
        else:
            cur.append(i)
            cur_tok += next_tok
    if cur:
        groups.append(cur)
    return groups


def _recursive_split(groups: list[list[int]], sims: list[float],
                     tokens_per_sent: list[int]) -> list[list[int]]:
    """If any group is still > MAX_TOKENS, split it at its lowest-sim valley."""
    refined: list[list[int]] = []
    for g in groups:
        toks = sum(tokens_per_sent[i] for i in g)
        if toks <= MAX_TOKENS or len(g) <= 1:
            refined.append(g)
            continue
        # find lowest cosine valley inside the group (between consecutive sentences)
        valleys = []
        for k in range(len(g) - 1):
            valleys.append((sims[g[k]], k + 1))
        if not valleys:
            refined.append(g)
            continue
        valleys.sort(key=lambda v: v[0])
        split_at = valleys[0][1]
        if split_at <= 0 or split_at >= len(g):
            refined.append(g)
            continue
        left, right = g[:split_at], g[split_at:]
        refined.extend(_recursive_split([left, right], sims, tokens_per_sent))
    return refined


def _merge_undersized_tail(groups: list[list[int]],
                           tokens_per_sent: list[int]) -> list[list[int]]:
    """Merge any group < MIN_TOKENS into the PREVIOUS one (or next if first)."""
    if not groups:
        return groups
    out: list[list[int]] = []
    for g in groups:
        toks = sum(tokens_per_sent[i] for i in g)
        if toks < MIN_TOKENS and out:
            out[-1].extend(g)
        elif toks < MIN_TOKENS and not out:
            # First group is sub-floor; carry forward — will merge with next
            out.append(g)
        else:
            if out and sum(tokens_per_sent[i] for i in out[-1]) < MIN_TOKENS:
                out[-1].extend(g)
            else:
                out.append(g)
    return out


def _section_to_chunks(
    section: dict,
    doc_id: str,
    embedder,
    *,
    force_semantic: bool = False,
) -> list[SemanticChunk]:
    sentences = _split_sentences(section["text"])
    if not sentences:
        return []
    # If the entire section is small, emit it as ONE chunk (no embedding needed)
    total_toks = sum(count_tokens(s) for s in sentences)
    if not force_semantic and total_toks <= MAX_TOKENS and total_toks >= MIN_TOKENS:
        return [SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=" ".join(sentences),
            section_path=section["section_path"],
            page_start=section["page_start"],
            page_end=section["page_end"],
            token_count=total_toks,
            sentence_count=len(sentences),
            source_block_ids=section["source_block_ids"],
            source_chunk_id=section.get("source_chunk_id"),
            coherence_score=1.0,
        )]
    if total_toks < MIN_TOKENS:
        # Tiny section — still emit, downstream merge will handle it
        return [SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=" ".join(sentences),
            section_path=section["section_path"],
            page_start=section["page_start"],
            page_end=section["page_end"],
            token_count=total_toks,
            sentence_count=len(sentences),
            source_block_ids=section["source_block_ids"],
            source_chunk_id=section.get("source_chunk_id"),
            coherence_score=1.0,
        )]

    # Embed every sentence ONCE
    embs = embedder.embed(sentences)
    tokens_per_sent = [count_tokens(s) for s in sentences]
    sims = [_cosine(embs[i], embs[i + 1]) for i in range(len(sentences) - 1)]

    groups = _initial_grouping(sentences, sims, tokens_per_sent)
    groups = _recursive_split(groups, sims, tokens_per_sent)
    groups = _merge_undersized_tail(groups, tokens_per_sent)

    chunks: list[SemanticChunk] = []
    for g in groups:
        text = " ".join(sentences[i] for i in g)
        toks = sum(tokens_per_sent[i] for i in g)
        # Coherence = avg cosine within group
        if len(g) > 1:
            inner = [sims[g[k]] for k in range(len(g) - 1)]
            coh = round(sum(inner) / len(inner), 4) if inner else 1.0
        else:
            coh = 1.0
        chunks.append(SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=section["section_path"],
            page_start=section["page_start"],
            page_end=section["page_end"],
            token_count=toks,
            sentence_count=len(g),
            source_block_ids=section["source_block_ids"],
            source_chunk_id=section.get("source_chunk_id"),
            coherence_score=coh,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def chunk_parsed_doc_semantic(parsed: dict) -> list[SemanticChunk]:
    """Top-level: parsed.json dict in, list of SemanticChunk out.

    Emits two kinds of chunks:
      - prose chunks  (semantic recursive grouping within each section)
      - figure chunks (one per figure with caption + VLM description)

    Tables are excluded from chunking entirely (per spec). Sub-100-token
    prose chunks merge with neighbouring PROSE chunks only — figures and
    tables are never used to satisfy the 100-token floor.
    """
    from live_assist.clients import get_embedder
    embedder = get_embedder()
    log.info(f"semantic_chunker using embedder: {embedder.name}")

    doc_id = parsed.get("doc_id") or str(uuid.uuid4())
    blocks = parsed.get("blocks", [])
    sections = _group_blocks_by_section(blocks)
    log.info(f"  {len(sections)} sections to chunk semantically")

    # Prose chunks (semantic groups within each section)
    prose_chunks: list[SemanticChunk] = []
    for sec in sections:
        prose_chunks.extend(_section_to_chunks(sec, doc_id, embedder))

    # Merge sub-floor PROSE chunks only — figures are never pulled in here
    prose_chunks = _merge_under_min_chunks(prose_chunks)

    # Asset chunks (atomic): one per figure, one per table
    figure_chunks = _build_figure_chunks_semantic(blocks, doc_id)
    table_chunks  = _build_table_chunks_semantic(blocks, doc_id)
    log.info(f"  prose_chunks={len(prose_chunks)}  "
             f"table_chunks={len(table_chunks)}  figure_chunks={len(figure_chunks)}")

    # Interleave by page; within the same page: prose → table → figure
    _type_order = {"prose": 0, "table": 1, "figure": 2}
    all_chunks = sorted(
        prose_chunks + table_chunks + figure_chunks,
        key=lambda c: (c.page_start, _type_order.get(c.chunk_type, 9)),
    )

    # Wire prev/next pointers over the combined reading-order sequence
    for i, c in enumerate(all_chunks):
        c.prev_chunk_id = all_chunks[i - 1].chunk_id if i > 0 else None
        c.next_chunk_id = all_chunks[i + 1].chunk_id if i + 1 < len(all_chunks) else None
    return all_chunks


def _chunks_to_sections(chunks_blob: dict | list[dict]) -> tuple[str, list[dict]]:
    """Treat existing structural chunks as semantic containers.

    This keeps parser/structure cleanup from `scripts/chunk.py` but replaces
    final section-boundary chunks with sentence-similarity chunks.
    """
    chunks = chunks_blob.get("chunks", chunks_blob) if isinstance(chunks_blob, dict) else chunks_blob
    if not isinstance(chunks, list):
        raise TypeError("chunks input must be a list or a dict with a 'chunks' list")
    doc_id = ""
    sections: list[dict] = []
    for c in chunks:
        text = (c.get("text") or c.get("text_with_context") or "").strip()
        if not text:
            continue
        if not doc_id:
            doc_id = c.get("doc_id") or ""
        source_block_ids = c.get("source_block_ids") or []
        sections.append({
            "section_path": c.get("section_path") or ["(chunk container)"],
            "page_start": c.get("page_start") or 1,
            "page_end": c.get("page_end") or c.get("page_start") or 1,
            "text": text,
            "source_block_ids": source_block_ids,
            "source_chunk_id": c.get("chunk_id"),
        })
    return doc_id or str(uuid.uuid4()), sections


def chunk_structural_chunks_semantic(chunks_blob: dict | list[dict]) -> list[SemanticChunk]:
    """Semantic-split the chunks produced by `scripts/chunk.py`.

    Unlike parsed-mode chunking, this forces sentence-level similarity checks
    even for containers that are already below MAX_TOKENS, so output chunks are
    semantic retrieval units instead of just section-boundary units.
    """
    from live_assist.clients import get_embedder
    embedder = get_embedder()
    log.info(f"semantic_chunker using embedder: {embedder.name}")

    doc_id, sections = _chunks_to_sections(chunks_blob)
    log.info(f"  {len(sections)} structural chunk containers to split semantically")

    all_chunks: list[SemanticChunk] = []
    for sec in sections:
        all_chunks.extend(_section_to_chunks(sec, doc_id, embedder, force_semantic=True))

    all_chunks = _merge_under_min_chunks(all_chunks)

    for i, c in enumerate(all_chunks):
        c.prev_chunk_id = all_chunks[i - 1].chunk_id if i > 0 else None
        c.next_chunk_id = all_chunks[i + 1].chunk_id if i + 1 < len(all_chunks) else None
    return all_chunks


def _merge_two_chunks(left: SemanticChunk, right: SemanticChunk) -> SemanticChunk:
    left.text = f"{left.text}\n\n{right.text}".strip()
    left.token_count = count_tokens(left.text)
    left.sentence_count += right.sentence_count
    left.page_start = min(left.page_start, right.page_start)
    left.page_end = max(left.page_end, right.page_end)
    seen = set(left.source_block_ids)
    for block_id in right.source_block_ids:
        if block_id not in seen:
            left.source_block_ids.append(block_id)
            seen.add(block_id)
    if left.coherence_score and right.coherence_score:
        left.coherence_score = round((left.coherence_score + right.coherence_score) / 2, 4)
    return left


def _merge_under_min_chunks(chunks: list[SemanticChunk]) -> list[SemanticChunk]:
    """Final floor pass across section/container boundaries."""
    if not chunks:
        return chunks
    out: list[SemanticChunk] = []
    for c in chunks:
        if out and (c.token_count < MIN_TOKENS or out[-1].token_count < MIN_TOKENS):
            _merge_two_chunks(out[-1], c)
        else:
            out.append(c)
    return out


def save_semantic_chunks(chunks: list[SemanticChunk], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    payload = {
        "chunk_count": len(chunks),
        "total_tokens": sum(c.token_count for c in chunks),
        "chunking_strategy": "section-aware recursive semantic chunking",
        "token_floor": MIN_TOKENS,
        "token_cap": MAX_TOKENS,
        "chunks": [asdict(c) for c in chunks],
    }
    out_path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path
