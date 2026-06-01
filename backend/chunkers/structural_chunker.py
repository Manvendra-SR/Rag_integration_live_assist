"""Structural chunking — prose + standalone figure + standalone table chunks.

Rules
-----
1. **Prose chunks** = one section's prose (section_heading + paragraphs +
   list_items + equations + captions). Tables and figures NEVER inline into
   prose chunks.
2. **No upper token cap on prose.** Sections stay whole even at 1500+ tokens.
3. **100-token floor, merge upward — prose-only.** A sub-100-token prose
   chunk merges into the previous PROSE chunk. Figure/table chunks are
   never used to satisfy the floor — that would be a waste of asset content.
4. **Figure chunks** (one per figure): caption + `metadata.vlm_description`.
   `chunk_type="figure"`. Same rule for any image-class (bar_chart, line_chart,
   pie_chart, scatter_plot, photograph, flow_chart, etc.) — each picture-block
   becomes one independent chunk.
5. **Table chunks** (one per table): caption + `metadata.markdown`. Atomic —
   never sentence-split because markdown rows aren't sentences.
   `chunk_type="table"`.
6. **Anthropic pipeline** (`scripts/enrich.py`) runs the LLM contextual-prefix
   pass over EVERY chunk type (prose / figure / table), so the table markdown
   and figure descriptions all get their own retrieval-time "situate in doc"
   summary prepended.
7. **Semantic pipeline** (`scripts/chunk_semantic.py`) groups prose by
   cosine valleys; figure and table chunks are atomic (not semantically split).
8. No mid-paragraph splitting ever.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import tiktoken

ENC = tiktoken.encoding_for_model("gpt-4")  # ~close approximation to bge-m3 tokens

MIN_TOKENS = 100  # user-requested: never emit chunks under 100 tokens; merge upward into previous chunk


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(ENC.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str]
    page_start: int
    page_end: int
    token_count: int
    block_count: int
    block_types: dict[str, int]
    source_block_ids: list[str] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)  # ids of sub-floor chunks merged in
    merged_sections: list[list[str]] = field(default_factory=list)  # section_paths of merged-in chunks (in merge order)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_type: str = "prose"   # 'prose' | 'figure' (tables are excluded entirely)
    caption: str | None = None  # populated for figure chunks
    image_path: str | None = None  # populated for figure chunks


def _format_block(block: dict[str, Any]) -> str:
    """Format a single block into markdown-ish text for the PROSE stream.

    Tables and figures return "" so they never enter the prose chunk text —
    figures get their own chunks via `_build_figure_chunks`; tables are
    excluded from chunking entirely per the new spec.
    """
    text = (block.get("text") or "").strip()
    t = block.get("type", "paragraph")
    # Excluded from prose chunks:
    if t == "table":
        return ""
    if t == "figure":
        return ""
    if not text:
        return ""
    if t == "section_heading":
        return f"## {text}"
    if t == "list_item":
        return f"- {text}"
    if t == "equation":
        return f"$$ {text} $$" if text else f"[EQUATION pg{block.get('page')} — see surrounding text]"
    if t == "caption":
        # Captions for figures/tables now live with their parent block via
        # `attach_orphan_captions`. We still emit them in prose as italic
        # markers because they reference content the reader can look up.
        return f"_Caption: {text}_"
    return text


def _build_table_chunks(blocks: list[dict[str, Any]], doc_id: str) -> list[Chunk]:
    """One chunk per table — atomic, never sentence-split.

    Chunk text = "TABLE N. caption (page X)\\n\\n<markdown>". The Anthropic
    pipeline's contextual-prefix pass will add a 2-3 sentence LLM-written
    framing on top at enrichment time (because that pass is type-agnostic).
    """
    out: list[Chunk] = []
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
        out.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=b.get("section_path") or [],
            page_start=page, page_end=page,
            token_count=count_tokens(text),
            block_count=1,
            block_types={"table": 1},
            source_block_ids=[b.get("block_id", "")],
            chunk_type="table",
            caption=cap or None,
        ))
    return out


def _build_figure_chunks(blocks: list[dict[str, Any]], doc_id: str) -> list[Chunk]:
    """One chunk per figure whose VLM description carries real content.

    Chunk text = "FIGURE N. caption (page X)\\n\\nVLM description ...".
    Figures with a VLM description shorter than MIN_TOKENS are still emitted
    if they have a caption (caption + description usually clears the bar);
    truly empty figures are skipped.
    """
    out: list[Chunk] = []
    for b in blocks:
        if b.get("type") != "figure":
            continue
        md = b.get("metadata") or {}
        desc = (md.get("vlm_description") or "").strip()
        cap  = (md.get("caption") or "").strip()
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
        out.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=b.get("section_path") or [],
            page_start=page, page_end=page,
            token_count=count_tokens(text),
            block_count=1,
            block_types={"figure": 1},
            source_block_ids=[b.get("block_id", "")],
            chunk_type="figure",
            caption=cap or None,
            image_path=path or None,
        ))
    return out


def _is_section_boundary(block: dict[str, Any]) -> bool:
    return block.get("type") == "section_heading"


def chunk_parsed_doc(parsed: dict[str, Any]) -> list[Chunk]:
    """Walk blocks, group by section, merge sub-floor sections upward."""
    doc_id = parsed.get("doc_id", "unknown")
    blocks = parsed.get("blocks", [])

    raw_chunks: list[Chunk] = []
    cur_lines: list[str] = []
    cur_pages: set[int] = set()
    cur_types: list[str] = []
    cur_block_ids: list[str] = []
    cur_section_path: list[str] = []

    def flush(_force: bool = False) -> None:
        nonlocal cur_lines, cur_pages, cur_types, cur_block_ids, cur_section_path
        if not cur_lines:
            return
        text = "\n\n".join(line for line in cur_lines if line)
        toks = count_tokens(text)
        if toks == 0:
            cur_lines = []
            cur_pages.clear()
            cur_types.clear()
            cur_block_ids.clear()
            return
        block_type_counts: dict[str, int] = {}
        for t in cur_types:
            block_type_counts[t] = block_type_counts.get(t, 0) + 1
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            section_path=cur_section_path.copy(),
            page_start=min(cur_pages) if cur_pages else 0,
            page_end=max(cur_pages) if cur_pages else 0,
            token_count=toks,
            block_count=len(cur_block_ids),
            block_types=block_type_counts,
            source_block_ids=cur_block_ids.copy(),
        )
        raw_chunks.append(chunk)
        cur_lines = []
        cur_pages.clear()
        cur_types.clear()
        cur_block_ids.clear()

    for block in blocks:
        if _is_section_boundary(block):
            # Flush previous section before starting new one
            flush()
            cur_section_path = block.get("section_path", []) or []
            line = _format_block(block)
            if line:
                cur_lines.append(line)
                if block.get("page") is not None:
                    cur_pages.add(block["page"])
                cur_types.append("section_heading")
                cur_block_ids.append(block.get("block_id", ""))
            continue
        line = _format_block(block)
        if not line:
            continue
        if not cur_section_path and block.get("section_path"):
            cur_section_path = block["section_path"]
        cur_lines.append(line)
        if block.get("page") is not None:
            cur_pages.add(block["page"])
        cur_types.append(block.get("type", "paragraph"))
        cur_block_ids.append(block.get("block_id", ""))

    flush()  # final

    # ----- Second pass — merge sub-floor PROSE chunks upward -----
    # Per the new spec: figures/tables never count toward the 100-token
    # floor, and they're never used to satisfy a merge. raw_chunks is
    # already prose-only because `_format_block` returns "" for figures
    # and tables — so this pass operates purely on prose.
    merged: list[Chunk] = []
    for c in raw_chunks:
        if c.token_count < MIN_TOKENS and merged:
            prev = merged[-1]
            # Merge text (the merged chunk's own ## heading lives inside c.text)
            prev.text = prev.text + "\n\n" + c.text
            prev.token_count = count_tokens(prev.text)
            prev.page_end = max(prev.page_end, c.page_end)
            prev.block_count += c.block_count
            for t, n in c.block_types.items():
                prev.block_types[t] = prev.block_types.get(t, 0) + n
            prev.source_block_ids.extend(c.source_block_ids)
            prev.merged_from.append(c.chunk_id)
            if c.section_path and c.section_path != prev.section_path:
                prev.merged_sections.append(c.section_path)
        elif c.token_count < MIN_TOKENS and not merged:
            # First chunk is sub-floor — keep it; the NEXT prose chunk will
            # pull this content forward on its own merge pass (see fallback
            # below). For simplicity we just keep it; downstream filters can
            # treat it as a context-anchor.
            merged.append(c)
        else:
            merged.append(c)

    # ----- Build figure + table chunks and interleave by page -----
    figure_chunks = _build_figure_chunks(blocks, doc_id)
    table_chunks  = _build_table_chunks(blocks, doc_id)
    # Sort key: by page, then prose → table → figure within the same page
    _type_order = {"prose": 0, "table": 1, "figure": 2}
    all_chunks = sorted(
        merged + table_chunks + figure_chunks,
        key=lambda c: (c.page_start, _type_order.get(c.chunk_type, 9)),
    )

    # Wire prev/next pointers across the combined reading-order sequence
    for i, c in enumerate(all_chunks):
        c.prev_chunk_id = all_chunks[i-1].chunk_id if i > 0 else None
        c.next_chunk_id = all_chunks[i+1].chunk_id if i + 1 < len(all_chunks) else None

    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_count": len(chunks),
        "total_tokens": sum(c.token_count for c in chunks),
        "chunks": [asdict(c) for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
