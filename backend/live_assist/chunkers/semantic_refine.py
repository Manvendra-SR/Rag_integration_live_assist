"""Semantic refinement of oversized structural chunks.

When a structural chunk exceeds REFINE_THRESHOLD tokens, split it at within-chunk
topic-shift points found via sentence-level cosine similarity minima. Each
resulting sub-chunk respects REFINE_MIN_TOKENS.

Conservative defaults — only triggers on genuinely oversized chunks.
"""
from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from typing import Any

import numpy as np
import tiktoken
from nltk.tokenize import sent_tokenize

log = logging.getLogger(__name__)

ENC = tiktoken.encoding_for_model("gpt-4")
REFINE_THRESHOLD = 600   # only refine chunks above this many tokens
REFINE_MIN_TOKENS = 100  # each resulting sub-chunk must be ≥ this
REFINE_TARGET = 350      # aim for chunks of about this size after refine
SMOOTH_WINDOW = 3        # smooth adjacency similarities over this window
REFINE_OVERLAP_TOKENS = 60  # carry-forward tokens added to start of each post-first sub-chunk


def _tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def _split_into_sentences(text: str) -> list[str]:
    sents = []
    # Respect markdown structure: keep heading line as its own sentence
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("- ") or line.startswith("* "):
            sents.append(line)
        else:
            for s in sent_tokenize(line):
                if s.strip():
                    sents.append(s.strip())
    return sents


def _find_split_indices(sims: list[float], window: int = SMOOTH_WINDOW) -> list[int]:
    """Return indices AFTER which to split (i.e., position in the gap array)."""
    if len(sims) < 3:
        return []
    # Smoothing
    s = np.array(sims, dtype=float)
    smoothed = np.convolve(s, np.ones(window) / window, mode="same")
    # Find local minima
    minima = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            minima.append((i, smoothed[i]))
    # Sort by similarity ascending — the deepest dips are best split points
    minima.sort(key=lambda x: x[1])
    return [m[0] for m in minima]


def refine_one_chunk(chunk: dict[str, Any], embedder_fn) -> list[dict[str, Any]]:
    """Refine a single chunk if oversized. Returns list of sub-chunks (or [chunk] if not refined)."""
    text = chunk["text"]
    if chunk["token_count"] <= REFINE_THRESHOLD:
        return [chunk]

    sents = _split_into_sentences(text)
    if len(sents) < 4:
        return [chunk]  # not enough sentences to safely split

    # Embed sentences with bge-m3
    embs = embedder_fn(sents)
    if embs is None or len(embs) != len(sents):
        return [chunk]

    # Adjacent cosine similarities (already L2-normalized)
    sims = [float(np.dot(embs[i], embs[i+1])) for i in range(len(sents) - 1)]

    # Find topic-shift candidates
    minima = _find_split_indices(sims)
    if not minima:
        return [chunk]

    # Greedily pick split points that respect MIN_TOKENS on both sides
    # and aim for sub-chunks near REFINE_TARGET
    chosen_splits = []
    sent_token_counts = [_tokens(s) for s in sents]
    cumulative = np.cumsum(sent_token_counts)
    total_tokens = int(cumulative[-1])

    for m in minima:
        # m is the index in the gap array → splits after sentence m
        left_tokens = int(cumulative[m])
        right_tokens = total_tokens - left_tokens
        # Check against already-chosen splits
        existing = sorted(chosen_splits + [m])
        valid = True
        prev_end = -1
        for s in existing:
            piece = int(cumulative[s]) - (int(cumulative[prev_end]) if prev_end >= 0 else 0)
            if piece < REFINE_MIN_TOKENS:
                valid = False
                break
            prev_end = s
        last_piece = total_tokens - (int(cumulative[existing[-1]]) if existing else 0)
        if last_piece < REFINE_MIN_TOKENS:
            valid = False
        if valid:
            chosen_splits.append(m)

    if not chosen_splits:
        return [chunk]
    chosen_splits.sort()

    # Build sub-chunks WITH overlap between adjacent sub-chunks.
    # Each sub-chunk after the first carries the last REFINE_OVERLAP_TOKENS-worth
    # of sentences from the previous sub-chunk as a "context bridge" so the
    # arbitrary mid-section boundary doesn't strand antecedents.
    def _trailing_sents_for_overlap(start: int, end: int, budget: int) -> list[str]:
        """Walk backwards from `end` (exclusive) until ~budget tokens accumulated; return sents in order."""
        acc = []
        running = 0
        for i in range(end - 1, start - 1, -1):
            t = sent_token_counts[i]
            if running + t > budget and acc:
                break
            acc.append(i)
            running += t
            if running >= budget:
                break
        acc.sort()
        return [sents[i] for i in acc]

    sub_chunks = []
    prev = 0
    boundaries = chosen_splits + [len(sents) - 1]
    for k, s in enumerate(boundaries):
        core_sents = sents[prev:s+1]
        if k == 0:
            sub_text = "\n\n".join(core_sents)
            overlap_tokens = 0
        else:
            overlap_sents = _trailing_sents_for_overlap(0, prev, REFINE_OVERLAP_TOKENS)
            overlap_text = "\n\n".join(overlap_sents)
            sub_text = overlap_text + "\n\n" + "\n\n".join(core_sents) if overlap_sents else "\n\n".join(core_sents)
            overlap_tokens = _tokens(overlap_text)
        sub_count = _tokens(sub_text)
        new_chunk = deepcopy(chunk)
        new_chunk["chunk_id"] = str(uuid.uuid4())
        new_chunk["text"] = sub_text
        new_chunk["token_count"] = sub_count
        new_chunk["block_count"] = len(core_sents)
        new_chunk["refined_from"] = chunk["chunk_id"]
        new_chunk["refine_part"] = f"{k+1}/{len(boundaries)}"
        new_chunk["overlap_with_prev_tokens"] = overlap_tokens

        # Filter merged_sections: keep only those whose heading text actually
        # appears in this sub-chunk. The structural chunker emits merged
        # headings inline as "## <heading text>", so if the heading isn't in
        # sub_text the section's content isn't either. This prevents refined
        # sub-chunks from claiming to span sections they don't actually cover.
        parent_merged = chunk.get("merged_sections") or []
        if parent_merged:
            sub_text_lower = sub_text.lower()
            kept = []
            for sec_path in parent_merged:
                heading = (sec_path[-1] if sec_path else "").strip()
                if heading and heading.lower() in sub_text_lower:
                    kept.append(sec_path)
            new_chunk["merged_sections"] = kept

        sub_chunks.append(new_chunk)
        prev = s + 1

    return sub_chunks


def refine_chunks(chunks: list[dict[str, Any]], embedder_fn) -> list[dict[str, Any]]:
    """Apply semantic refinement to all oversized chunks. Re-wires prev/next pointers."""
    out: list[dict[str, Any]] = []
    n_refined = 0
    n_oversized = 0
    for c in chunks:
        if c["token_count"] > REFINE_THRESHOLD:
            n_oversized += 1
            sub = refine_one_chunk(c, embedder_fn)
            if len(sub) > 1:
                n_refined += 1
                log.info(f"  refined chunk pg{c['page_start']}-{c['page_end']} ({c['token_count']} tok) → {len(sub)} sub-chunks")
            out.extend(sub)
        else:
            out.append(c)

    # Re-wire prev/next pointers (ids changed for refined chunks)
    for i, c in enumerate(out):
        c["prev_chunk_id"] = out[i-1]["chunk_id"] if i > 0 else None
        c["next_chunk_id"] = out[i+1]["chunk_id"] if i + 1 < len(out) else None

    log.info(f"  refinement: {n_oversized} oversized chunks, {n_refined} actually split, "
             f"chunk count {len(chunks)} → {len(out)}")
    return out
