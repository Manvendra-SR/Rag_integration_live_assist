"""Build strict grounded generation prompts from an assembled context package."""
from __future__ import annotations

import json
import os

GENERATION_MODE = os.getenv("GENERATION_MODE", "strict")


def build_system_prompt(mode: str = GENERATION_MODE) -> str:
    base = (
        "You are a retrieval-grounded answer generator.\n"
        "Read the provided context and write a clean, useful answer for the user.\n"
        "Synthesize the retrieved evidence instead of echoing chunk text mechanically.\n"
        "Use only facts supported by the provided context and source manifest.\n\n"
        "Context format:\n"
        "Each chunk starts with '[Chunk N | prose score=...] file=... pages=... section=...'.\n"
        "Higher scores generally indicate stronger relevance, but lower-ranked chunks may contain exact details.\n\n"
        "Return strict JSON only, with no markdown and no prose outside JSON.\n"
        "Required JSON schema:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "confidence": 0.0,\n'
        '  "citations": [\n'
        "    {\n"
        '      "chunk_id": "string",\n'
        '      "doc_id": "string",\n'
        '      "source_filename": "string",\n'
        '      "page": 0,\n'
        '      "chunk_index": 0,\n'
        '      "section": "string",\n'
        '      "excerpt": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Citation rules:\n"
        "Use only citations from the provided source list.\n"
        "Do not invent citations, page numbers, sections, or excerpts.\n"
        "Only cite sources that directly support the answer.\n"
        "For each citation, copy the excerpt verbatim or near-verbatim from the source manifest.\n\n"
        "Answering rules:\n"
        "Answer the user's question directly in the opening sentence.\n"
        "Prefer exact values and timing when the context provides them.\n"
        "For definition or plan questions, include all directly relevant operational details present in context: start month, percentage, required account/location, purpose, and example monetary amount.\n"
        "Do not omit a concrete example amount when the retrieved context provides one for the same plan and same concept.\n"
        "Combine supporting points from multiple retrieved chunks when helpful.\n"
        "If the evidence is partial or ambiguous, say that explicitly.\n"
        "Set confidence between 0.0 and 1.0 based only on the context.\n"
    )
    if mode == "strict":
        return (
            base
            + "\nStrict grounding mode:\n"
            + "Do not use outside knowledge, training memory, assumptions, or unstated facts.\n"
            + "If the answer is not explicitly supported by the context, return "
            + '"The answer is not contained in the provided context.", confidence 0.0, and an empty citations array.\n'
            + "Last line of defense: if it is not in the provided context, you must not say it."
        )
    return base


def build_user_prompt(context_payload: dict) -> str:
    sources = [
        {
            "chunk_id": source["chunk_id"],
            "doc_id": source["doc_id"],
            "source_filename": source["source_filename"],
            "page": source["page"],
            "chunk_index": source.get("chunk_index"),
            "section": source.get("section"),
            "excerpt": source.get("excerpt"),
        }
        for source in context_payload.get("sources", [])
    ]
    return json.dumps(
        {
            "query": context_payload["query"],
            "context": context_payload["context"],
            "sources": sources,
        },
        indent=2,
        ensure_ascii=False,
    )


def build_prompt_package(context_payload: dict, mode: str = GENERATION_MODE) -> dict:
    return {
        "query": context_payload["query"],
        "system_prompt": build_system_prompt(mode=mode),
        "user_prompt": build_user_prompt(context_payload),
        "generation_mode": mode,
        "source_count": len(context_payload.get("sources", [])),
        "selected_chunk_count": context_payload.get("selected_chunk_count", 0),
        "estimated_tokens": context_payload.get("estimated_tokens", 0),
        "sources": context_payload.get("sources", []),
    }
