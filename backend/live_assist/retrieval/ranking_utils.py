"""Reference-section detection + scoring penalty.

Port of RAG_project_old/Retrieval_Layer/ranking_utils.py — verbatim logic,
unchanged. The point: chunks from a paper's "References" section often match
queries by author names / years / journal words but contain no actual content
the user asked about. We detect them heuristically and multiply their
relevance score by `REFERENCE_PENALTY` so they sink in the ranking.
"""
from __future__ import annotations

import re


REFERENCE_SECTION_PATTERN = re.compile(
    r"\b(references?|bibliograph(?:y|ies)|works cited|citations?)\b",
    re.IGNORECASE,
)
AUTHOR_YEAR_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:,\s*[A-Z]\.)+\s*\(\d{4}\)")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
DOI_PATTERN = re.compile(r"\bdoi:\s*10\.\S+", re.IGNORECASE)
JOURNAL_PATTERN = re.compile(
    r"\b(neurology|radiology|neuroimage|brain|plos one|frontiers|ann\.?|j\.|vol\.?|pp\.?)\b",
    re.IGNORECASE,
)

REFERENCE_PENALTY = 0.45


def is_reference_like(section: str | None, content: str | None) -> bool:
    section_text = section or ""
    content_text = content or ""
    combined = f"{section_text}\n{content_text}"
    if REFERENCE_SECTION_PATTERN.search(section_text):
        return True
    score = 0
    if len(YEAR_PATTERN.findall(combined)) >= 4:
        score += 1
    if len(DOI_PATTERN.findall(combined)) >= 1:
        score += 1
    if len(AUTHOR_YEAR_PATTERN.findall(combined)) >= 2:
        score += 1
    if len(JOURNAL_PATTERN.findall(combined)) >= 3:
        score += 1
    lines = [line.strip() for line in content_text.splitlines() if line.strip()]
    citation_like_lines = sum(
        1 for line in lines
        if YEAR_PATTERN.search(line) and (DOI_PATTERN.search(line) or "," in line)
    )
    if citation_like_lines >= 3:
        score += 1
    return score >= 2


def penalize_reference_score(score: float, is_reference: bool,
                             penalty: float = REFERENCE_PENALTY) -> float:
    if not is_reference:
        return score
    return score * penalty
