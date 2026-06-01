"""LLM-ready markdown formatter for tables.

Produces clean, self-contained markdown tables suitable for direct inclusion
in a generation LLM's context window. Every table gets:
  - Caption as a bold header
  - Header row with proper |---| separator
  - Consistent column count per row
  - Sanitized cells (escaped pipes, flattened newlines, collapsed whitespace)
  - Source provenance footer
  - Quality disclaimer when extraction was degraded
"""
from __future__ import annotations

import re
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def _sanitize_cell(text: str) -> str:
    """Make a cell value safe to drop into a markdown table."""
    if not text:
        return ""
    s = str(text)
    # Flatten newlines (keep them readable with <br>)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " <br> ")
    # Escape pipes that would break the table
    s = s.replace("|", "\\|")
    # Collapse whitespace runs
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _normalize_grid(rows: list[list[Any]]) -> list[list[str]]:
    """Coerce rows into a rectangular grid of strings.

    Drops fully-empty rows, pads short rows, truncates over-long rows.
    """
    if not rows:
        return []
    clean: list[list[str]] = []
    for r in rows:
        # Each row item may be a dict (Docling format: {'text': ..., 'is_header': ...})
        # or a string (Tesseract / TATR format)
        cells: list[str] = []
        for c in r:
            if isinstance(c, dict):
                cells.append(_sanitize_cell(c.get("text", "")))
            else:
                cells.append(_sanitize_cell(c))
        if any(cell for cell in cells):  # drop fully-empty rows
            clean.append(cells)
    if not clean:
        return []
    # Use the modal column count to decide width — handles a stray malformed row
    from collections import Counter
    widths = Counter(len(r) for r in clean)
    target_w = max(widths, key=widths.get)
    out: list[list[str]] = []
    for r in clean:
        if len(r) < target_w:
            r = r + [""] * (target_w - len(r))
        elif len(r) > target_w:
            # Merge overflow cells into the last column rather than dropping
            head = r[:target_w - 1]
            tail = " ".join(r[target_w - 1:])
            r = head + [tail]
        out.append(r)
    return out


def _detect_header_row(rows: list[list[Any]]) -> bool:
    """Heuristic — first row is a header if cells came tagged as `is_header`
    OR the first row's cells are notably shorter than the body cells."""
    if not rows:
        return False
    first = rows[0]
    # Docling tagging
    if any(isinstance(c, dict) and c.get("is_header") for c in first):
        return True
    # Heuristic: header cells are short, body cells longer
    if len(rows) >= 2:
        first_lens = [len(str(c.get("text", "") if isinstance(c, dict) else c)) for c in first]
        body_lens = [len(str(c.get("text", "") if isinstance(c, dict) else c))
                     for r in rows[1:] for c in r]
        if first_lens and body_lens:
            avg_first = sum(first_lens) / len(first_lens)
            avg_body = sum(body_lens) / len(body_lens)
            if avg_first <= 20 and avg_first < avg_body * 0.6:
                return True
    return False


def rows_to_llm_markdown(
    rows: list[list[Any]],
    caption: str | None = None,
    page: int | None = None,
    source_tier: str = "docling_tableformer",
    is_degraded: bool = False,
    table_number: str | None = None,
) -> str:
    """Render rows as LLM-ready markdown with caption + provenance.

    Args:
      rows: list of rows (each a list of cells; cells may be str or dict)
      caption: original table caption text (e.g. "TABLE 3 | Performance...")
      page: source PDF page number
      source_tier: which extractor produced this ("docling_tableformer", "tatr",
                   "tesseract_ocr", "pp_structure")
      is_degraded: True if structure is uncertain (Tier 3 fallback or 1-col output)
      table_number: optional explicit "Table 3" label if known
    """
    grid = _normalize_grid(rows)
    parts: list[str] = []

    # Caption header
    if caption:
        cap_clean = caption.strip()
        parts.append(f"**{cap_clean}**")
        parts.append("")

    if not grid:
        # No usable rows — emit a placeholder
        if caption:
            parts.append("_[Table content could not be extracted from this region. "
                         f"See source on page {page or '?'}.]_")
        else:
            parts.append("_[Empty table region.]_")
        return "\n".join(parts)

    has_header = _detect_header_row(rows) and len(grid) >= 2
    n_cols = len(grid[0])

    # Build markdown rows
    def fmt_row(r: list[str]) -> str:
        return "| " + " | ".join(r) + " |"

    if has_header:
        parts.append(fmt_row(grid[0]))
        parts.append("|" + "|".join(["---"] * n_cols) + "|")
        for r in grid[1:]:
            parts.append(fmt_row(r))
    else:
        # No clear header — emit a synthetic one so the table is valid markdown
        parts.append(fmt_row([f"col_{i+1}" for i in range(n_cols)]))
        parts.append("|" + "|".join(["---"] * n_cols) + "|")
        for r in grid:
            parts.append(fmt_row(r))

    # Provenance + quality footer
    parts.append("")
    provenance = f"_[Source: page {page}, parser: {source_tier}]_" if page \
                 else f"_[Source: unknown page, parser: {source_tier}]_"
    parts.append(provenance)

    if is_degraded:
        parts.append("_[Structure partially recovered — verify exact values against source PDF]_")
    elif n_cols == 1 and len(grid) > 1:
        # Single-column table is almost certainly a structure-loss signal
        parts.append("_[Only one column detected — original table may have multiple columns; "
                     "verify against source PDF]_")

    return "\n".join(parts)
