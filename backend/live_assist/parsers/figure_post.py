"""Post-process Docling layout output.

Three cleanups:
  1. `strip_marginal_figures` — retag logo/banner false-positives as `page_marginal`.
  2. `attach_orphan_captions` — link orphan caption blocks to their nearest
     figure or table on the same page (handles journal templates where Docling's
     built-in caption association fails, e.g. Frontiers `FIGURE N | ...` format).
  3. `repair_equation_text` — re-extract equation text from the PDF's own text
     layer via fitz at the equation's bbox. Fixes Docling's font-decoding glitches
     (paper2: `=`→`D`, `+`→`C`) and gives clean ASCII without the 30s/equation
     cost of the CodeFormulaV2 VLM.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

# Tunable thresholds (in PDF points — 1 inch = 72 pt)
TOP_BOTTOM_BAND_PT = 40   # within this many points of page edge = marginal zone
MAX_STRIP_HEIGHT_PT = 25  # marginal figures are short horizontal strips
DEFAULT_PAGE_H = 792.0    # US Letter; override per-doc if known


def _is_marginal_strip(bbox: list[float], page_h: float) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    h = abs(y1 - y0)
    top    = max(y0, y1) > page_h - TOP_BOTTOM_BAND_PT
    bottom = min(y0, y1) < TOP_BOTTOM_BAND_PT
    return (top or bottom) and h < MAX_STRIP_HEIGHT_PT


# Classifier-based logo/icon detection — fires regardless of page position,
# so a corporate logo in the middle of a page gets stripped too.
_LOGO_ICON_CLASSES = {"logo", "icon"}


def _is_logo_or_icon_class(block: dict[str, Any]) -> bool:
    """True if Docling's picture classifier tagged this block as logo/icon."""
    anns = (block.get("metadata") or {}).get("annotations") or []
    for a in anns:
        if a.get("kind") == "classification" and a.get("class") in _LOGO_ICON_CLASSES:
            return True
    return False


def strip_marginal_figures(parsed: dict[str, Any],
                           page_heights: dict[int, float] | None = None
                           ) -> dict[str, Any]:
    """Return a new parsed dict where logo/banner false-positives are retyped
    as `page_marginal` (kept in blocks, removed from figure_count).

    Args:
      parsed: the loaded parsed.json dict (output of docling_parser).
      page_heights: optional {page_no: height_pt}. Falls back to 792pt (US Letter).
    """
    out = deepcopy(parsed)
    n_retyped = 0

    n_retyped_by_position = 0
    n_retyped_by_class = 0
    for b in out.get("blocks", []):
        if b.get("type") != "figure":
            continue
        pg = b.get("page")
        ph = (page_heights or {}).get(pg, DEFAULT_PAGE_H)
        reason = None
        if _is_marginal_strip(b.get("bbox") or [], ph):
            reason = "marginal_strip"
            n_retyped_by_position += 1
        elif _is_logo_or_icon_class(b):
            reason = "classifier_logo_or_icon"
            n_retyped_by_class += 1
        if reason:
            b["type"] = "page_marginal"
            b.setdefault("metadata", {})["reclassified_from"] = "figure"
            b["metadata"]["reclassified_reason"] = reason
            n_retyped += 1

    # Update asset counts
    assets = out.setdefault("assets", {})
    assets["figure_count"] = sum(
        1 for b in out["blocks"] if b.get("type") == "figure"
    )
    assets["page_marginal_count"] = sum(
        1 for b in out["blocks"] if b.get("type") == "page_marginal"
    )
    assets["marginal_figures_stripped"] = n_retyped
    assets["marginal_by_position"] = n_retyped_by_position
    assets["marginal_by_classifier"] = n_retyped_by_class

    return out


# =====================================================================
# Caption attachment
# =====================================================================

# Broad caption pattern — research papers use many words for "figure" and "table".
# Number may have a letter suffix (e.g., "Figure 3a", "Tab. 2b") which we capture
# but ignore for matching purposes (suffix variants share the same parent asset).
_FIGURE_WORDS = (
    "FIGURE", "Figure", "FIG", "Fig", "ILLUSTRATION", "Illustration",
    "SCHEME", "Scheme", "DIAGRAM", "Diagram", "CHART", "Chart",
    "PLOT", "Plot", "IMAGE", "Image", "PICTURE", "Picture", "GRAPH", "Graph",
)
_TABLE_WORDS = ("TABLE", "Table", "TAB", "Tab", "TBL", "Tbl")
_EQUATION_WORDS = ("EQUATION", "Equation", "EQ", "Eq", "FORMULA", "Formula")
_ALL_CAPTION_WORDS = _FIGURE_WORDS + _TABLE_WORDS + _EQUATION_WORDS

# Match: <keyword> [optional period] <whitespace> <number> [optional letter suffix]
# Followed by a punctuation/space/end-of-string (so "Figure 5b:" matches but
# "Figureheadband" doesn't).
_CAPTION_RX = re.compile(
    r"^\s*(?P<kind>" + "|".join(re.escape(w) for w in _ALL_CAPTION_WORDS) + r")"
    r"\.?\s*(?P<num>\d+)(?P<suffix>[a-zA-Z])?(?=[\s:.,|\-—–)\]\}]|$)",
    re.IGNORECASE,
)


def _parse_caption_anchor(text: str) -> tuple[str, int] | None:
    """Return ('figure'|'table', number) tuple when text starts with a caption keyword.

    Recognises a wide set of synonyms commonly used in scientific papers:
      Figures:  Figure, Fig, Fig., Illustration, Scheme, Diagram, Chart, Plot,
                Image, Picture, Graph
      Tables:   Table, Tab, Tab., Tbl
    Optional letter suffix on the number is accepted (e.g. 'Figure 3a').
    """
    if not text:
        return None
    m = _CAPTION_RX.match(text.strip())
    if not m:
        return None
    kw = m.group("kind").lower()
    if kw.startswith("tab") or kw.startswith("tbl"):
        target = "table"
    elif kw.startswith("eq") or kw.startswith("formula"):
        target = "equation"
    else:
        target = "figure"
    return (target, int(m.group("num")))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    if not bbox or len(bbox) != 4:
        return (0.0, 0.0)
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between bbox centers (ignores page; assume same page)."""
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def attach_orphan_captions(parsed: dict[str, Any]) -> dict[str, Any]:
    """Link orphan `caption` blocks to their nearest figure/table on the same page.

    Strategy:
      - If the caption text matches "Figure N" / "Table N", pick the asset with
        matching number on the same page (if exactly one).
      - Otherwise, attach to the nearest asset on the same page by bbox distance.
      - Once attached, mark the caption block with `attached_to_block_id`.
    """
    out = deepcopy(parsed)
    blocks = out["blocks"]

    # Index assets by page and type (figures, tables, equations are all
    # caption-eligible targets)
    figures_by_page: dict[int, list[dict[str, Any]]] = {}
    tables_by_page:  dict[int, list[dict[str, Any]]] = {}
    equations_by_page: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        if b.get("type") == "figure":
            figures_by_page.setdefault(b.get("page"), []).append(b)
        elif b.get("type") == "table":
            tables_by_page.setdefault(b.get("page"), []).append(b)
        elif b.get("type") == "equation":
            equations_by_page.setdefault(b.get("page"), []).append(b)

    captions = [b for b in blocks if b.get("type") == "caption"]
    n_attached_fig = 0
    n_attached_tab = 0
    n_attached_eq  = 0
    n_orphaned = 0

    _POOL_BY_KIND = {
        "figure": figures_by_page,
        "table": tables_by_page,
        "equation": equations_by_page,
    }

    for cap in captions:
        pg = cap.get("page")
        text = (cap.get("text") or "").strip()
        anchor = _parse_caption_anchor(text)

        target_pool: list[dict[str, Any]] = []
        if anchor is None:
            # Fallback — try figures, then tables, then equations
            target_pool = (figures_by_page.get(pg, [])
                           + tables_by_page.get(pg, [])
                           + equations_by_page.get(pg, []))
        else:
            target_kind, _num = anchor
            target_pool = _POOL_BY_KIND[target_kind].get(pg, [])

        if not target_pool:
            n_orphaned += 1
            continue

        # Find nearest by bbox distance
        cap_bbox = cap.get("bbox") or []
        target = min(target_pool, key=lambda b: _bbox_distance(cap_bbox, b.get("bbox") or []))

        meta = target.setdefault("metadata", {})
        meta["caption"] = text
        meta["caption_block_id"] = cap.get("block_id")
        meta["caption_anchor"] = (
            f"{anchor[0]}-{anchor[1]}" if anchor else "by-proximity"
        )

        cap["attached_to_block_id"] = target.get("block_id")
        cap["attached_to_type"] = target.get("type")

        t = target.get("type")
        if t == "figure":
            n_attached_fig += 1
        elif t == "table":
            n_attached_tab += 1
        elif t == "equation":
            n_attached_eq += 1

    assets = out.setdefault("assets", {})
    assets["captions_total"] = len(captions)
    assets["captions_attached_to_figures"] = n_attached_fig
    assets["captions_attached_to_tables"] = n_attached_tab
    assets["captions_attached_to_equations"] = n_attached_eq
    assets["captions_orphaned"] = n_orphaned

    return out


# =====================================================================
# Figure description via Ollama VLM (run for ALL images)
# =====================================================================

# Only skip pure decoration — everything else gets a VLM description so the
# LLM at retrieval/generation time can use the visual content.
_DESCRIPTION_SKIP_CLASSES = {
    "logo", "icon", "page_thumbnail", "stamp", "bar_code", "qr_code",
}


def _figure_class(figure_block: dict[str, Any]) -> str | None:
    """Return the top-confidence classification class for a figure block."""
    anns = (figure_block.get("metadata") or {}).get("annotations") or []
    for a in anns:
        if a.get("kind") == "classification":
            return a.get("class")
    return None


# Granite chart extraction was removed (HF download was anonymous-rate-limited).
# bar_chart / line_chart / pie_chart now fall through to the class-aware Ollama
# VLM path (extract_figure_description_with_vlm) which has dedicated prompts
# per chart type — bar_chart asks for "comparison + key finding + trend",
# line_chart for "trend / peaks / inflections", pie_chart for "composition +
# dominant slice". See table_extract.py::_FIGURE_PROMPT_BY_CLASS.

# Figure-classifier label for an image that is actually a rendered table.
# These get routed to the table extractor (1200-tok VLM) instead of the
# 1000-tok figure describer, and the block is retyped to `type=table`.
_TABLE_AS_IMAGE_CLASSES = {"table", "table_as_image"}


def describe_figures_with_vlm(parsed: dict[str, Any],
                              project_root: str | None = None,
                              min_pixel_dim: int = 80) -> dict[str, Any]:
    """Class-routed picture-block dispatcher (T-05, VLM-only variant).

    Each `type=figure` block lands in exactly one downstream extractor based
    on its Docling picture-classifier label:

      (a) `table` / `table_as_image`
            → hand off to `extract_table_with_vlm` (1200-tok cap, temp 0),
              retype block to `type=table`. Result lives in `metadata.markdown`.
              extraction_path = "vlm_table_1200".

      (b) skip-classes {logo, icon, page_thumbnail, stamp, bar_code, qr_code,
          signature}
            → no extraction at all. extraction_path = "skipped".

      (c) Everything else — including bar_chart / line_chart / pie_chart /
          flow_chart / engineering_drawing / photograph / scatter_plot /
          box_plot / geographical_map / other / default
            → Tesseract OCR for embedded text + Ollama VLM with the
              class-aware interpretive prompt (1000-tok cap).
              extraction_path = "vlm_figure_1000".

    Note: Granite Vision chart extraction was removed (HF download was
    anonymous-rate-limited and we want one consistent VLM path).

    Caption attachment is handled upstream by `attach_orphan_captions` and is
    NOT touched here.
    """
    out = deepcopy(parsed)
    # NOTE: iterate over a list snapshot so we can retype blocks in-place.
    figures = [b for b in out["blocks"] if b.get("type") == "figure"]

    try:
        from .table_extract import (
            extract_figure_description_with_vlm, extract_figure_embedded_text,
            extract_table_with_vlm, OLLAMA_VLM_MODEL,
        )
    except ImportError:
        from table_extract import (
            extract_figure_description_with_vlm, extract_figure_embedded_text,
            extract_table_with_vlm, OLLAMA_VLM_MODEL,
        )

    from pathlib import Path as _P
    root = _P(project_root) if project_root else _P.cwd()

    # Counters
    n_described = 0       # vlm_figure_1000 success
    n_ocr_only = 0
    n_vlm_failed = 0
    n_skipped_class = 0
    n_skipped_small = 0
    n_no_image = 0
    n_table_rerouted = 0  # picture classified as table → table extractor

    for f in figures:
        md = f.setdefault("metadata", {})
        # Initialise schema fields so they always exist
        md.setdefault("embedded_text", None)
        md.setdefault("vlm_description", None)
        md.setdefault("description_model", None)
        md.setdefault("extraction_path", None)
        # caption was already attached by attach_orphan_captions if present
        md.setdefault("caption", md.get("caption"))

        img_rel = md.get("image_path")
        if not img_rel:
            n_no_image += 1
            md["description_skip_reason"] = "no_image_path"
            md["extraction_path"] = "skipped"
            continue
        img_path = root / img_rel
        if not img_path.exists():
            n_no_image += 1
            md["description_skip_reason"] = "image_file_missing"
            md["extraction_path"] = "skipped"
            continue

        klass = _figure_class(f)

        # --- (b) skip pure decoration ----------------------------------
        if klass in _DESCRIPTION_SKIP_CLASSES:
            n_skipped_class += 1
            md["description_skip_reason"] = f"class:{klass}"
            md["extraction_path"] = "skipped"
            continue

        # Image dimension guard (used by all remaining paths)
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            n_no_image += 1
            md["description_skip_reason"] = "image_open_failed"
            md["extraction_path"] = "skipped"
            continue
        if w < min_pixel_dim or h < min_pixel_dim:
            n_skipped_small += 1
            md["description_skip_reason"] = f"too_small:{w}x{h}"
            md["extraction_path"] = "skipped"
            continue

        # --- (a) picture classified as a table → table extractor --------
        if klass in _TABLE_AS_IMAGE_CLASSES:
            markdown = extract_table_with_vlm(str(img_path))
            if markdown:
                md["markdown"] = markdown
                md["extraction_path"] = "vlm_table_1200"
                md["description_model"] = OLLAMA_VLM_MODEL
                md["reclassified_from"] = "figure"
                f["type"] = "table"          # retype so chunker treats as table
                n_table_rerouted += 1
            else:
                # VLM failed — fall through to description so block isn't empty
                md["chart_fallback_reason"] = "table_vlm_failed"
                md["extraction_path"] = "vlm_table_1200_failed"
            if f["type"] == "table":
                continue   # done — block has become a table

        # --- (d) generic figure: OCR + class-aware VLM description -----
        ocr_text = extract_figure_embedded_text(str(img_path))
        md["embedded_text"] = ocr_text

        # Pass the attached caption into the VLM prompt as ground-truth
        # context (e.g. "FIGURE 4. CNN architecture used in this study.")
        cap = md.get("caption")
        desc = extract_figure_description_with_vlm(
            str(img_path), figure_class=klass, caption=cap,
        )
        if desc:
            md["vlm_description"] = desc
            md["description_model"] = OLLAMA_VLM_MODEL
            md["description_prompt_class"] = klass or "default"
            md["caption_used_in_prompt"] = bool(cap and cap.strip())
            md["extraction_path"] = "vlm_figure_1000"
            n_described += 1
        else:
            n_vlm_failed += 1
            md["extraction_path"] = "vlm_figure_1000_failed"
            if ocr_text:
                n_ocr_only += 1

    assets = out.setdefault("assets", {})
    assets["figures_described_by_vlm"] = n_described
    assets["figures_ocr_only"] = n_ocr_only
    assets["figures_vlm_failed"] = n_vlm_failed
    assets["figures_skipped_by_class"] = n_skipped_class
    assets["figures_skipped_too_small"] = n_skipped_small
    assets["figures_no_image"] = n_no_image
    assets["figures_rerouted_to_table_vlm"] = n_table_rerouted

    return out


# =====================================================================
# Equation text repair via fitz (PDF text-layer re-extraction at bbox)
# =====================================================================

_FONT_GLYPH_NAMES = ("/Sigma", "/Pi", "/Delta", "/Alpha", "/Beta", "/Gamma",
                     "/sigma", "/pi", "/delta", "/Omega", "/Theta", "/Lambda")
_GREEK_AND_MATH = ("σ", "π", "Σ", "Π", "∅", "∝", "∥", "∣", "²", "³",
                   "Δ", "θ", "λ", "Ω", "∫", "∑", "∞", "≈", "≤", "≥", "≠",
                   "α", "β", "γ", "μ", "ν", "τ", "ω", "ρ", "φ", "ψ", "χ",
                   "→", "←", "↔", "⊆", "⊂", "∈", "∉", "·")


def _has_font_glyph_name(text: str) -> bool:
    """True if text contains a literal font glyph name like /Sigma1 — these
    are signs Docling failed to map a font to a Unicode character."""
    return "/" in text and any(g in text for g in _FONT_GLYPH_NAMES)


def _has_replacement_chars(text: str) -> bool:
    """True if text contains the Unicode replacement char (font/encoding failure)."""
    return "�" in text  # the � character


def _has_basic_operators(text: str) -> bool:
    return any(op in text for op in ("=", "+", "-", "/", "*", "<", ">"))


def _has_math_symbols(text: str) -> bool:
    return any(g in text for g in _GREEK_AND_MATH)


def _quality_score(text: str) -> int:
    """Higher = better equation text quality. Used to compare Docling vs fitz."""
    if not text or len(text) < 3:
        return -100
    score = 0
    if _has_font_glyph_name(text): score -= 50    # /Sigma1 → really bad
    if _has_replacement_chars(text): score -= 30  # � → bad
    if _has_basic_operators(text): score += 10    # has =, + → good
    if _has_math_symbols(text): score += 5         # has σ, ∅ → good
    # Penalty for being too long — likely body text bleed
    if len(text) > 200: score -= 10
    return score


def repair_equation_text(parsed: dict[str, Any], pdf_path: str) -> dict[str, Any]:
    """For each equation block, re-extract its text from the PDF text layer
    using fitz. Replaces Docling's font-decoded version with PDF-native text
    when the new version looks better.

    Args:
      parsed: loaded parsed.json dict
      pdf_path: path to the original PDF file (required for fitz to open)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # fitz not installed — skip silently and leave Docling text alone
        return parsed

    out = deepcopy(parsed)
    eqs = [b for b in out["blocks"] if b.get("type") == "equation"]
    if not eqs:
        return out

    n_repaired = 0
    n_better = 0
    n_kept = 0
    pdf = fitz.open(pdf_path)
    try:
        for eq in eqs:
            pg = eq.get("page")
            bb = eq.get("bbox")
            if not pg or not bb or len(bb) != 4:
                n_kept += 1
                continue
            page = pdf[pg - 1]
            ph = page.rect.height
            # Convert PDF coords (y-up from bottom) → fitz coords (y-down from top)
            pdf_top = max(bb[1], bb[3])
            pdf_bottom = min(bb[1], bb[3])
            fitz_top = ph - pdf_top
            fitz_bottom = ph - pdf_bottom
            # Use exact bbox — padding causes body text bleed on tight layouts
            rect = fitz.Rect(
                min(bb[0], bb[2]),
                fitz_top,
                max(bb[0], bb[2]),
                fitz_bottom,
            )
            raw = page.get_textbox(rect).strip()
            raw_clean = " ".join(raw.split())
            orig_text = eq.get("text") or ""
            eq.setdefault("metadata", {})["docling_text"] = orig_text

            if not raw_clean:
                eq["metadata"]["text_source"] = "docling"
                n_kept += 1
                continue

            # Compare quality scores — only swap if fitz is meaningfully better
            score_docling = _quality_score(orig_text)
            score_fitz    = _quality_score(raw_clean)
            if score_fitz > score_docling + 5:  # require clear win, not a tie
                eq["text"] = raw_clean
                eq["metadata"]["text_source"] = "fitz_pdf_layer"
                eq["metadata"]["quality_scores"] = {
                    "docling": score_docling, "fitz": score_fitz,
                }
                # Also overwrite the corrupt `latex` field — Docling's
                # CodeFormulaV2 sometimes decodes `=`→`D`, `+`→`C` etc., which
                # leaves the latex unusable. The clean fitz text is the best
                # LLM-readable representation we have without re-running a
                # math VLM, so we copy it across so downstream sees ONE
                # consistent symbol set in both fields.
                orig_latex = eq.get("latex") or ""
                if orig_latex and (
                    " D " in f" {orig_latex} " or " C " in f" {orig_latex} "
                ):
                    eq["metadata"]["docling_latex"] = orig_latex
                    eq["latex"] = raw_clean
                    eq["metadata"]["latex_source"] = "fitz_pdf_layer_copy"
                else:
                    eq["metadata"]["latex_source"] = "docling"
                n_repaired += 1
                n_better += 1
            else:
                eq["metadata"]["text_source"] = "docling"
                eq["metadata"]["quality_scores"] = {
                    "docling": score_docling, "fitz": score_fitz,
                }
                eq["metadata"]["latex_source"] = "docling"
                n_kept += 1

            # Extract the equation label (e.g. "(1)") from the text — usually
            # appears at the end of the equation. Useful for downstream
            # citations like "see Eq. (3)".
            import re as _re
            m = _re.search(r"\((\d+[a-z]?)\)\s*$", (eq.get("text") or "").strip())
            eq["metadata"]["equation_label"] = m.group(1) if m else None
    finally:
        pdf.close()

    # ---- Attach surrounding-paragraph context for each equation -----
    # Equations rarely have their own caption block in research PDFs; their
    # meaning is in the surrounding prose ("Precision is defined as..."). We
    # store the previous + next paragraph on the same page so the chunker /
    # retriever can keep the equation paired with its explanation.
    page_blocks: dict[int, list[dict[str, Any]]] = {}
    for b in out["blocks"]:
        page_blocks.setdefault(b.get("page"), []).append(b)
    for eq in eqs:
        sib = page_blocks.get(eq.get("page"), [])
        try:
            i = sib.index(eq)
        except ValueError:
            continue
        prev_txt, next_txt = None, None
        for j in range(i - 1, -1, -1):
            if sib[j].get("type") in ("paragraph", "list_item"):
                prev_txt = (sib[j].get("text") or "")[:400].strip() or None
                break
        for j in range(i + 1, len(sib)):
            if sib[j].get("type") in ("paragraph", "list_item"):
                next_txt = (sib[j].get("text") or "")[:400].strip() or None
                break
        eq.setdefault("metadata", {})
        eq["metadata"]["surrounding_text"] = {
            "before": prev_txt, "after": next_txt,
        }

    assets = out.setdefault("assets", {})
    assets["equations_repaired_by_fitz"] = n_repaired
    assets["equations_kept_docling_text"] = n_kept
    assets["equations_with_label"] = sum(
        1 for e in eqs if (e.get("metadata") or {}).get("equation_label")
    )

    return out


# =====================================================================
# Image-table OCR fallback
# =====================================================================

def _ocr_rows_from_image(img_path: str) -> tuple[list[list[str]], str]:
    """OCR an image (Tesseract psm=6) and return (rows, raw_text).

    Splits each output line on tab or 2+ whitespace runs to infer cells.
    Pads short rows to max width.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return [], ""
    try:
        img = Image.open(img_path)
    except Exception:
        return [], ""
    # Upscale if small — Tesseract works better at >150 DPI equivalent
    if img.size[0] < 1000:
        scale = max(1, 1500 // img.size[0])
        img = img.resize((img.size[0] * scale, img.size[1] * scale),
                         Image.LANCZOS)
    try:
        raw = pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return [], ""
    rows: list[list[str]] = []
    for line in raw.split("\n"):
        line = line.rstrip()
        if not line.strip():
            continue
        # Split on tabs OR runs of 2+ spaces (typical column separator)
        cells = re.split(r"\t|\s{2,}", line)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            rows.append(cells)
    # Pad short rows to max width
    if rows:
        maxw = max(len(r) for r in rows)
        rows = [r + [""] * (maxw - len(r)) for r in rows]
    return rows, raw


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    n_cols = len(rows[0])
    out_lines = []
    # Header
    out_lines.append("| " + " | ".join(rows[0]) + " |")
    out_lines.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in rows[1:]:
        out_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(out_lines)


def repair_image_tables_with_ocr(parsed: dict[str, Any],
                                 project_root: str | None = None,
                                 min_rows_required: int = 2) -> dict[str, Any]:
    """2-tier table extraction:
      Tier 1: Docling TableFormer (text-PDF tables; already ran before this step)
      Tier 2: VLM via Ollama (image-rendered tables — returns markdown directly)

    The VLM call replaces the old TATR + per-cell-Tesseract cascade. Tesseract
    on borderless image tables was unreliable; a vision model handles them
    end-to-end and produces clean markdown without any hand-stitching.

    Also emits a uniform `llm_markdown` field on every table block (Tier 1 + 2)
    so the chunker reads one field regardless of which tier produced it.
    """
    out = deepcopy(parsed)
    tables = [b for b in out["blocks"] if b.get("type") == "table"]

    try:
        from .table_extract import extract_table_with_vlm, _count_rows_cols, OLLAMA_VLM_MODEL
        from .llm_markdown import rows_to_llm_markdown
    except ImportError:
        from table_extract import extract_table_with_vlm, _count_rows_cols, OLLAMA_VLM_MODEL
        from llm_markdown import rows_to_llm_markdown

    from pathlib import Path as _P
    root = _P(project_root) if project_root else _P.cwd()

    n_tier1_ok = 0
    n_tier2_ok = 0
    n_failed = 0
    n_no_image = 0

    for t in tables:
        md = t.setdefault("metadata", {})
        n_rows = md.get("num_rows", 0) or 0
        already_extracted = (n_rows > 0 and md.get("markdown", "").strip())

        if already_extracted:
            # Tier 1 (Docling TableFormer) handled it — text-PDF table with grid
            md["table_text_source"] = md.get("table_text_source", "docling_tableformer")
            md["extraction_tier"] = 1
            md["structure_quality"] = "high"
            n_tier1_ok += 1
            # llm_markdown built below from rows + caption
            md["llm_markdown"] = rows_to_llm_markdown(
                md.get("rows") or [],
                caption=md.get("caption"),
                page=t.get("page"),
                source_tier=md["table_text_source"],
                is_degraded=False,
            )
            continue

        # Tier 2 — VLM on the saved table image
        img_rel = md.get("image_path")
        img_path = (root / img_rel) if img_rel else None
        if not img_rel or not img_path.exists():
            n_no_image += 1
            md["table_text_source"] = "none"
            md["extraction_tier"] = 0
            md["structure_quality"] = "missing"
            md["llm_markdown"] = rows_to_llm_markdown(
                [], caption=md.get("caption"), page=t.get("page"),
                source_tier="none", is_degraded=True,
            )
            continue

        vlm_markdown = extract_table_with_vlm(str(img_path))
        if not vlm_markdown:
            n_failed += 1
            md["table_text_source"] = "vlm_failed"
            md["extraction_tier"] = 2
            md["structure_quality"] = "missing"
            md["llm_markdown"] = rows_to_llm_markdown(
                [], caption=md.get("caption"), page=t.get("page"),
                source_tier="vlm_failed", is_degraded=True,
            )
            continue

        # VLM returned markdown directly — store it verbatim
        n_data_rows, n_cols = _count_rows_cols(vlm_markdown)
        md["markdown"] = vlm_markdown
        md["num_rows"] = n_data_rows
        md["num_cols"] = n_cols
        md["table_text_source"] = f"vlm:{OLLAMA_VLM_MODEL}"
        md["extraction_tier"] = 2
        md["structure_quality"] = "vlm"
        # rows list is left empty for VLM-extracted tables — the markdown IS
        # the authoritative representation (no row/col cells to enumerate)
        md["rows"] = []

        # Build the LLM-ready markdown: caption + VLM markdown + provenance.
        # We don't go through rows_to_llm_markdown because VLM already returned
        # a complete table; we just wrap it with header and footer.
        parts = []
        if md.get("caption"):
            parts.append(f"**{md['caption'].strip()}**")
            parts.append("")
        parts.append(vlm_markdown)
        parts.append("")
        parts.append(f"_[Source: page {t.get('page')}, parser: {md['table_text_source']}]_")
        md["llm_markdown"] = "\n".join(parts)
        n_tier2_ok += 1

    assets = out.setdefault("assets", {})
    assets["tables_tier1_ok"] = n_tier1_ok
    assets["tables_tier2_ok"] = n_tier2_ok
    assets["tables_failed"] = n_failed
    assets["tables_no_image"] = n_no_image
    # Legacy keys (other code might read these)
    assets["tables_ocr_success"] = n_tier2_ok
    assets["tables_already_extracted"] = n_tier1_ok
    return out
