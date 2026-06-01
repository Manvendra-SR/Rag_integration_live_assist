"""Vision-based extraction helpers for tables and figures via Ollama VLM.

Table pipeline:
  Tier 1: Docling TableFormer (runs inside Docling, before this module)
  Tier 2: VLM via Ollama (qwen2.5-vl:7b default) — returns markdown directly

Figure pipeline:
  Tier 1: Docling DocumentPictureClassifier (already attached as class label)
  Tier 2a: Tesseract OCR for embedded text labels (cheap, ~50ms)
  Tier 2b: VLM description of figure content (slow, ~5-15s, rich semantic text)

The VLM never invents structure — it just describes / transcribes what's in
the image. Captions and Docling classification labels stay authoritative.
"""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(__name__)

# VLM access is delegated to clients.model_factory.get_vlm(). Provider /
# model / api-key are all chosen there from .env (PARSER_VLM_PROVIDER,
# PARSER_VLM_MODEL, GEMINI_API_KEY, etc.) with automatic cloud→local
# fallback. Legacy OLLAMA_VLM_MODEL / OLLAMA_HOST env vars are kept for
# backward compatibility — they are read by the factory's Ollama local
# builder if PARSER_OLLAMA_VISION_MODEL is unset.
OLLAMA_VLM_MODEL = (os.environ.get("PARSER_OLLAMA_VISION_MODEL")
                    or os.environ.get("OLLAMA_VLM_MODEL", "llama3.2-vision:11b"))
OLLAMA_HOST = (os.environ.get("PARSER_OLLAMA_API_URL", "").split("/api/")[0]
               or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
_t = os.environ.get("VLM_TIMEOUT_S", "").strip()
VLM_TIMEOUT_S: int | None = int(_t) if _t.isdigit() and int(_t) > 0 else None

PROMPT_TEMPLATE = """Convert this table image to a clean markdown table.

Rules:
- Use proper markdown table syntax with | separators.
- First row is the header row.
- Add a separator row |---|---|---| immediately after the header.
- Preserve all numeric values, units, symbols, and special characters EXACTLY as shown — DO NOT change, round, reorder, paraphrase, or invent values.
- If a cell spans multiple lines, join with a single space.
- If a cell is empty, leave it blank between pipes.
- Output the table ONCE. Do NOT repeat rows. Do NOT append extra empty separator rows. Do NOT echo the same phrase or row at the end.
- Stop generating as soon as the last real row of the table is written.
- Do NOT add commentary, explanation, or summary — output ONLY the markdown table.
- Do NOT wrap the output in ```markdown or any code fences.

Output the markdown table now:"""


def _extract_markdown_table(text: str) -> str | None:
    """Pull out just the markdown table from LLM output (strip preamble/code fences)."""
    if not text:
        return None
    # Strip common code-fence wrappers
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line and any closing fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Collect contiguous lines that look like table rows
    table_lines = []
    in_table = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table:
            # End of table when we hit a non-pipe line
            if not s or not (s.startswith("|") or "|" in s):
                break
        # else: still skipping pre-table commentary
    if not table_lines:
        return None
    # Must have at least 2 rows (header + 1 data) to count as a table
    if len(table_lines) < 2:
        return None
    return "\n".join(table_lines)


def estimate_table_token_budget(image_path: str,
                                min_budget: int = 800,
                                max_budget: int = 8000,
                                tokens_per_cell: int = 8) -> int:
    """Image-structure-based VLM budget. NO OCR used.

    Strategy: detect table rows + columns directly from the image's pixel
    density profile (no text recognition needed). Tables have a regular
    pattern of text bands separated by whitespace — this can be found by
    finding low-intensity stripes in the horizontal/vertical projections.

    Budget = (rows + 1) × cols × tokens_per_cell + cols × 4 + 100
             clamped to [min_budget, max_budget]

    Where tokens_per_cell averages around 8 (mix of short numeric cells like
    "0.95" ≈ 2 tokens and longer text cells like "Classification" ≈ 4 tokens
    plus markdown overhead per cell).

    Falls back to a safe default if image processing fails.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return 3000  # safe-but-bounded fallback

    try:
        img = Image.open(image_path).convert("L")  # grayscale
        arr = np.array(img)
        h, w = arr.shape

        # ---- detect rows: find horizontal text bands ----
        # Mean pixel intensity per row. Whitespace rows ≈ 255, text rows < 240.
        h_proj = arr.mean(axis=1)
        text_rows_mask = h_proj < 240  # True where there's any ink
        # Count contiguous groups of True (each group = one text row)
        edges = np.diff(text_rows_mask.astype(int))
        n_rows = int(np.sum(edges == 1))  # number of "rising edges" = group starts
        # If the first row already has text, that's another group not caught by edges
        if text_rows_mask[0]:
            n_rows += 1
        n_rows = max(2, n_rows)  # at least header + 1 data row

        # ---- detect columns: find vertical text bands ----
        v_proj = arr.mean(axis=0)
        text_cols_mask = v_proj < 240
        col_edges = np.diff(text_cols_mask.astype(int))
        n_cols = int(np.sum(col_edges == 1))
        if text_cols_mask[0]:
            n_cols += 1
        n_cols = max(2, n_cols)

        # ---- compute budget ----
        budget = (n_rows + 1) * n_cols * tokens_per_cell + n_cols * 4 + 100
        budget = max(min_budget, min(max_budget, budget))
        log.debug(f"  image-structure budget: {n_rows}r × {n_cols}c "
                  f"× {tokens_per_cell}tok = {budget} for {os.path.basename(image_path)}")
        return budget
    except Exception as exc:
        log.warning(f"image-structure budget failed for {image_path}: {exc}")
        return 3000


def _strip_trailing_separator_rows(md: str) -> str:
    """Drop trailing rows that are pure separators (|---|---|...) — VLM models
    sometimes spam these as a runaway tail after the real table ends.

    The legitimate header separator (always at line 2 of a markdown table) is
    preserved; only trailing/repeated separator rows are dropped.
    """
    if not md:
        return md
    sep_re = re.compile(r"^\s*\|(\s*[-:|]+\s*\|)+\s*$")
    lines = md.split("\n")
    if len(lines) < 3:
        return md
    # Walk backwards from the end, drop empty + separator lines.
    end = len(lines)
    while end > 2:  # always keep the first 2 lines (header + 1 separator)
        line = lines[end - 1].strip()
        if not line or sep_re.match(line):
            end -= 1
        else:
            break
    return "\n".join(lines[:end])


def _strip_repetition_tail(text: str, min_run: int = 3) -> str:
    """Trim degenerate repetition at the end of VLM output.

    Detects two patterns:
      (a) the same line repeated ≥ `min_run` times at the tail — keep one.
      (b) a short n-gram (2-6 words) repeated ≥ `min_run` times at the tail — cut.

    Conservative: only acts on the trailing region, never touches earlier text.
    """
    if not text:
        return text
    # ---- (a) repeated trailing lines ----
    lines = text.rstrip().split("\n")
    if len(lines) >= min_run:
        last = lines[-1].strip()
        if last:
            i = len(lines) - 1
            while i > 0 and lines[i].strip() == last:
                i -= 1
            run = (len(lines) - 1) - i
            if run >= min_run:
                lines = lines[: i + 2]  # keep one copy of the repeated line
                text = "\n".join(lines)
    # ---- (b) repeated trailing n-gram in last line / blob ----
    tail = text[-600:]
    words = tail.split()
    if len(words) >= min_run * 3:
        for n in range(6, 1, -1):
            if len(words) < n * min_run:
                continue
            ngram = words[-n:]
            # Count how many times this ngram repeats contiguously at the end
            reps = 1
            j = len(words) - n
            while j - n >= 0 and words[j - n:j] == ngram:
                reps += 1
                j -= n
            if reps >= min_run:
                # Cut from the start of the repeated block, keep one copy
                cut_words = words[: j + n]
                new_tail = " ".join(cut_words)
                text = text[: -len(tail)] + new_tail
                break
    return text.rstrip()


def _count_rows_cols(md_table: str) -> tuple[int, int]:
    """Count data rows and columns from a markdown table string."""
    if not md_table:
        return 0, 0
    lines = [l for l in md_table.split("\n") if l.strip().startswith("|")]
    # Filter out the separator row (|---|---|...)
    sep_re = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")
    data_lines = [l for l in lines if not sep_re.match(l)]
    if not data_lines:
        return 0, 0
    # Columns = number of pipe separators in the first row minus 1
    n_cols = max(1, data_lines[0].count("|") - 1)
    # Data rows = total minus header
    n_data_rows = max(0, len(data_lines) - 1)
    return n_data_rows + 1, n_cols  # include header row in row count


# Class-aware interpretive prompts. The goal of each is NOT "describe what is
# in the image" but "what information does this image convey to the reader" —
# the form that's useful inside a retrieved chunk fed to a generation LLM.
#
# Shared rules appended to every prompt (kept in one place so changes are uniform).
_SHARED_RULES = """
General rules:
- EXTRACT information from the image — do not merely describe what the image
  looks like. The reader cannot see the image; they need its contents.
- Read and quote any visible numeric values, percentages, axis labels, legend
  entries, and text labels verbatim. Do NOT round or paraphrase numbers.
- If a value is not legible, say so explicitly — do NOT invent it.
- Do not begin with "This figure" / "The figure" / "The image" — go straight
  to the content.
- Do not output markdown headers or bullet points — write plain prose.
- Write each idea ONCE. Do NOT repeat the same sentence, phrase, or word group.
- Stop after the last useful sentence. Do not pad with filler or restate prior sentences."""


_FIGURE_PROMPT_BY_CLASS: dict[str, str] = {
    # --- Process / decision diagrams -------------------------------------
    "flow_chart": """Extract the procedure shown in this flowchart so a reader can execute it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- The overall purpose of the process (one sentence).
- Every step in execution order, named by its visible box label (verbatim).
- At every decision point (diamond), state the condition being tested and the destination of each branch (yes/no, true/false, value).
- Any loops, parallel paths, retries, or terminal states (end / fail / success).""" + _SHARED_RULES,

    "engineering_drawing": """Extract the system design shown in this block / architecture diagram so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- Overall function the system performs.
- Every named component / module (use the visible labels verbatim) and its responsibility.
- Direction of information / signal flow between components (which arrow goes where).
- External inputs entering the system and outputs leaving it.""" + _SHARED_RULES,

    # --- Quantitative plots ----------------------------------------------
    "bar_chart": """Extract the quantitative comparison shown in this bar chart so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- X-axis categories (list all of them) and y-axis quantity with units.
- The numeric value of EACH bar (use the value labels printed on or above the bars if present; otherwise read from the y-axis to one decimal). Quote them verbatim, e.g. "ResNet-18: 81.4% / 83.51%".
- Which bar(s) are highest and lowest, and the magnitude of the gap.
- The takeaway pattern across categories (rising / falling / clustered / grouped-by-legend) and what every legend entry represents.""" + _SHARED_RULES,

    "line_chart": """Extract the trend shown in this line chart so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- X-axis variable + units and y-axis variable + units.
- For every visible line: its legend label, its start value, its end value, and its overall shape (monotonic up/down, plateau, oscillation, crossover).
- Specific notable points — peaks, troughs, inflections, intersections — with approximate (x, y) values.
- The takeaway: what trend is the chart demonstrating?""" + _SHARED_RULES,

    "pie_chart": """Extract the composition shown in this pie chart so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- What whole is being broken down.
- Every visible slice with its label and percentage (quote verbatim — e.g. "Class A: 42%"). If a slice's percentage isn't printed, estimate it to the nearest 5% and say "approximately".
- The dominant slice and any notably small slices.
- The takeaway the chart is making.""" + _SHARED_RULES,

    "scatter_plot": """Extract the relationship shown in this scatter plot so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- The two plotted variables with units.
- The shape of the relationship (positive / negative / no correlation; linear / non-linear; tight / spread).
- Any visible groups, clusters, outliers, or annotated regions and approximately where they sit.
- Every legend entry and what each color / marker represents.""" + _SHARED_RULES,

    "box_plot": """Extract the distribution comparison shown in this box plot so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- X-axis groups and y-axis measured quantity with units.
- Each group's median value (read from the box centre line) and approximate IQR width.
- Any visible outliers, skew, overlap, or notably different groups.
- The takeaway: which group has the highest / lowest median and which has the most variability?""" + _SHARED_RULES,

    # --- Photographic / radiological -------------------------------------
    "photograph": """Extract the visible content of this photograph so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- The subject (object, person, scene, equipment, sample, anatomical region).
- Any labels, callouts, panel letters (a, b, c…), scale bars, or annotations visible — quoted verbatim.
- The likely informational purpose in context (illustrating a setup, showing an outcome, documenting a sample, demonstrating a condition / pathology).""" + _SHARED_RULES,

    # --- Tables-as-images caught by the classifier ---------------------
    "table_as_image": """Extract the contents of this table-rendered-as-image so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- Column headers and row labels.
- The key cell values verbatim (especially numeric ones).
- The takeaway finding (best / worst / largest difference / total).
Note: a separate table-extraction pass may also produce a full markdown table.""" + _SHARED_RULES,
}


# Generic fallback for unknown / "other" / unclassified figures.
_FIGURE_PROMPT_DEFAULT = """Extract the visible content and informational message of this figure so a reader can reason about it without seeing the image.

Required content (5-7 sentences of prose, at least ~120 words for substantive coverage):
- What kind of figure it is (chart, diagram, photograph, schematic, etc.).
- The actual content it depicts and what point it is making.
- Every visible axis label, legend entry, annotated region, or text label — quoted verbatim.
- The takeaway: what would the reader learn from this figure?""" + _SHARED_RULES


# Normalise common class-name aliases coming out of Docling's classifier.
_FIG_CLASS_ALIAS = {
    "flowchart": "flow_chart",
    "flow-chart": "flow_chart",
    "barchart": "bar_chart",
    "linechart": "line_chart",
    "piechart": "pie_chart",
    "scatter": "scatter_plot",
    "boxplot": "box_plot",
    "diagram": "engineering_drawing",
    "block_diagram": "engineering_drawing",
    "architecture": "engineering_drawing",
    "schematic": "engineering_drawing",
    "photo": "photograph",
    "image": "photograph",
    "mri": "photograph",
    "x_ray": "photograph",
}


def _select_figure_prompt(figure_class: str | None) -> tuple[str, str]:
    """Return (prompt_text, resolved_class). Falls back to the generic prompt
    when the class is unknown or None."""
    if not figure_class:
        return _FIGURE_PROMPT_DEFAULT, "default"
    key = figure_class.strip().lower()
    key = _FIG_CLASS_ALIAS.get(key, key)
    prompt = _FIGURE_PROMPT_BY_CLASS.get(key)
    if prompt is None:
        return _FIGURE_PROMPT_DEFAULT, "default"
    return prompt, key


# Kept for backward compatibility — old callers that imported the constant
# directly still see a sensible default prompt. New callers should use
# `_select_figure_prompt(figure_class)`.
FIGURE_DESCRIPTION_PROMPT = _FIGURE_PROMPT_DEFAULT


def extract_figure_description_with_vlm(image_path: str,
                                        model: str | None = None,   # deprecated; model picked by clients.get_vlm()
                                        max_tokens: int = 1000,
                                        figure_class: str | None = None,
                                        caption: str | None = None) -> str | None:
    """Send a figure image to the VLM and get back an INTERPRETIVE, retrieval-
    ready description. The prompt is selected by `figure_class` (Docling's
    DocumentPictureClassifier output) so flowcharts get step-by-step
    instructions, bar/line charts get trend takeaways, photographs get
    purpose-of-image prose, and so on. NOT for tables (use
    extract_table_with_vlm for those).

    When `caption` is provided (e.g. "FIGURE 4. CNN architecture used in this
    study."), it is injected at the top of the prompt as document-level
    context so the VLM can align its description with the author's stated
    intent and use correct domain terminology.

    `max_tokens` caps Ollama's `num_predict` to prevent runaway generation.
    1000 fits a 3-5 sentence interpretive summary with headroom. Output is
    post-processed by `_strip_repetition_tail()` to drop tail-end repetition.
    """
    if not os.path.exists(image_path):
        return None
    prompt, resolved = _select_figure_prompt(figure_class)
    # Inject the source-document caption as document-level context. Placed
    # BEFORE the task instructions so the VLM reads context first and then
    # task. We instruct the model to USE it but not REPEAT it verbatim.
    if caption and caption.strip():
        cap_clean = caption.strip().replace("\n", " ")
        prompt = (
            f"Caption from source document: \"{cap_clean}\"\n"
            "Use this caption as ground-truth context (subject, scope, "
            "terminology). Do not repeat the caption verbatim in your output.\n\n"
            f"{prompt}"
        )
    try:
        from clients import get_vlm
        vlm = get_vlm()
    except Exception as exc:
        log.warning(f"VLM client unavailable: {exc}")
        return None
    log.info(f"  VLM figure call on {os.path.basename(image_path)} "
             f"(client={vlm.name}, num_predict={max_tokens}, "
             f"class={resolved}, caption={'yes' if caption else 'no'})")
    try:
        text = vlm.describe_image(image_path, prompt,
                                  options={"temperature": 0.1,
                                           "num_predict": max_tokens})
    except Exception as exc:
        log.warning(f"VLM description failed for {image_path}: {exc}")
        return None
    if not text:
        return None
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    text = _strip_repetition_tail(text)
    return text or None


def extract_figure_embedded_text(image_path: str) -> str | None:
    """OCR an image to capture any embedded text (flowchart box labels, chart
    axis labels, etc.). Cheap (~50ms) — runs in parallel with the slower VLM
    description. Returns None if Tesseract isn't installed or returns nothing."""
    if not os.path.exists(image_path):
        return None
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(image_path)
        # Upscale small images for better OCR
        if img.size[0] < 800:
            scale = max(1, 1200 // max(img.size[0], 1))
            img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.LANCZOS)
        raw = pytesseract.image_to_string(img, config="--psm 11")  # sparse text mode
    except Exception:
        return None
    # Normalize whitespace and dedupe lines
    lines = []
    seen = set()
    for line in raw.split("\n"):
        s = re.sub(r"\s+", " ", line).strip()
        if s and len(s) >= 2 and s not in seen:
            seen.add(s)
            lines.append(s)
    text = " | ".join(lines).strip()
    return text or None


def extract_table_with_vlm(image_path: str,
                           model: str | None = None,   # deprecated; model picked by clients.get_vlm()
                           max_tokens: int = 1200) -> str | None:
    """Send a table image to the VLM via Ollama; return markdown directly.

    Temperature is pinned to 0 (we want exact values, no paraphrasing).
    `max_tokens` defaults to 1200 — enough for a typical paper table while
    making runaway generation impossible. Output is then cleaned of:
      - trailing repeated separator rows (`|---|---|`)
      - tail-end repeated rows or n-grams (`_strip_repetition_tail`)

    Returns None on any failure.
    """
    if not os.path.exists(image_path):
        return None
    try:
        from clients import get_vlm
        vlm = get_vlm()
    except Exception as exc:
        log.warning(f"VLM client unavailable: {exc}")
        return None
    log.info(f"  VLM table call on {os.path.basename(image_path)} "
             f"(client={vlm.name}, temp=0, num_predict={max_tokens})")
    try:
        raw = vlm.describe_image(image_path, PROMPT_TEMPLATE,
                                 options={"temperature": 0.0,
                                          "num_predict": max_tokens}) or ""
    except Exception as exc:
        log.warning(f"VLM call failed for {image_path}: {exc}")
        return None
    md = _extract_markdown_table(raw)
    if md is None:
        log.warning(f"VLM output did not contain a markdown table for {image_path}")
        log.debug(f"  raw output (first 200 chars): {raw[:200]!r}")
        return None
    md = _strip_trailing_separator_rows(md)
    md = _strip_repetition_tail(md)
    md = _strip_trailing_separator_rows(md)  # in case repetition strip exposed more
    return md
