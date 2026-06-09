"""Docling-based PDF parser for the Advanced RAG pipeline.

Outputs a JSON document with section hierarchy, typed blocks, and asset
provenance (tables as structured cells, figures with crops, equations as
their own type with optional LaTeX).

See AGENTS.md → T-01 for the full schema spec.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langfuse import observe
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    CodeItem,
    DoclingDocument,
    DocItemLabel,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# Equation/Formula items live under a couple of names depending on Docling version.
try:  # newer Docling
    from docling_core.types.doc import FormulaItem  # type: ignore
except ImportError:  # older — equations come through as TextItem with label == "formula"
    FormulaItem = None  # type: ignore

log = logging.getLogger(__name__)

from live_assist.rag_pipeline.paths import PROJECT_ROOT, RUNTIME_DIR
PARSED_OUTPUT_DIR = RUNTIME_DIR / "parsed_documents"
ASSET_OUTPUT_DIR = RUNTIME_DIR / "assets"

PARSER_NAME = "docling"

# ---------------------------------------------------------------------------
# Converter (singleton)
# ---------------------------------------------------------------------------


def _detect_device() -> AcceleratorDevice:
    """Prefer MPS on Apple Silicon, fall back to CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return AcceleratorDevice.MPS
        if torch.cuda.is_available():
            return AcceleratorDevice.CUDA
    except Exception:
        pass
    return AcceleratorDevice.CPU


def _build_pipeline_options() -> PdfPipelineOptions:
    opts = PdfPipelineOptions()
    # Core extraction
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    # Enrichment passes — only structural detection, no VLM-heavy LaTeX/code
    # transcription. Equations and code blocks are still DETECTED by the layout
    # model (RT-DETR) as their own block types; we only skip the slow VLM
    # (CodeFormulaV2) that tries to re-render them as LaTeX. The plain text
    # inside the equation/code bbox is still extracted normally — that's all
    # the downstream LLM needs to compute with formulas.
    # Default: VLM off (use fitz post-process for equation text — 30,000× faster).
    # Override per-doc with PARSER_ENRICH_FORMULAS=true for math-heavy papers
    # where proper LaTeX output is required (Greek letters, fractions, sub/super).
    enrich_formulas = os.environ.get("PARSER_ENRICH_FORMULAS", "").lower() == "true"
    enrich_code = os.environ.get("PARSER_ENRICH_CODE", "").lower() == "true"
    # Picture classification (chart/photo/diagram tag) — default ON now.
    # Disable with PARSER_PICTURE_CLASSIFICATION=false if you want max speed.
    classify_pictures = os.environ.get("PARSER_PICTURE_CLASSIFICATION", "true").lower() != "false"
    # Granite chart extraction REMOVED — HF download was anonymous-rate-limited,
    # so bar/line/pie charts now go through the Ollama VLM path along with
    # everything else. Opt back in by setting PARSER_CHART_EXTRACTION=true
    # (will require ~8 GB Granite model download from HF).
    extract_charts = os.environ.get("PARSER_CHART_EXTRACTION", "false").lower() == "true"
    opts.do_formula_enrichment = enrich_formulas
    opts.do_code_enrichment = enrich_code
    opts.do_picture_classification = classify_pictures
    opts.do_picture_description = False       # VLM-based — keep off (expensive)
    if hasattr(opts, "do_chart_extraction"):
        opts.do_chart_extraction = extract_charts
    log.info(f"  formula enrichment (VLM): {'ON' if enrich_formulas else 'OFF (fitz post-process)'}")
    log.info(f"  picture classification:    {'ON' if classify_pictures else 'OFF'}")
    log.info(f"  chart extraction (Granite):{'ON' if extract_charts else 'OFF (Ollama VLM handles charts)'}")
    # Asset generation — save figure/table crops for review and downstream VLM.
    opts.images_scale = 2.0
    opts.generate_picture_images = True
    opts.generate_table_images = True
    # Device
    device = _detect_device()
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=device)
    log.info(f"Docling accelerator: {device.value}")
    return opts


_CONVERTER: DocumentConverter | None = None


def get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        opts = _build_pipeline_options()
        _CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            }
        )
    return _CONVERTER


# ---------------------------------------------------------------------------
# Tree walkers — extract section hierarchy + section_path for every item
# ---------------------------------------------------------------------------


@dataclass
class WalkContext:
    section_path: list[str]
    section_levels: list[int]


def _save_asset_image(item: Any, dest_dir: Path, prefix: str) -> str | None:
    """Save a figure/table image crop to disk; return relative path or None."""
    try:
        pil_image = item.get_image(item._parent) if hasattr(item, "get_image") else None
    except Exception:
        pil_image = None
    if pil_image is None and hasattr(item, "image") and item.image is not None:
        # Some Docling versions expose .image directly with .pil_image inside
        try:
            pil_image = item.image.pil_image
        except Exception:
            pil_image = None
    if pil_image is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{prefix}.png"
    try:
        pil_image.save(file_path, format="PNG")
        return str(file_path.relative_to(PROJECT_ROOT))
    except Exception as exc:
        log.warning(f"Failed to save asset {file_path}: {exc}")
        return None


def _provenance(item: Any) -> tuple[int | None, list[float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    p0 = prov[0]
    page = getattr(p0, "page_no", None)
    bbox = getattr(p0, "bbox", None)
    bbox_list = None
    if bbox is not None:
        try:
            bbox_list = [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
        except Exception:
            bbox_list = None
    return page, bbox_list


def _serialize_table(item: TableItem, asset_dir: Path, table_idx: int) -> dict[str, Any]:
    page, bbox = _provenance(item)
    # Structured cells
    rows = []
    try:
        data = item.data
        if data is not None and data.grid is not None:
            for row in data.grid:
                rows.append([{"text": cell.text or "", "row_span": cell.row_span,
                              "col_span": cell.col_span,
                              "is_header": cell.column_header or cell.row_header}
                             for cell in row])
    except Exception as exc:
        log.warning(f"Table grid serialization failed: {exc}")
    # Markdown export — what the chunker will likely embed
    md_text = ""
    try:
        md_text = item.export_to_markdown(item._parent) if hasattr(item, "export_to_markdown") else ""
    except Exception:
        try:
            md_text = item.export_to_markdown()  # newer signature
        except Exception:
            md_text = ""
    image_rel = _save_asset_image(item, asset_dir / "tables", f"table_p{page or 0:03d}_{table_idx:03d}")
    return {
        "page": page,
        "bbox": bbox,
        "rows": rows,
        "markdown": md_text,
        "image_path": image_rel,
        "num_rows": item.data.num_rows if item.data else 0,
        "num_cols": item.data.num_cols if item.data else 0,
    }


def _serialize_figure(item: PictureItem, asset_dir: Path, fig_idx: int) -> dict[str, Any]:
    page, bbox = _provenance(item)
    image_rel = _save_asset_image(item, asset_dir / "figures", f"fig_p{page or 0:03d}_{fig_idx:03d}")
    # Caption: Docling stores captions as separate items linked via captions field
    caption_texts = []
    for cap_ref in getattr(item, "captions", []) or []:
        try:
            cap_item = cap_ref.resolve(item._parent) if hasattr(cap_ref, "resolve") else cap_ref
            if hasattr(cap_item, "text"):
                caption_texts.append(cap_item.text)
        except Exception:
            pass
    annotations = []
    chart_data: dict[str, Any] | None = None
    # NEW API (Docling ≥2.94): item.meta is a PictureMeta object with .classification
    # OLD API (deprecated): item.annotations is a list of PictureClassificationData
    meta = getattr(item, "meta", None)
    if meta is not None and not isinstance(meta, list):
        cls = getattr(meta, "classification", None)
        preds = getattr(cls, "predictions", None) if cls is not None else None
        if preds:
            top = preds[0]
            annotations.append({
                "kind": "classification",
                "class": getattr(top, "class_name", None),
                "confidence": float(getattr(top, "confidence", 0.0) or 0.0),
                "top_k": [
                    {"class": getattr(c, "class_name", None),
                     "confidence": float(getattr(c, "confidence", 0.0) or 0.0)}
                    for c in preds[:3]
                ],
                "provenance": getattr(top, "created_by", None),
            })
        # Granite Vision chart extraction (bar / line / pie only). Surface the
        # structured chart_data as both a markdown table and the raw cell grid,
        # so retrieval and the generation LLM can read exact values.
        tab = getattr(meta, "tabular_chart", None)
        td = getattr(tab, "chart_data", None) if tab is not None else None
        if td is not None:
            try:
                rows = []
                for row in (getattr(td, "table_cells", []) or []):
                    # TableData uses a flat list of TableCell; group by row index.
                    pass
                # Easiest path: ask docling_core to render TableData as markdown.
                md = None
                try:
                    from docling_core.types.doc.document import TableItem as _TI  # noqa
                    # Use the lightweight rendering helper if available
                    md_lines = []
                    grid = getattr(td, "grid", None)
                    if grid:
                        for r in grid:
                            md_lines.append("| " + " | ".join(
                                (c.text if hasattr(c, "text") else str(c)) for c in r
                            ) + " |")
                        if len(md_lines) >= 1:
                            md_lines.insert(1, "|" + "|".join(["---"] * len(grid[0])) + "|")
                        md = "\n".join(md_lines)
                except Exception:
                    md = None
                chart_data = {
                    "title": getattr(tab, "title", None),
                    "num_rows": getattr(td, "num_rows", None),
                    "num_cols": getattr(td, "num_cols", None),
                    "markdown": md,
                    "extracted_by": "granite_vision",
                }
            except Exception as exc:
                log.debug(f"chart_data serialization failed: {exc}")
                chart_data = None
    # Fall back to deprecated annotations list
    if not annotations:
        for ann in getattr(item, "annotations", []) or []:
            if hasattr(ann, "predicted_classes") and ann.predicted_classes:
                top = ann.predicted_classes[0]
                annotations.append({
                    "kind": getattr(ann, "kind", "classification"),
                    "class": getattr(top, "class_name", None),
                    "confidence": float(getattr(top, "confidence", 0.0) or 0.0),
                    "top_k": [
                        {"class": getattr(c, "class_name", None),
                         "confidence": float(getattr(c, "confidence", 0.0) or 0.0)}
                        for c in ann.predicted_classes[:3]
                    ],
                    "provenance": getattr(ann, "provenance", None),
                })
    return {
        "page": page,
        "bbox": bbox,
        "caption": " ".join(caption_texts).strip() or None,
        "image_path": image_rel,
        "annotations": annotations,
        "chart_data": chart_data,
    }


def _serialize_code(item: CodeItem) -> dict[str, Any]:
    page, bbox = _provenance(item)
    return {
        "page": page,
        "bbox": bbox,
        "text": item.text or "",
        "language": getattr(item, "code_language", None) or None,
    }


def _serialize_equation(item: Any) -> dict[str, Any]:
    page, bbox = _provenance(item)
    return {
        "page": page,
        "bbox": bbox,
        "text": getattr(item, "text", "") or "",
        "latex": getattr(item, "orig", None),
    }


def _is_equation(item: Any) -> bool:
    if FormulaItem is not None and isinstance(item, FormulaItem):
        return True
    label = getattr(item, "label", None)
    if label is not None and str(label).endswith("formula"):
        return True
    return False


def _classify_block(item: Any) -> str:
    if isinstance(item, SectionHeaderItem):
        return "section_heading"
    if isinstance(item, TableItem):
        return "table"
    if isinstance(item, PictureItem):
        return "figure"
    if isinstance(item, CodeItem):
        return "code"
    if _is_equation(item):
        return "equation"
    label = str(getattr(item, "label", "") or "").lower()
    if "list" in label:
        return "list_item"
    if "caption" in label:
        return "caption"
    if "title" in label and not isinstance(item, SectionHeaderItem):
        return "title"
    if "footnote" in label:
        return "footnote"
    if "header" in label or "footer" in label:
        return "page_marginal"
    return "paragraph"


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


@observe(name="parse_pdf")
def parse_pdf(pdf_path: Path, output_dir: Path = PARSED_OUTPUT_DIR) -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    log.info(f"Parsing {pdf_path.name} with Docling …")
    converter = get_converter()

    t0 = time.perf_counter()
    result = converter.convert(str(pdf_path))
    elapsed = time.perf_counter() - t0
    log.info(f"  Docling parse: {elapsed:.2f}s")

    doc: DoclingDocument = result.document
    doc_id = str(uuid.uuid4())
    asset_dir = ASSET_OUTPUT_DIR / pdf_path.stem
    asset_dir.mkdir(parents=True, exist_ok=True)

    # ------------- Walk the tree, build blocks with section_path -------------
    blocks: list[dict[str, Any]] = []
    section_stack: list[tuple[int, str]] = []  # [(level, heading), ...]
    table_idx = 0
    figure_idx = 0

    for item, level in doc.iterate_items():
        page, bbox = _provenance(item)
        block_type = _classify_block(item)

        # Maintain heading stack — pop until the current item's level fits
        if isinstance(item, SectionHeaderItem):
            heading_level = getattr(item, "level", 1) or 1
            while section_stack and section_stack[-1][0] >= heading_level:
                section_stack.pop()
            section_stack.append((heading_level, item.text or ""))

        section_path = [h for _, h in section_stack]

        block: dict[str, Any] = {
            "block_id": str(uuid.uuid4()),
            "type": block_type,
            "page": page,
            "bbox": bbox,
            "text": getattr(item, "text", "") or "",
            "section_path": section_path,
            "tree_level": level,
            "metadata": {},
        }

        if isinstance(item, TableItem):
            block["metadata"] = _serialize_table(item, asset_dir, table_idx)
            table_idx += 1
        elif isinstance(item, PictureItem):
            block["metadata"] = _serialize_figure(item, asset_dir, figure_idx)
            figure_idx += 1
        elif isinstance(item, CodeItem):
            block["metadata"] = _serialize_code(item)
        elif _is_equation(item):
            block["metadata"] = _serialize_equation(item)
            # Fix: Docling puts equation content in `orig` (latex), leaving
            # item.text empty. Copy it into block.text so downstream chunker
            # and retrieval can actually see the equation.
            if not block["text"]:
                latex = block["metadata"].get("latex")
                if latex:
                    block["text"] = str(latex)

        blocks.append(block)

    # ------------- Build hierarchical document_tree -------------
    document_tree = _build_section_tree(blocks, doc)

    # ------------- Asset summary -------------
    counts = {
        "figure_count": sum(1 for b in blocks if b["type"] == "figure"),
        "table_count":  sum(1 for b in blocks if b["type"] == "table"),
        "equation_count": sum(1 for b in blocks if b["type"] == "equation"),
        "code_block_count": sum(1 for b in blocks if b["type"] == "code"),
        "section_heading_count": sum(1 for b in blocks if b["type"] == "section_heading"),
        "paragraph_count": sum(1 for b in blocks if b["type"] == "paragraph"),
    }

    # Page count
    page_count = 0
    try:
        page_count = len(doc.pages or {})
    except Exception:
        page_count = max((b.get("page") or 0) for b in blocks) or 0

    payload = {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "parser_name": PARSER_NAME,
        "parser_version": _get_docling_version(),
        "parsed_at": datetime.utcnow().isoformat() + "Z",
        "doc_profile": {
            "page_count": page_count,
            "title": doc.name or pdf_path.stem,
        },
        "document_tree": document_tree,
        "blocks": blocks,
        "assets": counts,
        "timings": {
            "parse_seconds": round(elapsed, 3),
            "device": str(_detect_device().value),
        },
        # Native Docling exports (handy for downstream consumers)
        "exports": {
            "markdown": doc.export_to_markdown(),
        },
    }

    # Post-process: strip logos, attach captions, repair equations, image tables,
    # and run VLM description on every meaningful figure
    try:
        from .figure_post import (
            strip_marginal_figures, attach_orphan_captions, repair_equation_text,
            repair_image_tables_with_ocr, describe_figures_with_vlm,
        )
    except ImportError:
        from figure_post import (
            strip_marginal_figures, attach_orphan_captions, repair_equation_text,
            repair_image_tables_with_ocr, describe_figures_with_vlm,
        )
    # Get per-page heights from Docling's page dict for accurate top/bottom detection
    page_heights = {}
    try:
        for pno, p in (doc.pages or {}).items():
            if p and hasattr(p, "size") and p.size:
                page_heights[int(pno)] = float(p.size.height)
    except Exception:
        pass
    payload = strip_marginal_figures(payload, page_heights=page_heights or None)
    n_stripped = payload.get("assets", {}).get("marginal_figures_stripped", 0)
    if n_stripped:
        log.info(f"  figure post-process: retyped {n_stripped} logo/banner false-positives as page_marginal")

    # Attach orphan captions to their nearest figure/table on same page
    payload = attach_orphan_captions(payload)
    a = payload.get("assets", {})
    n_cap_fig = a.get("captions_attached_to_figures", 0)
    n_cap_tab = a.get("captions_attached_to_tables", 0)
    n_cap_orphan = a.get("captions_orphaned", 0)
    if n_cap_fig or n_cap_tab or n_cap_orphan:
        log.info(f"  caption post-process: attached {n_cap_fig} to figures, "
                 f"{n_cap_tab} to tables, {n_cap_orphan} still orphan")

    # Repair equation text from PDF text layer (fitz) when Docling output is garbled.
    # Skipped automatically if VLM formula enrichment already ran (Docling text is clean).
    if os.environ.get("PARSER_ENRICH_FORMULAS", "").lower() != "true":
        payload = repair_equation_text(payload, str(pdf_path))
        n_rep = payload.get("assets", {}).get("equations_repaired_by_fitz", 0)
        n_kept = payload.get("assets", {}).get("equations_kept_docling_text", 0)
        if n_rep or n_kept:
            log.info(f"  equation post-process: {n_rep} text repaired via fitz, {n_kept} kept Docling text")

    # VLM fallback for image-rendered tables (TableFormer failed = num_rows=0)
    payload = repair_image_tables_with_ocr(payload, project_root=str(PROJECT_ROOT))
    a = payload.get("assets", {})
    log.info(f"  table extraction: tier1(TableFormer)={a.get('tables_tier1_ok',0)} "
             f"tier2(VLM)={a.get('tables_tier2_ok',0)} "
             f"failed={a.get('tables_failed',0)} no_image={a.get('tables_no_image',0)}")

    # VLM description for every meaningful figure (OCR embedded text + describe)
    if os.environ.get("PARSER_DESCRIBE_FIGURES", "true").lower() != "false":
        payload = describe_figures_with_vlm(payload, project_root=str(PROJECT_ROOT))
        a = payload.get("assets", {})
        log.info(f"  figure description: described={a.get('figures_described_by_vlm',0)} "
                 f"ocr_only={a.get('figures_ocr_only',0)} "
                 f"vlm_failed={a.get('figures_vlm_failed',0)} "
                 f"skipped_class={a.get('figures_skipped_by_class',0)} "
                 f"skipped_small={a.get('figures_skipped_too_small',0)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.parsed.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  Saved {out_path}  ({len(blocks)} blocks)")
    return out_path


def _get_docling_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("docling")
    except Exception:
        return "unknown"


def _build_section_tree(blocks: list[dict[str, Any]], doc: DoclingDocument) -> dict[str, Any]:
    """Nest paragraphs/tables/figures under their owning section headings."""
    root = {"title": doc.name or "Document", "sections": [], "orphans": []}
    section_stack: list[dict[str, Any]] = []

    for block in blocks:
        if block["type"] == "section_heading":
            node = {
                "heading": block["text"],
                "level": block.get("tree_level", 1),
                "section_path": block["section_path"],
                "page_start": block["page"],
                "page_end": block["page"],
                "subsections": [],
                "items": [],
            }
            # Pop stack to the right depth
            while section_stack and section_stack[-1]["level"] >= node["level"]:
                section_stack.pop()
            if section_stack:
                section_stack[-1]["subsections"].append(node)
            else:
                root["sections"].append(node)
            section_stack.append(node)
        else:
            if section_stack:
                parent = section_stack[-1]
                parent["items"].append({
                    "type": block["type"],
                    "page": block["page"],
                    "text": block["text"][:600] if block["text"] else "",
                    "metadata_keys": list(block.get("metadata", {}).keys()),
                })
                # Expand parent page_end
                if block.get("page") is not None and parent.get("page_end") is not None:
                    parent["page_end"] = max(parent["page_end"], block["page"])
            else:
                root["orphans"].append({
                    "type": block["type"],
                    "page": block["page"],
                    "text": (block["text"] or "")[:200],
                })
    return root
