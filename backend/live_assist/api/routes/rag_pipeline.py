from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

# ── Ingest status tracker (in-memory, per upload) ─────────────────────────────
_ingest_status: dict[str, dict[str, Any]] = {}
_UPLOAD_DIR = Path(__file__).resolve().parents[4] / "runtime" / "uploads"


# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    mode: str = "reranked"   # semantic | bm25 | hybrid | reranked
    top_k: int = 5
    doc_filter: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_ingest(job_id: str, pdf_path: Path) -> None:
    """Runs in a background thread via asyncio.to_thread."""
    from pipeline import ingest_pdf

    def on_progress(stage: str, msg: str) -> None:
        _ingest_status[job_id]["stages"].append({"stage": stage, "msg": msg})
        _ingest_status[job_id]["last_update"] = time.time()

    _ingest_status[job_id]["status"] = "running"
    result = ingest_pdf(pdf_path, on_progress=on_progress)
    _ingest_status[job_id].update(result)
    _ingest_status[job_id]["status"] = result.get("status", "error")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """
    Upload a PDF to be ingested into the RAG pipeline.
    Ingestion runs in the background — poll /rag/status/{job_id} for progress.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{int(time.time())}_{Path(file.filename).stem}"
    pdf_path = _UPLOAD_DIR / file.filename

    # Save uploaded file to disk
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialise status tracker
    _ingest_status[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "status": "queued",
        "stages": [],
        "submitted_at": time.time(),
    }

    # Run ingestion in a background thread so the response returns immediately
    background_tasks.add_task(
        asyncio.to_thread, _run_ingest, job_id, pdf_path
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "filename": file.filename,
        "message": "Ingestion started in the background. Poll /rag/status/{job_id} for updates.",
    }


@router.get("/status/{job_id}")
async def ingest_status(job_id: str) -> dict:
    """Poll the ingestion status for a given job_id."""
    if job_id not in _ingest_status:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _ingest_status[job_id]


@router.get("/jobs")
async def list_jobs() -> dict:
    """List all ingestion jobs and their statuses."""
    return {
        "jobs": [
            {
                "job_id": jid,
                "filename": info.get("filename"),
                "status": info.get("status"),
                "submitted_at": info.get("submitted_at"),
                "chunk_count": info.get("chunk_count"),
                "elapsed_seconds": info.get("elapsed_seconds"),
            }
            for jid, info in _ingest_status.items()
        ]
    }


@router.post("/query")
async def rag_query(request: QueryRequest) -> dict:
    """
    Run a direct RAG query against all indexed documents.
    This is separate from the live-assist workflow — useful for testing.
    """
    from pipeline import query as pipeline_query

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = await asyncio.to_thread(
        pipeline_query,
        request.question,
        mode=request.mode,
        top_k=request.top_k,
        doc_filter=request.doc_filter,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Query failed."))

    return result


@router.get("/debug")
async def debug_state() -> dict:
    """
    Inspect the current state of ChromaDB and BM25 index.
    Use this to diagnose 'no results' issues.
    """
    import os
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[4]
    runtime = backend_root / "runtime"

    info: dict = {"runtime_root": str(runtime)}

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", str(runtime / "chroma_db"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "DocumentChunk")
    info["chroma"] = {"persist_dir": chroma_dir, "collection": collection_name}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        all_cols = [getattr(c, "name", str(c)) for c in client.list_collections()]
        info["chroma"]["all_collections"] = all_cols
        if collection_name in all_cols:
            coll = client.get_collection(collection_name)
            count = coll.count()
            info["chroma"]["chunk_count"] = count
            if count > 0:
                # Sample first chunk to see what metadata looks like
                sample = coll.get(limit=1, include=["metadatas", "documents"])
                info["chroma"]["sample_metadata"] = sample["metadatas"][0] if sample["metadatas"] else {}
                info["chroma"]["sample_text_preview"] = (sample["documents"][0] or "")[:200] if sample["documents"] else ""
            else:
                info["chroma"]["chunk_count"] = 0
                info["chroma"]["warning"] = "Collection exists but is EMPTY — ingestion may not have completed"
        else:
            info["chroma"]["warning"] = f"Collection '{collection_name}' does NOT exist yet — no PDF has been ingested"
    except Exception as e:
        info["chroma"]["error"] = str(e)

    # ── BM25 Index ────────────────────────────────────────────────────────────
    bm25_dir = runtime / "bm25"
    bm25_pkl = bm25_dir / "anthropic.pkl"
    bm25_docs = bm25_dir / "anthropic.docs.json"
    info["bm25"] = {
        "index_path": str(bm25_pkl),
        "index_exists": bm25_pkl.exists(),
        "docs_path": str(bm25_docs),
        "docs_exist": bm25_docs.exists(),
    }
    if bm25_docs.exists():
        import json
        docs = json.loads(bm25_docs.read_text(encoding="utf-8"))
        info["bm25"]["chunk_count"] = len(docs)
        info["bm25"]["source_files"] = list({d.get("source_filename") for d in docs})

    # ── Runtime directories ───────────────────────────────────────────────────
    for name, subdir in [
        ("uploads", runtime / "uploads"),
        ("parsed", runtime / "parsed_documents_fast"),
        ("chunks", runtime / "chunks"),
        ("enriched", runtime / "chunks_anthropic"),
    ]:
        if subdir.exists():
            files = list(subdir.glob("*"))
            info[name] = {"exists": True, "file_count": len(files),
                          "files": [f.name for f in files[:10]]}
        else:
            info[name] = {"exists": False}

    return info

