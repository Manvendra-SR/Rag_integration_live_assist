from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from live_assist.core.config import get_settings
from live_assist.core.models import (
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    JobStatusResponse,
)
from live_assist.rag_pipeline import documents, jobs, paths
from live_assist.rag_pipeline.ingestion import ingest_pdf

router = APIRouter(prefix="/rag/documents", tags=["RAG Documents"])
settings = get_settings()

def _run_ingest(
    job_id: str,
    user_id: str,
    document_id: str,
    filename: str,
    pdf_path: Path
) -> None:
    def on_progress(stage: str, msg: str) -> None:
        jobs.append_job_stage(job_id, stage, msg)

    jobs.update_job(job_id, {"status": "running"})
    
    result = ingest_pdf(
        pdf_path=pdf_path,
        user_id=user_id,
        document_id=document_id,
        on_progress=on_progress,
    )
    
    if "stages" in result:
        result["stage_stats"] = result.pop("stages")
    
    jobs.update_job(job_id, result)
    status = result.get("status", "error")
    jobs.update_job(job_id, {"status": status})

    # Save document info upon successful ingestion
    if status == "ok":
        doc_info = DocumentInfo(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            status="ready",
            uploaded_at=time.time(),
        )
    else:
        doc_info = DocumentInfo(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            status="error",
            uploaded_at=time.time(),
        )
    documents.save_document_info(doc_info)

    # ── Write Ingestion Log ───────────────────────────────────────────────────
    try:
        from live_assist.rag_pipeline.paths import LOGS_INGESTION_DIR
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_INGESTION_DIR / f"ingest_{document_id}_{timestamp_str}.log"
        
        stages = result.get("stage_stats", {})
        total_time = result.get("elapsed_seconds", 0)
        
        lines = [
            "=" * 50,
            "INGESTION EXECUTION LOG",
            f"Document ID: {document_id}",
            f"Filename:    {filename}",
            f"Status:      {status}",
            f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Time:  {total_time}s",
            "=" * 50,
            ""
        ]
        
        for stage_name, stats in stages.items():
            lines.append(f"[{stage_name.upper()}]")
            for k, v in stats.items():
                if k == "elapsed_s":
                    lines.append(f"  Duration: {v}s")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")
            
        if status == "error":
            lines.append("[ERROR DETAILS]")
            lines.append(f"  {result.get('error', 'Unknown error')}")
            
        log_file.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Failed to write ingestion log: {e}")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Using the fixed DEV user ID for now
    user_id = settings.live_feedback_user_id
    document_id = str(uuid.uuid4())
    job_id = f"{int(time.time())}_{document_id[:8]}"

    # Copy uploaded file
    paths.ensure_dirs()
    pdf_path = paths.UPLOAD_DIR / f"{document_id}_{file.filename}"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize tracking
    jobs.create_job(job_id, file.filename)
    
    # Save initial document info
    doc_info = DocumentInfo(
        document_id=document_id,
        user_id=user_id,
        filename=file.filename,
        status="ingesting",
        uploaded_at=time.time(),
    )
    documents.save_document_info(doc_info)

    # Dispatch background task
    background_tasks.add_task(
        asyncio.to_thread,
        _run_ingest,
        job_id,
        user_id,
        document_id,
        file.filename,
        pdf_path,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "filename": file.filename,
        "message": "Ingestion started in the background.",
    }


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> dict:
    # Filtered by current user
    user_id = settings.live_feedback_user_id
    docs = documents.get_documents_by_user(user_id)
    return {"documents": docs}
