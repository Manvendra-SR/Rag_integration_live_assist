from __future__ import annotations

from pathlib import Path

from live_assist.core.config import get_settings

settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = PROJECT_ROOT / settings.rag_runtime_dir

UPLOAD_DIR = RUNTIME_DIR / "uploads"
PARSED_DIR = RUNTIME_DIR / "parsed_documents_fast"
CHUNKS_DIR = RUNTIME_DIR / "chunks"
ENRICHED_DIR = RUNTIME_DIR / f"chunks_{settings.rag_pipeline_name}"
BM25_DIR = RUNTIME_DIR / "bm25"
CHROMA_DB_DIR = RUNTIME_DIR / "chroma_db"
LOGS_INGESTION_DIR = RUNTIME_DIR / "logs" / "ingestion"
LOGS_QUERY_DIR = RUNTIME_DIR / "logs" / "query"

def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, PARSED_DIR, CHUNKS_DIR, ENRICHED_DIR, BM25_DIR, CHROMA_DB_DIR, LOGS_INGESTION_DIR, LOGS_QUERY_DIR):
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()
