from __future__ import annotations

import json
from pathlib import Path

from live_assist.core.models import DocumentInfo
from live_assist.rag_pipeline.paths import RUNTIME_DIR

_METADATA_DIR = RUNTIME_DIR / "documents_metadata"

def _ensure_dir() -> None:
    _METADATA_DIR.mkdir(parents=True, exist_ok=True)

def save_document_info(doc_info: DocumentInfo) -> None:
    _ensure_dir()
    path = _METADATA_DIR / f"{doc_info.document_id}.json"
    path.write_text(doc_info.model_dump_json(indent=2), encoding="utf-8")

def get_documents_by_user(user_id: str) -> list[DocumentInfo]:
    _ensure_dir()
    docs = []
    for path in _METADATA_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("user_id") == user_id:
                docs.append(DocumentInfo(**data))
        except Exception:
            pass
    return docs

def get_document(document_id: str) -> DocumentInfo | None:
    _ensure_dir()
    path = _METADATA_DIR / f"{document_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return DocumentInfo(**data)
        except Exception:
            return None
    return None
