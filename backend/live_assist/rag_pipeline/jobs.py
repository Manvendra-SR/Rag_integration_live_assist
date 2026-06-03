from __future__ import annotations

import time
from typing import Any

from live_assist.core.models import JobStatusResponse

_jobs: dict[str, dict[str, Any]] = {}

def create_job(job_id: str, filename: str) -> None:
    _jobs[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
        "stages": [],
        "submitted_at": time.time(),
        "chunk_count": 0,
        "total_tokens": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }

def update_job(job_id: str, updates: dict[str, Any]) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(updates)

def append_job_stage(job_id: str, stage: str, msg: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["stages"].append({"stage": stage, "msg": msg, "time": time.time()})

def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)

def get_all_jobs() -> list[dict[str, Any]]:
    return list(_jobs.values())
