from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter

from live_assist.core.config import get_settings
from live_assist.core.models import LiveFeedbackRequest, ManualQuestionRequest
from live_assist.core.terminal_log import api_timing
from live_assist.live_cycle.service import handle_manual_question, handle_transcript_turn
from live_assist.storage.transcript_store import session_store

router = APIRouter(prefix="/livefeedback", tags=["LiveFeedback"])
settings = get_settings()
_active_webhook_requests = 0
_active_webhook_lock = asyncio.Lock()


@router.post("/webhook")
async def live_feedback_webhook(request: LiveFeedbackRequest) -> dict:
    global _active_webhook_requests
    body = request.body
    session_id = body.call_id or settings.live_feedback_session_id
    metadata = body.metadata or {}
    chunk_id = int(metadata.get("chunk_id") or 0)
    turn_id = int(metadata.get("turn_id") or chunk_id or 0)
    started_at = time.perf_counter()
    async with _active_webhook_lock:
        _active_webhook_requests += 1
        active_requests = _active_webhook_requests
    api_timing(
        session_id,
        "webhook_received",
        chunk_id=chunk_id,
        turn_id=turn_id,
        active_requests=active_requests,
    )
    try:
        result = await handle_transcript_turn(
            session_id=session_id,
            speaker=body.speaker,
            text=body.transcript,
            timestamp=body.timestamp,
            has_interruptions=body.hasInterruptions,
            raw_text=body.raw_text,
            translated_text=body.translated_text,
            source=body.source or settings.input_source,
            metadata=metadata,
        )
        api_timing(
            session_id,
            "response_returned",
            chunk_id=chunk_id,
            turn_id=turn_id,
            duration_ms=f"{(time.perf_counter() - started_at) * 1000:.1f}",
        )
        return result
    finally:
        async with _active_webhook_lock:
            _active_webhook_requests -= 1
            active_requests = _active_webhook_requests
        api_timing(
            session_id,
            "webhook_completed",
            chunk_id=chunk_id,
            turn_id=turn_id,
            active_requests=active_requests,
        )


@router.post("/manual_question")
async def manual_question(request: ManualQuestionRequest) -> dict:
    session_id = request.call_id or settings.live_feedback_session_id
    return await handle_manual_question(
        session_id=session_id,
        question=request.question,
        timestamp=request.timestamp,
        source=request.source,
        metadata=request.metadata,
    )


@router.post("/call_end")
async def call_end(call_id: str) -> dict[str, str]:
    session_store.clear_all(call_id)
    return {"status": "completed", "call_id": call_id}
