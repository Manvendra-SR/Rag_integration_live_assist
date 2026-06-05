from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from live_assist.audio.speakers import normalize_speaker, workflow_target_for_speaker
from live_assist.core.config import get_settings
from live_assist.core.diagnostics import log_event
from live_assist.core.models import AssistResult, Speaker, TranscriptTurn
from live_assist.core.terminal_log import api_timing, debug_log
from live_assist.storage.sqlite import get_recent_transcript_turns
from live_assist.storage.transcript_store import persist_turn, session_store

settings = get_settings()
_workflow_app = None


def log_text(value: str, limit: int = 300) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}..."
    return json.dumps(cleaned, ensure_ascii=False)


def get_workflow_app():
    global _workflow_app
    if _workflow_app is None:
        from live_assist.live_cycle.graph import create_workflow

        _workflow_app = create_workflow()
    return _workflow_app


def _workflow_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{settings.live_feedback_user_id}::{session_id}"}}


def _coerce_timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    try:
        return float(value)
    except (TypeError, ValueError):
        return time.time()


def _invoke_turn_workflow(
    session_id: str,
    speaker: Speaker,
    text: str,
    *,
    manual_question: bool = False,
    trace_id: str = "",
    utterance_id: str = "",
    chunk_id: int = 0,
    turn_id: int = 0,
    doc_filter: str = "",
    retrieval_mode: str = "",
) -> dict[str, Any]:
    api_timing(
        session_id,
        "langgraph_worker_started",
        chunk_id=chunk_id,
        turn_id=turn_id,
        thread=threading.current_thread().name,
    )
    started_at = time.perf_counter()
    try:
        return get_workflow_app().invoke(
            {
                "speaker": speaker.value,
                "trace_id": trace_id,
                "utterance_id": utterance_id,
                "chunk_id": chunk_id,
                "turn_id": turn_id,
                "turn_text": text,
                "question": text,
                "user_id": settings.live_feedback_user_id,
                "session_id": session_id,
                "manual_question": manual_question,
                "doc_filter": doc_filter,
                "retrieval_mode": retrieval_mode,
            },
            config=_workflow_config(session_id),
        )
    finally:
        api_timing(
            session_id,
            "langgraph_worker_completed",
            chunk_id=chunk_id,
            turn_id=turn_id,
            thread=threading.current_thread().name,
            duration_ms=f"{(time.perf_counter() - started_at) * 1000:.1f}",
        )


def _count_customer_turns(messages: list[dict[str, Any]]) -> int:
    return len(
        [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
    )


def _build_summary_turns_from_chunks(
    session_id: str,
    last_n_turns: int,
) -> list[dict[str, str]]:
    chunks = get_recent_transcript_turns(session_id, limit=max(last_n_turns * 4, 20))
    turns = []
    current_turn: dict[str, str] | None = None

    for chunk in chunks:
        speaker = str(chunk.get("speaker", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        if speaker == Speaker.CUSTOMER.value:
            current_turn = {"question": text, "assistant": "", "human_worker": ""}
            turns.append(current_turn)
            continue

        if current_turn is None:
            current_turn = {"question": "", "assistant": "", "human_worker": ""}
            turns.append(current_turn)

        if speaker == Speaker.ASSISTANT.value:
            current_turn["assistant"] = text
        elif speaker == Speaker.WORKER.value:
            if current_turn["human_worker"]:
                current_turn["human_worker"] = f"{current_turn['human_worker']}\n{text}"
            else:
                current_turn["human_worker"] = text

    return turns[-last_n_turns:]


async def _store_assistant_answer(session_id: str, answer: str) -> None:
    clean_answer = answer.strip()
    if not clean_answer:
        return

    turn = TranscriptTurn(
        session_id=session_id,
        speaker=Speaker.ASSISTANT,
        timestamp=time.time(),
        raw_text=clean_answer,
        translated_text=None,
        source="live_assist",
        triggered_live_assist=False,
        workflow_target="assistant_answer",
        metadata={"response_type": "live_assist_answer"},
    )
    await persist_turn(turn)


async def _run_live_assist_workflow(
    *,
    session_id: str,
    speaker: Speaker,
    text: str,
    manual_question: bool = False,
    trace_id: str = "",
    utterance_id: str = "",
    chunk_id: int = 0,
    turn_id: int = 0,
    doc_filter: str = "",
    retrieval_mode: str = "",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        api_timing(session_id, "langgraph_to_thread_submitted", chunk_id=chunk_id, turn_id=turn_id)
        log_event(
            "workflow_start",
            call_id=session_id,
            speaker=speaker.value,
            trace_id=trace_id,
            utterance_id=utterance_id,
            text=text,
            manual_question=manual_question,
        )
        # ── Semantic Cache Lookup (only when a specific doc_filter is set) ──────
        cache_hit_result: dict | None = None
        cache_lookup_ms: float = 0.0
        cache_result_label: str = "SKIPPED"
        cache_similarity: float | None = None
        _cache_instance = None

        if manual_question and doc_filter:
            try:
                from live_assist.rag_pipeline.paths import CACHE_DIR, INDEX_VERSION_FILE
                from live_assist.rag_pipeline.semantic_cache import SemanticCache
                _settings = get_settings()
                if _settings.rag_cache_enabled:
                    _cache_instance = SemanticCache(
                        CACHE_DIR,
                        INDEX_VERSION_FILE,
                        similarity_threshold=_settings.rag_cache_similarity_threshold,
                        max_age_days=_settings.rag_cache_max_age_days,
                    )
                    _t_cache = time.perf_counter()
                    hit = _cache_instance.lookup(
                        raw_query=text,
                        doc_filter=doc_filter,
                        retrieval_mode=retrieval_mode or _settings.rag_retrieval_mode,
                    )
                    cache_lookup_ms = (time.perf_counter() - _t_cache) * 1000
                    cache_similarity = hit.get("similarity")
                    if hit.get("result") == "HIT":
                        cache_hit_result = hit["workflow_response"]
                        cache_result_label = "HIT"
                        sim_str = f"{cache_similarity:.4f}" if cache_similarity is not None else "N/A"
                        debug_log(
                            f"[SemanticCache] HIT | sim={sim_str} | "
                            f"lookup_ms={cache_lookup_ms:.1f} | session={session_id}"
                        )
                    else:
                        cache_result_label = "MISS"
                        sim_str = f"{cache_similarity:.4f}" if cache_similarity is not None else "N/A"
                        debug_log(
                            f"[SemanticCache] MISS | best_sim={sim_str} | "
                            f"lookup_ms={cache_lookup_ms:.1f} | session={session_id}"
                        )
            except Exception as _ce:
                debug_log(f"[SemanticCache] lookup error (ignored): {_ce}")
                cache_result_label = "ERROR"

        if cache_hit_result is not None:
            workflow_response = cache_hit_result
            workflow_response["_cache_hit"] = True
            workflow_response["_cache_similarity"] = cache_similarity
            workflow_response["_cache_lookup_ms"] = cache_lookup_ms
            workflow_response["_cache_result"] = cache_result_label
        else:
            workflow_response = await asyncio.to_thread(
                _invoke_turn_workflow,
                session_id,
                speaker,
                text,
                manual_question=manual_question,
                trace_id=trace_id,
                utterance_id=utterance_id,
                chunk_id=chunk_id,
                turn_id=turn_id,
                doc_filter=doc_filter,
                retrieval_mode=retrieval_mode,
            )
            workflow_response["_cache_hit"] = False
            workflow_response["_cache_lookup_ms"] = cache_lookup_ms
            workflow_response["_cache_result"] = cache_result_label
            workflow_response["_cache_similarity"] = cache_similarity  # may be None if no entries yet

            # Write to cache on a successful miss (only for manual questions with doc_filter)
            if manual_question and doc_filter and _cache_instance and workflow_response.get("answer"):
                try:
                    from live_assist.core.config import get_settings as _gs
                    _cache_instance.write(
                        raw_query=text,
                        doc_filter=doc_filter,
                        retrieval_mode=retrieval_mode or _gs().rag_retrieval_mode,
                        workflow_response=workflow_response,
                    )
                except Exception as _cw:
                    debug_log(f"[SemanticCache] write error (ignored): {_cw}")
        api_timing(
            session_id,
            "langgraph_to_thread_completed",
            chunk_id=chunk_id,
            turn_id=turn_id,
            duration_ms=f"{(time.perf_counter() - started_at) * 1000:.1f}",
        )
        answer = workflow_response.get("answer", "")
        if answer:
            await _store_assistant_answer(session_id, answer)
        debug_log(
            f"[Timing] call={session_id} stage=workflow_done "
            f"duration_ms={(time.perf_counter() - started_at) * 1000:.1f}"
        )
        log_event(
            "workflow_done",
            call_id=session_id,
            speaker=speaker.value,
            trace_id=trace_id,
            utterance_id=utterance_id,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            answer=answer,
            should_generate_answer=workflow_response.get("should_generate_answer"),
        )
        invoked_status = "workflow_invoked" if workflow_response.get("should_generate_answer") else "context_updated_only"

        metadata = {
            "product": workflow_response.get("product", ""),
            "product_context": workflow_response.get("product_context", ""),
            "route": workflow_response.get("route", "rag_answer"),
            "last_5_turns": workflow_response.get("last_5_turns", []),
            "enriched_query": workflow_response.get("rewriten_question", ""),
            "enrich_duration_ms": workflow_response.get("enrich_duration_ms", 0.0),
            "rag_top_chunks": workflow_response.get("rag_top_chunks", []),
            "rag_retrieve_duration_ms": workflow_response.get("rag_retrieve_duration_ms", 0.0),
            "generation_duration_ms": workflow_response.get("generation_duration_ms", 0.0),
            "workflow_duration_ms": (time.perf_counter() - started_at) * 1000,
        }

        # ── Write Query Log ───────────────────────────────────────────────────
        try:
            from live_assist.rag_pipeline.paths import LOGS_QUERY_DIR
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOGS_QUERY_DIR / f"query_{session_id}_{turn_id}_{timestamp_str}.log"
            
            lines = [
                "=" * 50,
                "QUERY EXECUTION LOG",
                f"Session ID:   {session_id}",
                f"Turn ID:      {turn_id}",
                f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total Time:   {metadata['workflow_duration_ms']:.1f}ms",
                f"  (Stage sum covers enrichment + retrieval + generation.",
                f"   Remainder is: graph overhead, context ingestion, DB persist, log_event calls.)",
                "=" * 50,
                "",
                "[CACHE]",
                f"  Eligible:    {'Yes' if doc_filter else 'No (All Documents)'}",
                f"  Result:      {workflow_response.get('_cache_result', 'SKIPPED')}",
                f"  Lookup Time: {workflow_response.get('_cache_lookup_ms', 0.0):.1f}ms",
            ]

            _cache_sim = workflow_response.get("_cache_similarity")
            _cache_result = workflow_response.get("_cache_result", "SKIPPED")
            if _cache_sim is not None:
                from live_assist.core.config import get_settings as _gs2
                _thresh = _gs2().rag_cache_similarity_threshold
                lines.append(
                    f"  Similarity:  {_cache_sim:.4f}  (threshold: {_thresh})"
                )
            elif _cache_result == "SKIPPED":
                lines.append("  Similarity:  N/A  (cache not evaluated)")

            _is_hit = workflow_response.get("_cache_hit", False)
            lines += [
                "",
                "[QUERY PREPROCESSING (ENRICHMENT)]",
                f"  Raw Query:        {workflow_response.get('question', text)}",
            ]
            if _is_hit:
                lines.append("  Enriched Query:   (served from cache — enrichment skipped)")
                lines.append(f"  Duration:         0.0ms")
            else:
                lines += [
                    f"  Enriched Query:   {metadata['enriched_query']}",
                    f"  Duration:         {metadata['enrich_duration_ms']:.1f}ms",
                ]

            lines.append("")
            if _is_hit:
                lines += [
                    "[RETRIEVAL]",
                    "  Served from cache — retrieval skipped",
                    "",
                    "[GENERATION]",
                    "  Served from cache — generation skipped",
                    "",
                ]
            else:
                lines += [
                    "[RETRIEVAL]",
                    f"  Route:            {metadata['route']}",
                    f"  Retrieval Mode:   {workflow_response.get('retrieval_mode', 'default')}",
                    f"  Doc Filter:       {doc_filter or 'None'}",
                    f"  Product Filter:   {metadata['product']}",
                    f"  Chunks Found:     {len(metadata['rag_top_chunks'])}",
                    f"  Duration:         {metadata['rag_retrieve_duration_ms']:.1f}ms",
                    "",
                    "[GENERATION]",
                    f"  Answer Generated: {'Yes' if answer else 'No'}",
                    f"  Duration:         {metadata['generation_duration_ms']:.1f}ms",
                    "",
                ]

            lines += [
                "=" * 50,
                "RETRIEVED CHUNKS",
                "=" * 50,
                ""
            ]

            raw_chunks = workflow_response.get("rag_raw_chunks", [])
            for i, chunk in enumerate(raw_chunks, start=1):
                doc_name = chunk.get("source_filename", "Unknown")
                score = chunk.get("relevance_score", chunk.get("score", "N/A"))
                if isinstance(score, float):
                    score = f"{score:.4f}"
                chunk_id = chunk.get("chunk_id", "N/A")
                content = chunk.get("text_with_context", chunk.get("text", chunk.get("page_content", "")))
                
                lines.extend([
                    f"[Chunk {i}]",
                    f"Document: {doc_name}",
                    f"Score: {score}",
                    f"Chunk ID: {chunk_id}",
                    "",
                    "Content:",
                    content,
                    "",
                    "-" * 50,
                    ""
                ])

            log_file.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"Failed to write query log: {e}")

        return AssistResult(
            status=invoked_status,
            answer=answer,
            summary_result={
                "running_summary": workflow_response.get("running_summary", ""),
                "conversation_turn_count": workflow_response.get("conversation_turn_count", 0),
                "summary_turn_count": workflow_response.get("summary_turn_count", 0),
            },
            metadata=metadata,
        ).model_dump()
    except Exception as exc:
        debug_log(
            f"[Live Assist Error] call={session_id} type={type(exc).__name__} "
            f"error={log_text(str(exc) or repr(exc))}"
        )
        log_event(
            "workflow_failed",
            call_id=session_id,
            speaker=speaker.value,
            trace_id=trace_id,
            utterance_id=utterance_id,
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        return AssistResult(
            status="workflow_failed",
            answer="",
            metadata={
                "error_type": type(exc).__name__,
                "error": str(exc) or repr(exc),
            },
        ).model_dump()


def _maybe_generate_summary(session_id: str) -> dict[str, Any]:
    from live_assist.live_cycle.graph import summarize_conversation

    workflow_app = get_workflow_app()
    snapshot = workflow_app.get_state(_workflow_config(session_id))
    values = snapshot.values or {}
    messages = values.get("messages", [])
    completed_turns = _count_customer_turns(messages)

    if completed_turns == 0 or completed_turns % 5 != 0:
        return {"status": "summary_not_due", "completed_turns": completed_turns}

    recent_turns = _build_summary_turns_from_chunks(
        session_id=session_id,
        last_n_turns=settings.live_feedback_recent_turns,
    )
    result = summarize_conversation(
        user_id=values.get("user_id") or settings.live_feedback_user_id,
        session_id=values.get("session_id") or session_id,
        summary_turns=recent_turns,
        product=values.get("product", ""),
    )
    result["completed_turns"] = completed_turns
    result["turns_used"] = len(recent_turns)
    return result


async def handle_transcript_turn(
    *,
    session_id: str,
    speaker: str,
    text: str,
    timestamp: Any = None,
    has_interruptions: bool = False,
    raw_text: str | None = None,
    translated_text: str | None = None,
    source: str = "stream",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_speaker(speaker)
    clean_text = text.strip()
    if not clean_text:
        return {"status": "ignored", "reason": "empty_transcript"}

    workflow_target = workflow_target_for_speaker(normalized)
    should_trigger = normalized == Speaker.CUSTOMER
    metadata = metadata or {}
    trace_id = str(metadata.get("trace_id") or "")
    utterance_id = str(metadata.get("utterance_id") or "")
    chunk_id = int(metadata.get("chunk_id") or 0)
    turn_id = int(metadata.get("turn_id") or chunk_id or 0)
    api_timing(session_id, "transcript_persist_started", chunk_id=chunk_id, turn_id=turn_id)
    debug_log(
        f"[Turn Received] call={session_id} speaker={normalized.value} "
        f"source={source} text={log_text(clean_text)}"
    )
    log_event(
        "turn_received",
        call_id=session_id,
        speaker=normalized.value,
        trace_id=trace_id,
        utterance_id=utterance_id,
        source=source,
        text=clean_text,
    )
    turn = TranscriptTurn(
        session_id=session_id,
        speaker=normalized,
        timestamp=_coerce_timestamp(timestamp),
        raw_text=raw_text or clean_text,
        translated_text=translated_text if translated_text else None,
        source=source,
        triggered_live_assist=should_trigger,
        workflow_target=workflow_target,
        metadata={
            "hasInterruptions": has_interruptions,
            **metadata,
        },
    )
    await persist_turn(turn)
    api_timing(session_id, "transcript_persist_completed", chunk_id=chunk_id, turn_id=turn_id)

    if normalized == Speaker.CUSTOMER:
        debug_log(
            f"[Live Assist Trigger] call={session_id} source={source} "
            f"question={log_text(clean_text)}"
        )
    elif normalized == Speaker.WORKER:
        debug_log(
            f"[Live Assist Context Update] call={session_id} speaker={normalized.value} "
            f"text={log_text(clean_text)}"
        )

    action_result = await _run_live_assist_workflow(
        session_id=session_id,
        speaker=normalized,
        text=clean_text,
        trace_id=trace_id,
        utterance_id=utterance_id,
        chunk_id=chunk_id,
        turn_id=turn_id,
    )

    current_batch = session_store.get_current_batch(session_id)
    response_status = "processed"
    if len(current_batch) < settings.live_feedback_batch_size:
        response_status = "buffered"
    else:
        session_store.clear_current_batch(session_id)

    return {
        "status": response_status,
        "session_id": session_id,
        "call_id": session_id,
        "speaker": normalized.value,
        "workflow_target": workflow_target,
        "utterance": clean_text,
        "triggered_live_assist": should_trigger,
        "action_result": action_result,
    }


async def handle_manual_question(
    *,
    session_id: str,
    question: str,
    timestamp: Any = None,
    source: str = "agent_manual_question",
    metadata: dict[str, Any] | None = None,
    doc_filter: str = "",
    retrieval_mode: str = "",
) -> dict[str, Any]:
    clean_question = question.strip()
    if not clean_question:
        return {"status": "ignored", "reason": "empty_question"}
    metadata = metadata or {}
    trace_id = str(metadata.get("trace_id") or "")
    utterance_id = str(metadata.get("utterance_id") or "")

    debug_log(
        f"[Turn Received] call={session_id} speaker={Speaker.WORKER.value} "
        f"source={source} text={log_text(clean_question)}"
    )
    log_event(
        "turn_received",
        call_id=session_id,
        speaker=Speaker.WORKER.value,
        trace_id=trace_id,
        utterance_id=utterance_id,
        source=source,
        text=clean_question,
        manual_question=True,
    )

    turn = TranscriptTurn(
        session_id=session_id,
        speaker=Speaker.WORKER,
        timestamp=_coerce_timestamp(timestamp),
        raw_text=clean_question,
        translated_text=None,
        source=source,
        triggered_live_assist=True,
        workflow_target="manual_live_assist_question",
        metadata={
            "manual_question": True,
            **metadata,
        },
    )
    await persist_turn(turn)

    debug_log(
        f"[Live Assist Trigger] call={session_id} source={source} "
        f"question={log_text(clean_question)}"
    )
    action_result = await _run_live_assist_workflow(
        session_id=session_id,
        speaker=Speaker.WORKER,
        text=clean_question,
        manual_question=True,
        trace_id=trace_id,
        utterance_id=utterance_id,
        doc_filter=doc_filter,
        retrieval_mode=retrieval_mode,
    )

    return {
        "status": "processed",
        "session_id": session_id,
        "call_id": session_id,
        "speaker": Speaker.WORKER.value,
        "workflow_target": "manual_live_assist_question",
        "utterance": clean_question,
        "triggered_live_assist": True,
        "action_result": action_result,
    }
