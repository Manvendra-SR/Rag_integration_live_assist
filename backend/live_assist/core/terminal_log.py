from __future__ import annotations

import json
from datetime import datetime

from live_assist.core.config import get_settings


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def compact_text(value: str, limit: int = 300) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}..."
    return json.dumps(cleaned, ensure_ascii=False)


def backend_log(label: str, **values) -> None:
    parts = [f"{_timestamp()} | {label}"]
    for key, value in values.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print(" | ".join(parts), flush=True)


def client_timing(call_id: str, event: str, chunk_id: int | str = "", turn_id: int | str = "", **values) -> None:
    backend_log(
        "[ClientTiming]",
        call_id=call_id,
        chunk_id=chunk_id,
        turn_id=turn_id,
        event=event,
        **values,
    )


def api_timing(call_id: str, event: str, chunk_id: int | str = "", turn_id: int | str = "", **values) -> None:
    backend_log(
        "[APITiming]",
        call_id=call_id,
        chunk_id=chunk_id,
        turn_id=turn_id,
        event=event,
        **values,
    )


def api_summary_timing(call_id: str, chunk_id: int | str = "", turn_id: int | str = "", **values) -> None:
    backend_log(
        "[APISummaryTiming]",
        call_id=call_id,
        chunk_id=chunk_id,
        turn_id=turn_id,
        **values,
    )


def debug_log(message: str) -> None:
    settings = get_settings()
    if (
        settings.diagnostics_enabled
        or settings.desktop_audio_capture_mode == "desktop_native_diagnostic"
    ):
        print(message, flush=True)
