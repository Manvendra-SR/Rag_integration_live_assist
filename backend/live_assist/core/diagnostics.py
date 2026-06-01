from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from live_assist.core.config import BACKEND_DIR

_QUEUE: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=10000)
_STARTED = False
_STOPPING = False
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_FILES: dict[str, Path] = {}
_LATEST_FILES: dict[str, Path] = {}


def _diagnostics_enabled() -> bool:
    explicit = os.getenv("LIVE_ASSIST_DIAGNOSTICS_ENABLED", "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    mode = os.getenv("DESKTOP_AUDIO_CAPTURE_MODE", "").strip().lower()
    return mode == "desktop_native_diagnostic"


def _safe_call_id(call_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_id or "unknown")
    return cleaned[:120] or "unknown"


def _snippet(value: Any, limit: int = 300) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = " ".join(value.split())
    if len(cleaned) > limit:
        return f"{cleaned[:limit]}..."
    return cleaned


def _diagnostics_dir() -> Path:
    path = BACKEND_DIR / "data" / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths_for_call(call_id: str) -> tuple[Path, Path]:
    safe_call = _safe_call_id(call_id)
    if safe_call not in _FILES:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        directory = _diagnostics_dir()
        _FILES[safe_call] = directory / f"live_assist_{safe_call}_{timestamp}.jsonl"
        _LATEST_FILES[safe_call] = directory / f"latest_{safe_call}.jsonl"
        _LATEST_FILES[safe_call].write_text("", encoding="utf-8")
    return _FILES[safe_call], _LATEST_FILES[safe_call]


def _writer() -> None:
    while True:
        event = _QUEUE.get()
        if event is None:
            _QUEUE.task_done()
            return
        try:
            call_id = str(event.get("call_id") or "unknown")
            log_path, latest_path = _paths_for_call(call_id)
            line = json.dumps(event, ensure_ascii=False, default=str)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            with latest_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:
            print(f"[Diagnostics Error] type={type(exc).__name__} error={exc}", flush=True)
        finally:
            _QUEUE.task_done()


def start_diagnostics_logger() -> None:
    global _STARTED, _THREAD
    with _LOCK:
        if _STARTED:
            return
        thread = threading.Thread(target=_writer, name="live-assist-diagnostics", daemon=True)
        thread.start()
        _THREAD = thread
        _STARTED = True
        atexit.register(stop_diagnostics_logger)


def stop_diagnostics_logger() -> None:
    global _STOPPING
    if _STARTED and not _STOPPING:
        _STOPPING = True
        try:
            _QUEUE.put_nowait(None)
        except queue.Full:
            pass
        if _THREAD and _THREAD.is_alive():
            _THREAD.join(timeout=2.0)


def log_event(stage: str, call_id: str = "", **fields: Any) -> None:
    if not _diagnostics_enabled():
        return
    start_diagnostics_logger()
    event: dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "call_id": call_id or fields.pop("session_id", "") or "unknown",
        "stage": stage,
    }
    for key, value in fields.items():
        if value is None:
            continue
        event[key] = _snippet(value)
    try:
        _QUEUE.put_nowait(event)
    except queue.Full:
        print(
            f"[Diagnostics Drop] call={event['call_id']} stage={stage} reason=queue_full",
            flush=True,
        )
