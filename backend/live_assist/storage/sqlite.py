from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import aiosqlite

from live_assist.core.config import get_settings
from live_assist.core.models import TranscriptTurn


def _db_path(db_url: str | None = None) -> str:
    settings = get_settings()
    raw_path = db_url or settings.live_feedback_sqlite_path
    if raw_path.startswith("sqlite:///"):
        raw_path = raw_path.replace("sqlite:///", "", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def init_db(db_url: str | None = None) -> str:
    db_path = _db_path(db_url)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                source TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                translated_text TEXT,
                triggered_live_assist INTEGER NOT NULL,
                workflow_target TEXT NOT NULL,
                product TEXT,
                ts REAL NOT NULL,
                metadata_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


async def save_transcript_turn(turn: TranscriptTurn, db_url: str | None = None) -> None:
    db_path = _db_path(db_url)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO calls(call_id, created_at) VALUES (?, ?)",
            (turn.session_id, turn.timestamp),
        )
        await db.execute(
            """
            INSERT INTO transcript_turns(
                session_id, speaker, source, raw_text, translated_text,
                triggered_live_assist, workflow_target, product, ts, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.session_id,
                turn.speaker.value,
                turn.source,
                turn.raw_text,
                turn.translated_text,
                1 if turn.triggered_live_assist else 0,
                turn.workflow_target,
                turn.product,
                turn.timestamp,
                json.dumps(turn.metadata, ensure_ascii=True),
            ),
        )
        await db.commit()


def get_recent_transcript_turns(
    session_id: str,
    limit: int = 10,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    db_path = _db_path(db_url)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT speaker, raw_text, translated_text, source, triggered_live_assist,
                   workflow_target, product, ts, metadata_json
            FROM transcript_turns
            WHERE session_id = ?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    turns = []
    for row in reversed(rows):
        metadata = {}
        if row[8]:
            try:
                metadata = json.loads(row[8])
            except json.JSONDecodeError:
                metadata = {"raw": row[8]}
        turns.append(
            {
                "speaker": row[0],
                "text": row[2] or row[1],
                "raw_text": row[1],
                "translated_text": row[2],
                "source": row[3],
                "triggered_live_assist": bool(row[4]),
                "workflow_target": row[5],
                "product": row[6],
                "ts": row[7],
                "metadata": metadata,
            }
        )
    return turns


async def save_audit_result(
    call_id: str,
    result: dict[str, Any],
    db_url: str | None = None,
) -> None:
    db_path = _db_path(db_url)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO audits(call_id, result_json, created_at) VALUES (?, ?, ?)",
            (call_id, json.dumps(result, ensure_ascii=True), time.time()),
        )
        await db.commit()

