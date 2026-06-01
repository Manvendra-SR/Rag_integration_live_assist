from __future__ import annotations

import time
from typing import Any

from live_assist.core.models import TranscriptTurn
from live_assist.storage.sqlite import save_transcript_turn


class SessionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._store:
            self._store[session_id] = {
                "history": [],
                "current_batch": [],
                "last_updated": time.time(),
            }
        return self._store[session_id]

    def add_turn(self, turn: TranscriptTurn) -> None:
        session = self.get_session(turn.session_id)
        payload = turn.model_dump(mode="json")
        session["history"].append(payload)
        session["current_batch"].append(payload)
        session["last_updated"] = time.time()

    def add_payload(self, session_id: str, payload: dict[str, Any]) -> None:
        session = self.get_session(session_id)
        session["history"].append(payload)
        session["current_batch"].append(payload)
        session["last_updated"] = time.time()

    def get_current_batch(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.get_session(session_id)["current_batch"])

    def clear_current_batch(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id]["current_batch"] = []
            self._store[session_id]["last_updated"] = time.time()

    def clear_all(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id]["history"] = []
            self._store[session_id]["current_batch"] = []
            self._store[session_id]["last_updated"] = time.time()


session_store = SessionStore()


async def persist_turn(turn: TranscriptTurn) -> None:
    await save_transcript_turn(turn)
    session_store.add_turn(turn)

