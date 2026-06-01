from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from live_assist.core.config import get_settings


class SQLiteContextStore:
    def __init__(self) -> None:
        settings = get_settings()
        path = Path(settings.memory_sqlite_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_summary (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        products TEXT,
                        summary TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, session_id)
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get_previous_session_summary(self, user_id: str, current_session_id: str) -> str:
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                cursor = conn.execute(
                    """
                    SELECT summary, products FROM conversation_summary
                    WHERE user_id = ? AND session_id != ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (user_id, current_session_id),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

        if row:
            return f"Previous session summary: {row[0]}\nProducts discussed: {row[1]}"
        return ""

    def get_current_session_summary(self, user_id: str, session_id: str) -> str:
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                cursor = conn.execute(
                    """
                    SELECT summary FROM conversation_summary
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (user_id, session_id),
                )
                row = cursor.fetchone()
            finally:
                conn.close()
        return row[0] if row else ""

    def get_current_product_context(self, user_id: str, session_id: str) -> str:
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                cursor = conn.execute(
                    """
                    SELECT products FROM conversation_summary
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (user_id, session_id),
                )
                row = cursor.fetchone()
            finally:
                conn.close()
        return row[0] if row and row[0] else ""

    def save_product_context(
        self,
        user_id: str,
        session_id: str,
        products: str,
    ) -> None:
        existing_summary = self.get_current_session_summary(user_id, session_id)
        self.save_summary(user_id, session_id, products, existing_summary)

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        products: str,
        summary: str,
    ) -> None:
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_summary (user_id, session_id, products, summary)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, session_id) DO UPDATE SET
                        products = excluded.products,
                        summary = excluded.summary
                    """,
                    (user_id, session_id, products, summary),
                )
                conn.commit()
            finally:
                conn.close()


context_store = SQLiteContextStore()
