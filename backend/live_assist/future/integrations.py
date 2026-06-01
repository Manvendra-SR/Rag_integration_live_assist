from __future__ import annotations

from typing import Protocol


class ExternalContextProvider(Protocol):
    """Contract for CRM, calendar, email, memory, and winning-pattern context."""

    name: str

    async def get_context(self, session_id: str, customer_id: str | None = None) -> dict:
        """Return structured context for Live Assist orchestration."""

