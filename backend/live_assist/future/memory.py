from __future__ import annotations

from typing import Protocol


class MemoryProvider(Protocol):
    """Future long-term memory boundary."""

    async def read(self, customer_id: str) -> dict:
        """Read customer/account memory."""

    async def write(self, customer_id: str, memory_update: dict) -> None:
        """Persist memory after post-meeting processing or explicit updates."""

