from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sarvamai import AsyncSarvamAI

from live_assist.core.config import get_settings


class SarvamStreamingASR:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.sarvam_api_key:
            raise ValueError("PYTHON_WS_SARVAM_API_KEY is required")
        self.client = AsyncSarvamAI(api_subscription_key=self.settings.sarvam_api_key)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator:
        async with self.client.speech_to_text_streaming.connect(
            model=self.settings.sarvam_model,
            mode=self.settings.sarvam_mode,
            language_code=self.settings.sarvam_language_code,
            high_vad_sensitivity=True,
            vad_signals=True,
        ) as ws:
            yield ws

