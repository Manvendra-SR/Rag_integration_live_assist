"""
Abstract base class for STT (Speech-to-Text) providers.

Interface contract
------------------
Every concrete provider must implement ``connect()`` as an async context
manager that yields a *session* object (conventionally called ``ws``) with
the following three capabilities, which ``websocket_server.py`` relies on:

1. **Send audio** ::

       await ws.transcribe(audio=<b64_wav_str>, encoding="audio/wav", sample_rate=16000)

   ``audio`` is a base64-encoded mono WAV string produced by ``pcm_to_wav_b64()``.

2. **Receive transcript events** ::

       async for message in ws:
           event_type = getattr(message, "type", None)          # str
           text       = getattr(getattr(message, "data", None), "transcript", "") or ""
           if event_type != "data":
               continue

   The iterator must yield objects where:
   - ``.type`` equals ``"data"`` for transcript events
   - ``.data.transcript`` contains the transcript string

3. **Two independent sessions per call** — ``connect()`` is called twice
   (once for the customer channel, once for the worker channel). Each call
   must open an independent connection.

Adding a new provider
---------------------
1. Create ``providers/asr/<name>.py`` implementing this class.
2. Add the provider name to ``providers/asr/factory.py``.
3. Add the required settings to ``core/config.py``.
4. Set ``ASR_PROVIDER=<name>`` in ``.env``.
   No changes to ``websocket_server.py`` are needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSTTProvider(ABC):
    """Abstract base for all STT provider adapters."""

    @abstractmethod
    def connect(self) -> Any:
        """
        Return an async context manager that yields a streaming session.

        The session object must implement the interface described in the
        module docstring above (``transcribe()`` + async iteration).

        Typical usage in ``websocket_server.py``::

            async with provider.connect() as ws:
                await ws.transcribe(audio=b64_wav, encoding="audio/wav", sample_rate=16000)
                async for message in ws:
                    if getattr(message, "type", None) == "data":
                        text = getattr(getattr(message, "data", None), "transcript", "")
        """
        ...
