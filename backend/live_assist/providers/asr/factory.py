"""
STT provider factory.

Usage (in ``websocket_server.py``)::

    from live_assist.providers.asr.factory import create_asr_provider

    asr_provider = create_asr_provider(settings)

    async with asr_provider.connect() as ws:
        ...

Switching providers requires only a single ``.env`` change::

    ASR_PROVIDER=triton_http   # default for E2E Triton HTTPS endpoint
    ASR_PROVIDER=triton_grpc   # when E2E exposes external gRPC host:port
    ASR_PROVIDER=sarvam        # Sarvam Streaming ASR
    ASR_PROVIDER=whisper       # E2E Whisper WebSocket (legacy placeholder)

No other file needs to be modified.
"""
from __future__ import annotations

import logging

from live_assist.core.config import Settings
from live_assist.providers.asr.base import BaseSTTProvider

logger = logging.getLogger(__name__)

_SUPPORTED = ("sarvam", "whisper", "triton_http", "triton_grpc")


def create_asr_provider(settings: Settings) -> BaseSTTProvider:
    """
    Instantiate the correct STT provider based on ``ASR_PROVIDER`` in ``.env``.

    Parameters
    ----------
    settings:
        The application ``Settings`` object returned by ``get_settings()``.

    Returns
    -------
    BaseSTTProvider
        A concrete provider instance whose ``connect()`` method is an async
        context manager yielding a session compatible with ``websocket_server.py``.

    Raises
    ------
    ValueError
        If ``ASR_PROVIDER`` is set to an unsupported value.
    """
    provider = (settings.asr_provider or "sarvam").strip().lower()
    logger.info("ASR provider selected: %r", provider)

    if provider == "sarvam":
        from live_assist.providers.asr.sarvam import SarvamStreamingASR
        return SarvamStreamingASR()

    if provider == "whisper":
        from live_assist.providers.asr.whisper import WhisperStreamingASR
        return WhisperStreamingASR(settings)

    if provider == "triton_http":
        from live_assist.providers.asr.triton_http import TritonHttpASRProvider
        return TritonHttpASRProvider(settings)

    if provider == "triton_grpc":
        from live_assist.providers.asr.triton_grpc import TritonGrpcASRProvider
        return TritonGrpcASRProvider(settings)

    raise ValueError(
        f"Unsupported ASR_PROVIDER={provider!r}. "
        f"Supported values: {', '.join(repr(s) for s in _SUPPORTED)}."
    )
