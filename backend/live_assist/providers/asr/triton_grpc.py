"""
Triton gRPC ASR provider (optional, for when E2E exposes an external gRPC host:port).

Architecture
------------
Only activate this provider if E2E provides a publicly reachable gRPC endpoint.
Set ``TRITON_GRPC_URL=<host>:9000`` (no scheme prefix) in ``.env``.

The Whisper model served by Triton is a standard request/response model
(not decoupled), so ``client.infer()`` is used — NOT the streaming variant.

Provider interface
------------------
``connect()`` yields a ``TritonGrpcSession`` that satisfies the
``websocket_server.py`` contract:

    1. ``await ws.transcribe(audio, encoding, sample_rate)``  — push audio chunk
    2. ``async for message in ws:``                          — receive transcripts

Configuration (``.env``)
------------------------
``ASR_PROVIDER=triton_grpc``    – activate this provider
``TRITON_GRPC_URL``             – host:port, e.g. ``my-triton-host.example.com:9000``
                                  Do NOT include http:// or grpc:// prefix.
``TRITON_MODEL_NAME``           – model name (default: whisper)
``TRITON_AUTH_TOKEN``           – optional Bearer / metadata token
``TRITON_LANGUAGE``             – language hint (default: english)
``TRITON_TASK``                 – transcribe | translate (default: transcribe)
``TRITON_GRPC_TIMEOUT``         – per-call timeout in seconds (default: 30)

Dependencies
------------
    pip install tritonclient[grpc]
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from live_assist.providers.asr.base import BaseSTTProvider

logger = logging.getLogger(__name__)

_STOP = object()


# ── Normalised event types ─────────────────────────────────────────────────

@dataclass(frozen=True)
class _TranscriptData:
    transcript: str


@dataclass(frozen=True)
class _TranscriptEvent:
    type: str           # "data" for transcripts
    data: _TranscriptData


# ── Session ────────────────────────────────────────────────────────────────


class TritonGrpcSession:
    """
    Wraps a tritonclient.grpc inference call as an ASR session.

    ``transcribe()`` runs the blocking ``client.infer()`` in a thread pool
    executor so as not to block the asyncio event loop.  Results are placed
    on an internal queue; the async iterator drains the queue.
    """

    def __init__(
        self,
        client: Any,            # tritonclient.grpc.InferenceServerClient
        model_name: str,
        language: str,
        task: str,
        timeout: float,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._language = language
        self._task = task
        self._timeout = timeout
        self._loop = loop
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)

    # ── Send side ──────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio: str,           # base64-encoded WAV
        encoding: str = "audio/wav",
        sample_rate: int = 16000,
    ) -> None:
        """
        Run a synchronous Triton gRPC infer call in a thread pool executor
        and queue the transcript result.
        """
        try:
            transcript = await asyncio.get_event_loop().run_in_executor(
                None,
                self._sync_infer,
                audio,
            )
        except Exception as exc:
            logger.error(
                "TritonGrpcSession: infer failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            return

        if transcript:
            logger.debug("TritonGrpcSession: transcript=%r", transcript[:120])
            event = _TranscriptEvent(
                type="data",
                data=_TranscriptData(transcript=transcript),
            )
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("TritonGrpcSession: queue full — dropping transcript")

    def _sync_infer(self, audio_b64: str) -> str:
        """
        Blocking Triton gRPC inference call.  Runs inside a thread pool.

        AUDIO_BYTES expects a numpy object-array containing the raw WAV bytes.
        We decode the base64 string back to bytes for this purpose.
        """
        try:
            import numpy as np
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise RuntimeError(
                "tritonclient[grpc] and numpy are required for ASR_PROVIDER=triton_grpc.\n"
                "Install with:  pip install 'tritonclient[grpc]' numpy"
            ) from exc

        import base64

        # Decode the base64 WAV string back to raw bytes
        audio_bytes = base64.b64decode(audio_b64)

        params_str = json.dumps({
            "language": self._language,
            "task": self._task,
            "return_timestamps": False,
            "suffix": ".wav",
        })

        # AUDIO_BYTES input: numpy object array containing the raw WAV bytes
        audio_input = grpcclient.InferInput("AUDIO_BYTES", [1], "BYTES")
        audio_np = np.array([audio_bytes], dtype=object)
        audio_input.set_data_from_numpy(audio_np)

        # PARAMS input: numpy object array containing the params JSON string
        params_input = grpcclient.InferInput("PARAMS", [1], "BYTES")
        params_np = np.array([params_str.encode("utf-8")], dtype=object)
        params_input.set_data_from_numpy(params_np)

        outputs = [
            grpcclient.InferRequestedOutput("TRANSCRIPT"),
            grpcclient.InferRequestedOutput("JSON"),
        ]

        result = self._client.infer(
            model_name=self._model_name,
            inputs=[audio_input, params_input],
            outputs=outputs,
            client_timeout=self._timeout,
        )

        return self._parse_result(result)

    @staticmethod
    def _parse_result(result: Any) -> str:
        """Extract the transcript string from a Triton gRPC InferResult."""
        try:
            transcript_arr = result.as_numpy("TRANSCRIPT")
            if transcript_arr is not None and len(transcript_arr) > 0:
                raw = transcript_arr[0]
                text = (raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()
                if text:
                    return text
        except Exception:
            pass

        # Fallback: try the JSON output
        try:
            json_arr = result.as_numpy("JSON")
            if json_arr is not None and len(json_arr) > 0:
                raw = json_arr[0]
                inner_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                inner = json.loads(inner_str)
                text = (inner.get("text") or inner.get("transcript") or "").strip()
                if text:
                    return text
        except Exception:
            pass

        return ""

    # ── Receive side ────────────────────────────────────────────────────────

    def __aiter__(self) -> AsyncIterator[_TranscriptEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[_TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            yield item

    def _close(self) -> None:
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            pass


# ── Provider ───────────────────────────────────────────────────────────────


class TritonGrpcASRProvider(BaseSTTProvider):
    """
    E2E Triton gRPC inference ASR provider.

    Only use this when E2E provides a real, externally reachable gRPC host:port.
    Set ``ASR_PROVIDER=triton_grpc`` and ``TRITON_GRPC_URL=<host>:9000`` in
    ``.env``.

    ``.env`` variables consumed
    ---------------------------
    ``TRITON_GRPC_URL``      – host:port (required, no scheme)
    ``TRITON_MODEL_NAME``    – model name (default: whisper)
    ``TRITON_AUTH_TOKEN``    – optional token
    ``TRITON_LANGUAGE``      – language hint (default: english)
    ``TRITON_TASK``          – transcribe | translate (default: transcribe)
    ``TRITON_GRPC_TIMEOUT``  – per-call timeout seconds (default: 30)
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._validate_config()

    def _validate_config(self) -> None:
        if not self._grpc_url:
            raise ValueError(
                "TRITON_GRPC_URL must be set in .env when ASR_PROVIDER=triton_grpc.\n"
                "Example: TRITON_GRPC_URL=my-triton-host.example.com:9000\n"
                "Do NOT include http:// or grpc:// — host:port only."
            )
        if "://" in self._grpc_url:
            raise ValueError(
                f"TRITON_GRPC_URL must be host:port only (no scheme). Got: {self._grpc_url!r}"
            )

    @property
    def _grpc_url(self) -> str:
        return (getattr(self._settings, "triton_grpc_url", "") or "").strip()

    @property
    def _model_name(self) -> str:
        return getattr(self._settings, "triton_model_name", "") or "whisper"

    @property
    def _auth_token(self) -> str:
        token = getattr(self._settings, "triton_auth_token", "") or ""
        if not token:
            token = getattr(self._settings, "stt_api_key", "") or ""
        return token

    @property
    def _language(self) -> str:
        from live_assist.providers.asr.triton_http import _bcp47_to_english_name
        lang = getattr(self._settings, "triton_language", "") or ""
        if not lang:
            raw = (getattr(self._settings, "stt_language", "") or "en").lower()
            lang = _bcp47_to_english_name(raw)
        return lang or "english"

    @property
    def _task(self) -> str:
        task = getattr(self._settings, "triton_task", "") or ""
        if not task:
            task = getattr(self._settings, "stt_task", "") or "transcribe"
        return task

    @property
    def _timeout(self) -> float:
        try:
            return float(getattr(self._settings, "triton_grpc_timeout", 30) or 30)
        except (TypeError, ValueError):
            return 30.0

    def _build_client(self) -> Any:
        """Instantiate a tritonclient gRPC client (validates import at creation time)."""
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise RuntimeError(
                "tritonclient[grpc] is required for ASR_PROVIDER=triton_grpc.\n"
                "Install with:  pip install 'tritonclient[grpc]'"
            ) from exc

        metadata: list[tuple[str, str]] = []
        token = self._auth_token
        if token:
            metadata.append(("authorization", f"Bearer {token}"))

        client = grpcclient.InferenceServerClient(
            url=self._grpc_url,
            verbose=False,
        )
        logger.info(
            "TritonGrpcASRProvider: client created — url=%s model=%s",
            self._grpc_url,
            self._model_name,
        )
        return client

    def check_model_ready(self, client: Any) -> bool:
        """Synchronous model readiness check via gRPC."""
        try:
            ready = client.is_model_ready(self._model_name)
            logger.info(
                "TritonGrpcASRProvider: model ready=%s (model=%s url=%s)",
                ready,
                self._model_name,
                self._grpc_url,
            )
            return bool(ready)
        except Exception as exc:
            logger.warning(
                "TritonGrpcASRProvider: model ready check failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            return False

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[TritonGrpcSession]:
        """
        Open a Triton gRPC session.

        The gRPC client is instantiated synchronously (it is cheap and
        non-blocking for a channel object) and the readiness check is run
        in a thread pool executor to avoid blocking the event loop.

        Yields a ``TritonGrpcSession`` compatible with ``websocket_server.py``.
        """
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, self._build_client)

        # Optional health check — warn but do not abort if it fails
        ready = await loop.run_in_executor(None, self.check_model_ready, client)
        if not ready:
            logger.warning(
                "TritonGrpcASRProvider: model %r not ready at %s — "
                "proceeding anyway (may fail on first infer call)",
                self._model_name,
                self._grpc_url,
            )

        session = TritonGrpcSession(
            client=client,
            model_name=self._model_name,
            language=self._language,
            task=self._task,
            timeout=self._timeout,
            loop=loop,
        )
        try:
            yield session
        finally:
            session._close()
            try:
                client.close()
            except Exception:
                pass
            logger.info("TritonGrpcASRProvider: session closed")
