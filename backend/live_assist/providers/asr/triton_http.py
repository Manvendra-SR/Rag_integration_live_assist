"""
Triton HTTP ASR provider for E2E Whisper endpoint.

Uses the E2E-documented ``tritonclient.http`` approach:
  - ``InferenceServerClient`` with SSL and auth header per-request
  - ``InferInput`` with ``set_data_from_numpy()`` — raw bytes, not base64
  - ``client.infer()`` for request/response inference

Architecture
------------
The E2E Triton endpoint exposes a Triton HTTP inference API, accessed via the
``tritonclient[http]`` library exactly as shown in the E2E documentation:

    endpoint_url = 'infer.e2enetworks.net/project/<p-id>/endpoint/<is-id>/'
    headers = {'Authorization': 'Bearer <token>'}
    client = httpclient.InferenceServerClient(url=endpoint_url, ssl=True, ...)

``websocket_server.py`` expects each ASR "session" to look like a WebSocket:

    1. ``await ws.transcribe(audio, encoding, sample_rate)``  — push audio chunk
    2. ``async for message in ws:``                          — receive transcripts
       • ``message.type == "data"``
       • ``message.data.transcript`` carries the text

This file adapts the Triton HTTP request/response into that interface:

* ``transcribe()`` decodes the base64 WAV string back to raw bytes, wraps them
  in a numpy object array, and calls ``client.infer()`` in a thread-pool executor
  (the tritonclient HTTP library is synchronous).
* The response transcript is placed on an internal asyncio queue.
* The ``async for`` iterator drains that queue, yielding ``_TranscriptEvent``
  objects — the same shape that ``websocket_server.py`` already knows.

Configuration (``.env``)
------------------------
``ASR_PROVIDER=triton_http``       – activate this provider
``TRITON_HTTP_URL``                – E2E endpoint base URL (with https://)
    e.g. https://infer.e2enetworks.net/project/p-17206/endpoint/is-11234
``TRITON_MODEL_NAME``              – model name (default: whisper)
``TRITON_AUTH_TOKEN``              – optional; falls back to STT_API_KEY
``TRITON_LANGUAGE``                – language hint (default: english)
``TRITON_TASK``                    – transcribe | translate (default: transcribe)
``TRITON_HTTP_TIMEOUT``            – per-request timeout in seconds (default: 30)

Dependencies
------------
    pip install "tritonclient[http]" gevent numpy
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from live_assist.providers.asr.base import BaseSTTProvider

logger = logging.getLogger(__name__)

# Sentinel placed on the queue to signal end-of-session
_STOP = object()


# ── Normalised event types (same shape as whisper.py / sarvam.py) ─────────

@dataclass(frozen=True)
class _TranscriptData:
    """Carries the transcript string for a single event."""
    transcript: str


@dataclass(frozen=True)
class _TranscriptEvent:
    """
    Normalised event yielded by the async iterator.

    ``websocket_server.py`` reads:
        event_type = getattr(message, "type", None)       # must be "data"
        text       = getattr(getattr(message, "data", None), "transcript", "")
    """
    type: str           # "data" for transcripts, anything else is ignored
    data: _TranscriptData


# ── Session ────────────────────────────────────────────────────────────────


class TritonHttpSession:
    """
    Wraps the Triton HTTP inference client as an ASR session.

    Uses ``tritonclient.http`` (the E2E-documented approach) with numpy byte
    arrays.  The synchronous ``client.infer()`` runs in a thread-pool executor.

    One instance is created per channel (customer / worker) per call.
    ``transcribe()`` fires an infer call and places the result on ``_queue``.
    The ``async for`` iterator drains ``_queue`` and yields events.
    """

    def __init__(
        self,
        client: Any,        # tritonclient.http.InferenceServerClient
        auth_headers: dict[str, str],
        model_name: str,
        language: str,
        task: str,
        timeout: float,
    ) -> None:
        self._client = client
        self._auth_headers = auth_headers
        self._model_name = model_name
        self._language = language
        self._task = task
        self._timeout = timeout
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)

    # ── Send side ──────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio: str,           # base64-encoded WAV produced by pcm_to_wav_b64()
        encoding: str = "audio/wav",
        sample_rate: int = 16000,
    ) -> None:
        """
        Run a Triton HTTP infer call in a thread pool and queue the transcript.

        ``audio`` is the base64-encoded WAV string from ``pcm_to_wav_b64()``.
        We decode it back to raw bytes here — Triton expects the actual WAV
        bytes in a numpy object array, not base64.
        """
        wav_bytes = base64.b64decode(audio)
        try:
            transcript = await asyncio.get_event_loop().run_in_executor(
                None,
                self._sync_infer,
                wav_bytes,
            )
        except Exception as exc:
            logger.error(
                "TritonHttpSession: infer failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            return

        if transcript:
            logger.debug("TritonHttpSession: transcript=%r", transcript[:120])
            event = _TranscriptEvent(
                type="data",
                data=_TranscriptData(transcript=transcript),
            )
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("TritonHttpSession: queue full — dropping transcript")

    def _sync_infer(self, wav_bytes: bytes) -> str:
        """
        Blocking Triton HTTP inference call using tritonclient.http.

        This is the exact pattern from the E2E documentation, adapted for the
        Whisper model's AUDIO_BYTES + PARAMS input schema.

        Runs in a thread-pool executor to avoid blocking the asyncio event loop.
        """
        try:
            import numpy as np
            import tritonclient.http as httpclient
        except ImportError as exc:
            raise RuntimeError(
                "tritonclient[http] and numpy are required for ASR_PROVIDER=triton_http.\n"
                "Install with:  pip install 'tritonclient[http]' gevent numpy"
            ) from exc

        params_str = json.dumps({
            "language": self._language,
            "task": self._task,
            "return_timestamps": False,
            "suffix": ".wav",
        })

        # AUDIO_BYTES input: numpy object array containing raw WAV bytes
        audio_input = httpclient.InferInput("AUDIO_BYTES", [1], "BYTES")
        audio_np = np.array([wav_bytes], dtype=object)
        audio_input.set_data_from_numpy(audio_np)

        # PARAMS input: numpy object array containing the JSON params as bytes
        params_input = httpclient.InferInput("PARAMS", [1], "BYTES")
        params_np = np.array([params_str.encode("utf-8")], dtype=object)
        params_input.set_data_from_numpy(params_np)

        outputs = [
            httpclient.InferRequestedOutput("TRANSCRIPT"),
            httpclient.InferRequestedOutput("JSON"),
        ]

        logger.debug(
            "TritonHttpSession: infer model=%s language=%s task=%s wav_bytes=%d",
            self._model_name,
            self._language,
            self._task,
            len(wav_bytes),
        )

        result = self._client.infer(
            model_name=self._model_name,
            inputs=[audio_input, params_input],
            outputs=outputs,
            headers=self._auth_headers,
            timeout=int(self._timeout * 1000000),
        )

        return self._parse_result(result)

    @staticmethod
    def _parse_result(result: Any) -> str:
        """Extract the transcript string from a Triton HTTP InferResult."""
        try:
            transcript_arr = result.as_numpy("TRANSCRIPT")
            if transcript_arr is not None and len(transcript_arr) > 0:
                raw = transcript_arr[0]
                text = (
                    raw.decode("utf-8")
                    if isinstance(raw, (bytes, bytearray))
                    else str(raw)
                ).strip()
                if text:
                    logger.debug("TritonHttp parsed TRANSCRIPT: %r", text)
                    return text
        except Exception as e:
            logger.error("Error parsing TRANSCRIPT: %s", e)

        # Fallback: parse the JSON output
        try:
            json_arr = result.as_numpy("JSON")
            if json_arr is not None and len(json_arr) > 0:
                raw = json_arr[0]
                inner_str = (
                    raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                )
                inner = json.loads(inner_str)
                text = (inner.get("text") or inner.get("transcript") or "").strip()
                if text:
                    logger.debug("TritonHttp parsed JSON: %r", text)
                    return text
        except Exception as e:
            logger.error("Error parsing JSON: %s", e)

        logger.warning("TritonHttp returning empty string, result: %s", result.get_response())
        return ""

    # ── Receive side ────────────────────────────────────────────────────────

    def __aiter__(self) -> AsyncIterator[_TranscriptEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[_TranscriptEvent]:
        """
        Drain the internal transcript queue.

        Bridges the HTTP request/response model into the ``async for`` iterator
        interface expected by ``websocket_server.py``.  Exits when the sentinel
        ``_STOP`` is placed on the queue by ``TritonHttpASRProvider.connect()``.
        """
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            yield item

    def _close(self) -> None:
        """Signal the async iterator to stop."""
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            pass


# ── Provider ───────────────────────────────────────────────────────────────


class TritonHttpASRProvider(BaseSTTProvider):
    """
    E2E Triton HTTP inference ASR provider.

    Uses ``tritonclient.http`` exactly as documented by E2E:
    - SSL enabled
    - Auth token passed per infer() call via headers
    - Numpy object arrays for BYTES inputs (raw bytes, not base64)

    Set ``ASR_PROVIDER=triton_http`` in ``.env`` to activate.

    ``.env`` variables consumed
    ---------------------------
    ``TRITON_HTTP_URL``      – E2E endpoint HTTPS URL (required)
    ``TRITON_MODEL_NAME``    – model name (default: whisper)
    ``TRITON_AUTH_TOKEN``    – Bearer token (optional; falls back to STT_API_KEY)
    ``TRITON_LANGUAGE``      – language hint (default: english)
    ``TRITON_TASK``          – transcribe | translate (default: transcribe)
    ``TRITON_HTTP_TIMEOUT``  – per-request timeout seconds (default: 30)
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._validate_config()

    # ── Config ─────────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        if not self._triton_http_url:
            raise ValueError(
                "TRITON_HTTP_URL must be set in .env when ASR_PROVIDER=triton_http.\n"
                "Example: TRITON_HTTP_URL=https://infer.e2enetworks.net/project/p-17206/endpoint/is-11234"
            )

    @property
    def _triton_http_url(self) -> str:
        return (getattr(self._settings, "triton_http_url", "") or "").rstrip("/")

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
            return float(getattr(self._settings, "triton_http_timeout", 30) or 30)
        except (TypeError, ValueError):
            return 30.0

    def _build_endpoint_url(self) -> str:
        """
        Build the endpoint URL in the format tritonclient.http expects.

        The E2E documentation and tritonclient.http client require the URL
        WITHOUT the ``https://`` scheme prefix — the client handles SSL via its
        ``ssl=True`` parameter and ``ssl_context_factory``.

        Example:
            TRITON_HTTP_URL = https://infer.e2enetworks.net/project/p-17206/endpoint/is-11234
            tritonclient url = infer.e2enetworks.net/project/p-17206/endpoint/is-11234
        """
        url = self._triton_http_url
        for prefix in ("https://", "http://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
        return url

    def _build_auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = self._auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _build_client(self) -> Any:
        """
        Instantiate a tritonclient.http.InferenceServerClient with SSL.
        Exactly mirrors the E2E documentation example.
        """
        try:
            import tritonclient.http as httpclient
            import gevent.ssl as ssl
        except ImportError as exc:
            raise RuntimeError(
                "tritonclient[http] and gevent are required for ASR_PROVIDER=triton_http.\n"
                "Install with:  pip install 'tritonclient[http]' gevent"
            ) from exc

        endpoint_url = self._build_endpoint_url()
        logger.info(
            "TritonHttpASRProvider: creating client — endpoint=%s model=%s "
            "language=%s task=%s timeout=%ss",
            endpoint_url,
            self._model_name,
            self._language,
            self._task,
            self._timeout,
        )

        client = httpclient.InferenceServerClient(
            url=endpoint_url,
            verbose=False,
            ssl=True,
            ssl_options={},
            insecure=False,
            ssl_context_factory=ssl.create_default_context,
            network_timeout=self._timeout,
        )
        return client

    # ── Health check ───────────────────────────────────────────────────────

    def check_model_ready(self, client: Any) -> bool:
        """
        Check model readiness via tritonclient.http.
        Runs synchronously (call inside run_in_executor if needed).
        """
        try:
            auth_headers = self._build_auth_headers()
            ready = client.is_model_ready(self._model_name, headers=auth_headers)
            logger.info(
                "TritonHttpASRProvider: model ready=%s (model=%s)",
                ready,
                self._model_name,
            )
            return bool(ready)
        except Exception as exc:
            logger.warning(
                "TritonHttpASRProvider: model ready check failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            return False

    # ── Connect (async context manager) ───────────────────────────────────

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[TritonHttpSession]:
        """
        Open a Triton HTTP session.

        Creates a ``tritonclient.http.InferenceServerClient`` (synchronous,
        but cheap to create) and yields a ``TritonHttpSession`` that satisfies
        the ``websocket_server.py`` interface.

        Called twice per browser connection — once for customer, once for worker.
        """
        loop = asyncio.get_event_loop()

        # Build the client in the executor (imports gevent — can be slow first time)
        client = await loop.run_in_executor(None, self._build_client)

        # Optional health check — warn but do not abort if not ready
        ready = await loop.run_in_executor(None, self.check_model_ready, client)
        if not ready:
            logger.warning(
                "TritonHttpASRProvider: model %r may not be ready — "
                "proceeding anyway",
                self._model_name,
            )

        session = TritonHttpSession(
            client=client,
            auth_headers=self._build_auth_headers(),
            model_name=self._model_name,
            language=self._language,
            task=self._task,
            timeout=self._timeout,
        )
        try:
            yield session
        finally:
            session._close()
            try:
                client.close()
            except Exception:
                pass
            logger.info("TritonHttpASRProvider: session closed")


# ── Helpers ────────────────────────────────────────────────────────────────

_BCP47_TO_ENGLISH: dict[str, str] = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "marathi",
    "gu": "gujarati",
    "pa": "punjabi",
    "bn": "bengali",
    "ur": "urdu",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "ar": "arabic",
    "ru": "russian",
    "pt": "portuguese",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "tr": "turkish",
    "vi": "vietnamese",
    "id": "indonesian",
    "th": "thai",
}


def _bcp47_to_english_name(code: str) -> str:
    """Convert a BCP-47 language code to the English name Whisper expects."""
    code = (code or "").lower().split("-")[0].split("_")[0]
    if len(code) > 3 and code.isalpha():
        return code
    return _BCP47_TO_ENGLISH.get(code, code)
