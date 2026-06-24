"""
E2E Whisper Large-v3 real-time streaming ASR adapter.

Architecture
------------
This file is the **only** place that contains E2E-specific protocol details.
``websocket_server.py`` is completely unaware of this file's internals —
it only sees the ``BaseSTTProvider`` interface.

When E2E endpoint documentation arrives, update the four clearly-marked
TODO sections in this file. No other file needs to change.

Provider interface (what ``websocket_server.py`` expects)
---------------------------------------------------------
``WhisperStreamingASR.connect()`` must be an async context manager that
yields a session object (``WhisperStreamingSession``) satisfying:

1. ``await ws.transcribe(audio, encoding, sample_rate)``  — push audio chunk
2. ``async for message in ws:``                          — receive transcript events
   - ``message.type == "data"``   for transcript events
   - ``message.data.transcript``  contains the text string

Configuration (via .env — no code changes needed)
-------------------------------------------------
``STT_BASE_URL``  – Streaming endpoint URL (ws:// or wss://)
``STT_API_KEY``   – API key / bearer token for authentication
``STT_MODEL``     – Model identifier forwarded to the endpoint
``STT_TASK``      – "transcribe" (default) or "translate" (output in English)
``STT_LANGUAGE``  – BCP-47 language hint, e.g. "hi"; blank = auto-detect

TODO sections (update when E2E documentation arrives)
-----------------------------------------------------
[TODO-URL]      ``WhisperStreamingASR._build_websocket_url()``
[TODO-AUTH]     ``WhisperStreamingASR._build_auth_headers()``
[TODO-HANDSHAKE] ``WhisperStreamingSession._handshake()``
[TODO-SEND]     ``WhisperStreamingSession._build_audio_payload()``
[TODO-RECV]     ``WhisperStreamingSession._parse_event()``
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets
from websockets.legacy.client import WebSocketClientProtocol

from live_assist.providers.asr.base import BaseSTTProvider

logger = logging.getLogger(__name__)


# ── Normalised event types (mirrors Sarvam SDK event shape) ───────────────
#
# ``websocket_server.py`` reads:
#   event_type = getattr(message, "type", None)
#   text       = getattr(getattr(message, "data", None), "transcript", "") or ""
#
# These two dataclasses produce exactly that shape so the server loop
# requires zero changes.


@dataclass(frozen=True)
class _TranscriptData:
    """Holds the transcript string for a single event."""

    transcript: str


@dataclass(frozen=True)
class _TranscriptEvent:
    """
    Normalised transcript event object yielded by the async iterator.

    Fields
    ------
    type : str
        ``"data"`` for transcript events.  Any other value is silently
        skipped by the ``send_text()`` loop in ``websocket_server.py``.
    data : _TranscriptData
        Wrapper carrying the transcript string.
    """

    type: str            # "data" for transcripts, anything else is ignored
    data: _TranscriptData


# ── Streaming session ──────────────────────────────────────────────────────


class WhisperStreamingSession:
    """
    Wraps a live WebSocket connection to the E2E Whisper endpoint.

    Exposes the same interface as a Sarvam WebSocket session so that
    ``websocket_server.py`` works without modification.

    Two instances of this class run concurrently per call
    (one for the customer channel, one for the worker channel).

    E2E-specific protocol details are isolated in four methods:
    - ``_handshake()``          — [TODO-HANDSHAKE]
    - ``_build_audio_payload()`` — [TODO-SEND]
    - ``_parse_event()``        — [TODO-RECV]
    """

    def __init__(
        self,
        ws: WebSocketClientProtocol,
        model: str,
        task: str,
        language: str,
    ) -> None:
        self._ws = ws
        self._model = model
        self._task = task       # "transcribe" | "translate"
        self._language = language

    # ── Connection initialization ──────────────────────────────────────────

    async def _handshake(self) -> None:
        """
        [TODO-HANDSHAKE] Perform any session-initialization exchange required
        by the E2E endpoint immediately after the WebSocket opens.

        Examples of what may be needed (depends on E2E documentation):
        - Send a session configuration frame (model name, task, language, etc.)
        - Receive and validate a session-accepted acknowledgement
        - Exchange capability negotiation messages
        - Send a "start streaming" control message

        If the endpoint requires no handshake (audio can be sent immediately
        after connection), leave this method as a no-op.

        The method is called once by ``WhisperStreamingASR.connect()`` before
        yielding the session to the caller.
        """
        # TODO: implement handshake when E2E endpoint documentation is available
        logger.debug(
            "WhisperStreamingSession: _handshake() called — "
            "no handshake implemented yet (TODO-HANDSHAKE)"
        )

    # ── Send side ──────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio: str,          # base64-encoded WAV produced by pcm_to_wav_b64()
        encoding: str = "audio/wav",
        sample_rate: int = 16000,
    ) -> None:
        """
        Push one audio chunk to the E2E Whisper endpoint.

        Called by ``send_audio_to_stt()`` in ``websocket_server.py`` for
        every PCM chunk that passes the energy gate.

        Parameters
        ----------
        audio :
            Base64-encoded mono WAV string (16-bit PCM, ``sample_rate`` Hz).
            Produced by ``live_assist.audio.pcm.pcm_to_wav_b64()``.
        encoding :
            Always ``"audio/wav"`` in the current implementation.
        sample_rate :
            Typically 16000 Hz (controlled by ``PYTHON_WS_SAMPLE_RATE``).
        """
        payload = self._build_audio_payload(audio, encoding, sample_rate)
        await self._ws.send(payload)

    def _build_audio_payload(
        self,
        audio_b64: str,
        encoding: str,
        sample_rate: int,
    ) -> str | bytes:
        """
        [TODO-SEND] Construct the wire frame sent to the E2E endpoint.

        Update this method when E2E documentation is available.  No other
        file needs to change.

        The method should return either:
        - A JSON string (``str``) for text-frame protocols.
        - Raw bytes (``bytes``) for binary-frame protocols.

        Known unknowns to resolve from E2E documentation:
        □ JSON envelope schema (field names, nesting)
        □ Whether audio should be sent as base64 string or raw bytes
        □ Whether model/task/language are included per-frame or only in handshake
        □ Whether the endpoint expects framed JSON or raw binary audio

        Current placeholder — sends a JSON envelope:
        {
            "audio": "<base64-wav>",
            "encoding": "audio/wav",
            "sample_rate": 16000
        }
        This is a reasonable guess for an OpenAI-compatible streaming endpoint
        but MUST be verified against actual E2E documentation before use.
        """
        # TODO: replace with the actual frame format when E2E docs arrive
        frame: dict[str, Any] = {
            "audio": audio_b64,
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        if self._model:
            frame["model"] = self._model
        if self._task:
            frame["task"] = self._task
        if self._language:
            frame["language"] = self._language
        return json.dumps(frame)

    # ── Receive side ────────────────────────────────────────────────────────

    def __aiter__(self) -> AsyncIterator[_TranscriptEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[_TranscriptEvent]:
        """
        Drive the ``async for message in ws:`` loop in ``send_text()``.

        Reads raw frames from the WebSocket, passes each through
        ``_parse_event()``, and yields only non-None results.
        Non-transcript control messages (acks, errors, etc.) return None
        and are silently dropped here.
        """
        async for raw in self._ws:
            event = self._parse_event(raw)
            if event is not None:
                yield event

    def _parse_event(self, raw: str | bytes) -> _TranscriptEvent | None:
        """
        [TODO-RECV] Convert a raw WebSocket frame from the E2E endpoint into
        a normalised ``_TranscriptEvent``.

        Update this method when E2E documentation is available. No other
        file needs to change.

        The method must return:
        - ``_TranscriptEvent(type="data", data=_TranscriptData(transcript=<text>))``
          when ``raw`` contains a transcript (partial or final).
        - ``None`` for any non-transcript frame (session acks, errors, heartbeats,
          end-of-stream signals, etc.).

        Known unknowns to resolve from E2E documentation:
        □ JSON schema of transcript response frames
        □ Field names that carry the transcript text
        □ How partial vs final transcripts are distinguished (if at all)
        □ Whether binary frames are used for any response type
        □ What non-transcript event types look like (so they can be filtered)

        Current placeholder — attempts a best-effort parse of common patterns:
        Checks several field-name candidates that E2E or OpenAI-compatible
        endpoints commonly use.  This will likely need adjustment once the
        actual schema is known.
        """
        # TODO: replace with the verified E2E response schema when docs arrive
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug("WhisperStreamingSession: could not parse frame: %s", exc)
            return None

        if not isinstance(msg, dict):
            return None

        # ── Candidate field extraction (update once schema is confirmed) ──
        # Try the most common transcript field names used by streaming STT APIs.
        # The first non-empty value found is used.
        text: str = (
            msg.get("text")                                         # simple flat
            or msg.get("transcript")                                # flat
            or (msg.get("delta") or {}).get("text", "")            # delta envelope
            or (msg.get("result") or {}).get("text", "")           # result envelope
            or (msg.get("alternatives") or [{}])[0].get("transcript", "")  # array
            or ""
        )
        text = text.strip()
        if not text:
            return None

        return _TranscriptEvent(
            type="data",
            data=_TranscriptData(transcript=text),
        )


# ── Provider class ─────────────────────────────────────────────────────────


class WhisperStreamingASR(BaseSTTProvider):
    """
    E2E Whisper Large-v3 real-time streaming ASR provider.

    Drop-in replacement for ``SarvamStreamingASR``.  Controlled entirely
    through ``.env`` variables; no code changes required when endpoint
    details are confirmed.

    ``.env`` variables consumed
    ---------------------------
    ``STT_BASE_URL``  – WebSocket URL of the E2E streaming endpoint
    ``STT_API_KEY``   – Bearer token / API key
    ``STT_MODEL``     – Model name forwarded to the endpoint
    ``STT_TASK``      – ``"transcribe"`` (default) or ``"translate"``
    ``STT_LANGUAGE``  – BCP-47 language hint; blank = auto-detect
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._validate_config()

    def _validate_config(self) -> None:
        """Raise early with a clear message if required settings are absent."""
        if not self._settings.stt_base_url:
            raise ValueError(
                "STT_BASE_URL must be set in .env when ASR_PROVIDER=whisper.\n"
                "Set it to the E2E Whisper streaming endpoint URL."
            )
        if not self._settings.stt_api_key:
            raise ValueError(
                "STT_API_KEY must be set in .env when ASR_PROVIDER=whisper.\n"
                "Set it to the E2E API key / bearer token."
            )

    def _build_websocket_url(self) -> str:
        """
        [TODO-URL] Build the full WebSocket URL for the E2E streaming endpoint.

        E2E endpoint structure (confirmed via connectivity probe):
          STT_BASE_URL (REST base) : https://infer.e2enetworks.net/project/.../endpoint/.../v1
          WebSocket route          : /ws  (mounted at the bare endpoint root, NOT under /v1)
          Final WS URL             : wss://infer.e2enetworks.net/project/.../endpoint/.../ws

        The /v1 suffix in STT_BASE_URL is the OpenAI-compatible REST prefix.
        The WebSocket server is mounted at the endpoint root before /v1, so
        this method strips any trailing /v1 path component before appending
        the WebSocket route.

        Probe evidence: every route under .../v1/<path> returned HTTP 404.
        The WebSocket route is expected at the bare endpoint base.

        The WS path defaults to "/ws" and can be overridden via STT_WS_PATH
        in .env if E2E changes the route (no code change required).
        """
        base = (self._settings.stt_base_url or "").rstrip("/")

        # Convert http(s) scheme to ws(s)
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]

        # Strip the /v1 (or /v2) REST API prefix -- the WS server is mounted
        # at the bare endpoint root, not under the REST prefix.
        # e.g.  .../endpoint/is-11156/v1  -->  .../endpoint/is-11156
        # Probe evidence: all routes under .../v1/<path> returned HTTP 404.
        for rest_suffix in ("/v1", "/v2"):
            if base.endswith(rest_suffix):
                base = base[: -len(rest_suffix)]
                logger.debug(
                    "WhisperStreamingASR: stripped REST suffix %r from base URL",
                    rest_suffix,
                )
                break

        # Append the WebSocket route (override via STT_WS_PATH in .env if needed)
        ws_path = getattr(self._settings, "stt_ws_path", None) or "/ws"
        if not ws_path.startswith("/"):
            ws_path = "/" + ws_path

        return base + ws_path

    def _build_auth_headers(self) -> dict[str, str]:
        """
        [TODO-AUTH] Build HTTP headers for WebSocket authentication.

        Update this method when E2E documentation is available.  No other
        file needs to change.

        Known unknowns to resolve from E2E documentation:
        □ Header name (Authorization vs X-API-Key vs custom)
        □ Token format (Bearer <token> vs raw token vs other)
        □ Additional required headers (e.g. Content-Type, Accept, tenant ID)

        Current placeholder — standard Bearer token, which is the most
        common pattern for OpenAI-compatible endpoints.
        """
        # TODO: replace with the confirmed auth scheme when E2E docs arrive
        return {"Authorization": f"Bearer {self._settings.stt_api_key}"}

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[WhisperStreamingSession]:
        """
        Open a streaming WebSocket session to the E2E Whisper endpoint.

        Yields a ``WhisperStreamingSession`` that satisfies the
        ``websocket_server.py`` interface contract.

        Called twice per connection (once for customer, once for worker),
        producing two independent concurrent sessions — matching the
        Sarvam behaviour exactly.
        """
        base_url = (self._settings.stt_base_url or "").rstrip("/")
        url = self._build_websocket_url()
        headers = self._build_auth_headers()

        # ── Verbose connection diagnostics ──────────────────────────────────
        logger.debug("WhisperStreamingASR [CONNECT] base_url=%s", base_url)
        logger.debug("WhisperStreamingASR [CONNECT] final_ws_url=%s", url)
        # Log header keys only — never log the token value
        logger.debug(
            "WhisperStreamingASR [CONNECT] headers=%s",
            {k: (v[:12] + "...") if k.lower() == "authorization" else v
             for k, v in headers.items()},
        )
        logger.info("WhisperStreamingASR: connecting → %s", url)

        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=15,
            ) as ws:
                logger.info(
                    "WhisperStreamingASR: handshake OK — "
                    "subprotocol=%s remote=%s",
                    getattr(ws, "subprotocol", None),
                    getattr(ws, "remote_address", None),
                )
                session = WhisperStreamingSession(
                    ws=ws,
                    model=self._settings.stt_model or "",
                    task=self._settings.stt_task or "transcribe",
                    language=self._settings.stt_language or "",
                )
                await session._handshake()
                logger.info("WhisperStreamingASR: session ready")
                yield session
        except websockets.exceptions.InvalidStatus as exc:
            # Capture HTTP status code returned during the WS upgrade
            status = getattr(exc.response, "status_code", None)
            reason = getattr(exc.response, "reason_phrase", "") or ""
            logger.error(
                "WhisperStreamingASR: WS handshake rejected — "
                "HTTP %s %s  url=%s",
                status, reason, url,
            )
            logger.debug(
                "WhisperStreamingASR: rejection headers=%s",
                dict(getattr(exc.response, "headers", {})),
            )
            raise
        except Exception as exc:
            logger.error(
                "WhisperStreamingASR: connection failed — %s: %s  url=%s",
                type(exc).__name__, exc, url,
            )
            raise
