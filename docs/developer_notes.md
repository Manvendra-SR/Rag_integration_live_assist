# Developer Notes

## Why This Structure Exists

The old code had two useful flows:

- Aashi backend: extension audio, STT buffering, Live Assist, LangGraph, RAG,
  response generation, transcript storage.
- Manvendra frontend: desktop audio capture, stereo PCM generation, optional
  translation/transcription path.

This folder keeps the working behavior but separates responsibilities so future
teams can replace one component without rewriting the whole product.

## Module Boundaries

- `frontends/desktop-electron`: primary MVP audio client.
- `frontends/chrome-extension`: future alternate audio client.
- `native-audio-capture`: native desktop speaker/microphone capture spike for
  scalable desktop customer audio.
- `backend/live_assist/audio`: websocket ingestion, PCM handling, speaker split,
  utterance buffering.
- `backend/live_assist/providers`: replaceable ASR, translation, LLM, and RAG
  providers.
- `backend/live_assist/live_cycle`: business logic and LangGraph orchestration.
- `backend/live_assist/storage`: transcript/session/context persistence.
- `backend/live_assist/mcp`: future MCP server/client boundary.
- `backend/live_assist/future`: contracts for agents, memory, and integrations.

## Future Multi-Agent Live Assist

Today the graph is still an MVP RAG path:

```text
classify_turn_intent -> enrich_query -> retrieve_knowledge -> generate_assist_response
```

Later, `classify_turn_intent` should route a turn to one or more specialist
agents:

- Product Answer Agent
- Next Best Question Agent
- Stage Guidance Agent
- Objection Handling Agent
- Relationship Building Agent
- Compliance/Risk Agent
- Winning Pattern Agent

The graph should merge those outputs into one clean Live Assist UI payload:

```json
{
  "answer": "...",
  "next_question": "...",
  "stage_guidance": "...",
  "objection_help": "...",
  "risk_warning": "...",
  "source_context": {}
}
```

Do not put CRM/calendar/email logic directly inside audio or frontend code.
Those should be context providers or tools called by orchestration.

## Desktop Customer Audio Direction

Electron should remain the desktop UI shell. Native OS-level capture should own
customer/system audio for the desktop product:

- macOS: Swift helper using ScreenCaptureKit for system output and AVFoundation
  for microphone.
- Windows: WASAPI loopback for default render/output device and WASAPI input
  for microphone.
- Linux: deferred until needed.

The backend audio contract must stay stable: Customer on left, Worker on right,
PCM16 stereo into the websocket stream server.

Stable desktop mode is intentionally simple for headset reliability:

- no echo suppression
- no overlap suppression
- no text-similarity suppression
- no positive RMS gate before STT

Experimental echo tools should stay behind `PYTHON_WS_EXPERIMENTAL_ECHO_TOOLS`.
Echo suppression can be added later without changing the
frontend/backend PCM contract.

## Transcript Shape

Each turn should preserve:

- session id
- speaker/source
- timestamp
- raw text
- translated text if available
- whether Live Assist was triggered
- workflow target
- product/context metadata

This transcript later feeds:

- entity extraction
- post-meeting analysis
- sales memory
- winning-pattern analysis
- call audit
- CRM updates

## Scaling Path

Local MVP:

- SQLite for transcript/context
- local Chroma for RAG
- one API process
- one stream process

Production path:

- API and stream server deployed separately
- transcript/context moved to Postgres, Azure SQL, Cosmos DB, or similar
- vector search moved to Azure AI Search, Pinecone, Weaviate, or managed Chroma
- background worker/queue added for slow agents and post-meeting processing
- structured logs and monitoring added around ASR latency, workflow latency,
  provider errors, and per-call state
