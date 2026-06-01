# Live Assist MVP

Clean MVP foundation for the current Aashi + Manvendra Live Assist code.

The immediate runnable desktop path is:

```text
Desktop Electron app
  -> native macOS mic + system audio capture
  -> stereo PCM websocket audio
  -> backend stream server
  -> ASR/translation
  -> utterance buffering
  -> FastAPI Live Assist route
  -> LangGraph orchestration
  -> RAG retrieval
  -> response generation
  -> transcript + assist response back to desktop UI
```

The Chrome extension is preserved as a future alternate input client. Desktop
native capture, browser fallback capture, and extension clients should all speak
the same backend audio contract: stereo PCM16 where left channel is Customer and
right channel is Worker/Manager.

## Current MVP Flow

1. The desktop app starts the native macOS audio streamer by default.
2. The streamer captures system/meeting audio as `Customer` and mic audio as
   `Worker`.
3. The streamer sends stereo PCM16 frames to `ws://127.0.0.1:8089`.
4. `live_assist.audio.websocket_server` splits stereo audio into left/right mono.
5. The ASR provider streams each channel to Sarvam.
6. Partial transcripts are sent back to the frontend.
7. The frontend renders partial transcripts immediately as live drafts keyed by
   `utterance_id`; final transcripts freeze that draft and never overwrite older
   finalized turns.
8. Final utterances are posted to the FastAPI Live Assist route.
9. Customer turns trigger Live Assist:
   - store transcript turn
   - update context
   - enrich query
   - retrieve RAG context
   - generate answer
   - store assistant turn
   - send answer back to desktop UI
10. Worker turns only update transcript/context and do not generate answers.

## Local Run

From `live-assist-mvp/`:

```bash
cp .env.example .env
```

Fill in:

```env
PYTHON_WS_SARVAM_API_KEY=...
GROQ_API_KEY=...
```

Install backend dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the local Chroma RAG index once:

```bash
python scripts/index_rag.py
```

Start the API:

```bash
uvicorn live_assist.api.app:app --host 0.0.0.0 --port 8000
```

Start the stream server in another terminal:

```bash
cd backend
source .venv/bin/activate
python -m live_assist.audio.websocket_server
```

Start the desktop app:

```bash
cd frontends/desktop-electron
npm install
npm start
```

The default desktop mode is stable native capture. Click **Start Listening** in
the desktop app. Expected logs in stable mode are intentionally quiet:

```text
[Native Devices] default_input="..." default_output="..."
[Native Energy] system_rms=... mic_rms=...
[STT Text] call=... speaker=...
```

For packet-level logs, set `DESKTOP_AUDIO_CAPTURE_MODE=desktop_native_diagnostic`
or `LIVE_ASSIST_DIAGNOSTICS_ENABLED=true`.

You can also run the macOS native streamer directly:

```bash
cd native-audio-capture/macos
swift run native-audio-streamer \
  --ws-url ws://127.0.0.1:8089 \
  --call-id native-test \
  --session-id native-session-1
```

Use the probe only when diagnosing capture permissions or format issues:

```bash
swift run native-audio-probe --include-own-audio --debug-format
```

Check whether another Mac has the local prerequisites:

```bash
python3 scripts/check_local_setup.py
```

This reports dependency/key status only. It does not print provider keys.

## Configuration

Configuration is environment-based. Important settings:

- `INPUT_SOURCE=desktop_native_stable|desktop_native_diagnostic|desktop_browser_fallback|extension`
- `DESKTOP_AUDIO_CAPTURE_MODE=desktop_native_stable|desktop_native_diagnostic|desktop_browser_fallback`
- `DESKTOP_NATIVE_INCLUDE_OWN_AUDIO=true`
- `DESKTOP_NATIVE_CHUNK_FRAMES=4096`
- `LIVE_ASSIST_DIAGNOSTICS_ENABLED=false`
- `PYTHON_WS_STT_SEND_RMS_FLOOR=0`
- `PYTHON_WS_EXPERIMENTAL_ECHO_TOOLS=false`
- `ASR_PROVIDER=sarvam`
- `TRANSLATION_PROVIDER=sarvam|none`
- `LLM_PROVIDER=groq`
- `RAG_PROVIDER=chroma`
- `TRANSCRIPT_STORAGE=sqlite`
- `STATE_STORAGE=sqlite|memory`
- `LIVE_FEEDBACK_WEBHOOK_URL=http://127.0.0.1:8000/livefeedback/webhook`

Do not commit real `.env` files or provider keys.

The previous code contained hardcoded keys in legacy files. Rotate those keys
before pushing anything derived from this workspace to GitHub.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No mic RMS | Grant Microphone permission to Terminal/Electron host, then restart the app. |
| No system RMS | Grant Screen & System Audio Recording permission and keep `DESKTOP_NATIVE_INCLUDE_OWN_AUDIO=true`. |
| Port 8089 busy | Stop the old websocket server process, then restart `python -m live_assist.audio.websocket_server`. |
| Native reconnect logs | Confirm the websocket server is running at `ws://127.0.0.1:8089`. |
| Audio energy but no STT text | Check Sarvam key, language/mode, and `[STT Send OK]` / `[STT Event]` logs. |
| Customer transcript but no answer | Check `[Live Assist Trigger]`, `[Enriched Query]`, `[RAG Retrieved]`, and provider key/RAG index. |
| Transcript appears late | Check `[Timing]` logs for `stt_text_received`, `final_flush`, `workflow_start`, and `workflow_done`. |
| Old transcript text changes | Confirm `TRANSCRIPT` events include `utterance_id`; live drafts update, finalized turns should not. |
| Customer shown as Worker | Stable mode does not suppress echo; use a headset first. Native AEC/suppression is future work. |

## Portability Notes

This repo is portable, but local runtime artifacts are machine-specific:

- `.env` must be created locally and must not be committed.
- `backend/.venv*` must be recreated with `pip install -r requirements.txt`.
- `frontends/*/node_modules` must be recreated with `npm install`.
- `native-audio-capture/**/.build` must be recreated with `swift build`.
- Chroma/RAG data must be indexed locally or provided as deployment data.
- macOS Microphone and Screen & System Audio Recording permissions must be
  granted on each machine.

## Where To Add Future Work

- New Live Assist agents:
  `backend/live_assist/future/agents.py` for contracts, then add LangGraph nodes
  in `backend/live_assist/live_cycle/graph.py`.

- CRM/calendar/email/memory/winning-pattern context:
  add provider adapters under `backend/live_assist/future/integrations.py` or
  dedicated provider folders, then inject their context into the Live Assist graph.

- Entity extraction and processing pipeline:
  consume transcript turns from `backend/live_assist/storage/sqlite.py` and write
  structured entities into a future storage table/provider.

- MCP server/client:
  use `backend/live_assist/mcp/server_tools.py` for Live Assist tools exposed to
  other systems, and `backend/live_assist/mcp/client.py` for calling external MCP
  servers such as CRM, calendar, email, memory, or web search.

- Pre-meeting and post-meeting agents:
  keep them outside the realtime audio stream. They should read/write the same
  transcript, context, memory, and CRM boundaries used by Live Assist.

- Windows native desktop capture:
  implement a WASAPI loopback helper that emits the same PCM contract documented
  in `native-audio-capture/contract.md`.

## Docker

For local Docker:

```bash
cp .env.example .env
docker compose up --build
```

For Azure, API and stream are intentionally separate Dockerfiles so they can be
deployed and scaled independently.
