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

## Langfuse Observability

Live Assist integrates with [Langfuse Cloud](https://cloud.langfuse.com) for comprehensive observability of the RAG workflow, LLM calls, and custom functions. All traces, spans, token usage, and latencies are automatically captured and visualized in the Langfuse dashboard.

### Setup Instructions

#### 1. Obtain Langfuse Credentials

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com) and sign up or log in
2. Create a new project or select an existing one
3. Navigate to **Settings** → **API Keys**
4. Click **Create new API keys**
5. Copy the **Public Key** and **Secret Key** (save them securely - the secret key is only shown once)

#### 2. Configure Environment Variables

Add the following to your `.env` file:

```env
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
```

**Environment Variables:**

- `LANGFUSE_PUBLIC_KEY` (required): Your Langfuse project public key
- `LANGFUSE_SECRET_KEY` (required): Your Langfuse project secret key
- `LANGFUSE_HOST` (optional): Langfuse server URL, defaults to `https://cloud.langfuse.com`
- `LANGFUSE_ENABLED` (optional): Master switch to enable/disable tracing, defaults to `true`

#### 3. Restart the Backend

After adding the credentials, restart the backend API:

```bash
cd backend
source .venv/bin/activate
uvicorn live_assist.api.app:app --host 0.0.0.0 --port 8000
```

### Viewing Traces

#### Access the Langfuse Dashboard

1. Log in to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Select your project
3. Navigate to **Traces** in the left sidebar

#### Understanding the Trace Structure

Each RAG query creates a trace with the following structure:

```
Trace (grouped by session_id)
├── ingest_turn [span] - Store transcript turn
├── enrich_query [span] - Query enrichment
│   └── LLM call [generation] - llama-3.3-70b-versatile
├── cache_lookup [span] - Semantic cache check
├── retrieve_knowledge [span] - RAG retrieval
│   ├── retrieve_documents [span] - Vector search
│   └── rerank_documents [span] - Reranking
└── generate_assist_response [span] - Answer generation
    └── LLM call [generation] - llama-3.3-70b-versatile
```

#### Trace Metadata

Each trace includes rich metadata for filtering and analysis:

- **session_id**: Conversation identifier (groups multiple turns)
- **user_id**: User identifier
- **turn_id**: Turn number in the conversation
- **speaker**: `CUSTOMER` or `WORKER`
- **manual_question**: Whether query was explicit vs. from transcript
- **product**: Detected product context
- **route**: `rag_answer` or `context_only`

#### Viewing LLM Token Usage

1. Click on any trace to open the detail view
2. Expand **Generation** spans (LLM calls)
3. View token counts:
   - **Input tokens**: Tokens sent to the LLM
   - **Output tokens**: Tokens generated by the LLM
   - **Total tokens**: Sum of input + output
   - **Cost**: Estimated cost (if configured)

#### Viewing Cache Performance

1. Find the **cache_lookup** span in any trace
2. Check the metadata:
   - `result: "HIT"` - Cache hit, retrieval/generation skipped
   - `result: "MISS"` - Cache miss, full RAG workflow executed
   - `result: "SKIPPED"` - Cache bypassed (non-manual questions)
   - `similarity`: Similarity score between queries

#### Viewing Retrieval Details

1. Expand the **retrieve_knowledge** span
2. View metadata:
   - **retrieval_mode**: `reranked`, `vector`, or `hybrid`
   - **chunks_found**: Number of chunks retrieved
   - **doc_filter**: Document filter applied

### Disabling Langfuse

To temporarily disable Langfuse tracing:

```env
LANGFUSE_ENABLED=false
```

Or remove/comment out the Langfuse credentials in `.env`. The system will continue operating normally without tracing.

### Troubleshooting Langfuse Issues

| Issue | Solution |
| --- | --- |
| **Traces not appearing in dashboard** | 1. Verify credentials are correct in `.env`<br>2. Check backend logs for Langfuse warnings<br>3. Wait 5-10 seconds for traces to appear<br>4. Verify `LANGFUSE_ENABLED=true` |
| **"Langfuse credentials not configured" warning** | Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to `.env` |
| **"Langfuse authentication failed"** | 1. Verify keys are correct (no extra spaces)<br>2. Regenerate API keys in Langfuse dashboard<br>3. Update `.env` with new keys |
| **"Failed to flush Langfuse traces"** | 1. Check internet connectivity<br>2. Verify `LANGFUSE_HOST` URL is correct<br>3. Check Langfuse Cloud status page<br>4. System continues operating, traces may be lost |
| **Missing LLM token counts** | 1. Verify LLM calls are executing successfully<br>2. Check that callbacks are properly configured<br>3. Inspect generation spans in Langfuse |
| **Incomplete traces (missing spans)** | 1. Check backend logs for decorator errors<br>2. Verify workflow completed successfully<br>3. Check for early returns or exceptions |
| **High latency after enabling Langfuse** | 1. Normal overhead is 5-10ms per trace<br>2. Check `handler.flush()` calls (adds ~50-100ms)<br>3. Consider async flushing for production |

### Verifying Langfuse Integration

To verify the integration is working correctly:

1. **Health Check Endpoint:**

   ```bash
   curl http://localhost:8000/health/langfuse
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "message": "Langfuse connection successful",
     "host": "https://cloud.langfuse.com"
   }
   ```

2. **Test Query:**

   Send a test query via the Live Assist API and check the Langfuse dashboard within 5 seconds. The trace should appear with all expected spans.

3. **Check Backend Logs:**

   Look for log messages:
   ```
   INFO: Created Langfuse trace: https://cloud.langfuse.com/project/.../traces/...
   ```

### Data Privacy Considerations

- Query text and answers are sent to Langfuse Cloud
- User IDs and session IDs are included as metadata
- Langfuse Cloud is SOC 2 compliant
- Review [Langfuse data retention policies](https://langfuse.com/docs/data-security-privacy) for your compliance requirements

### Additional Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse LangGraph Integration Guide](https://langfuse.com/guides/cookbook/integration_langgraph)
- [Langfuse Python SDK](https://github.com/langfuse/langfuse-python)

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
