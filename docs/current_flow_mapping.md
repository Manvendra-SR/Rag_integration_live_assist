# Current Flow Mapping

## Desktop Audio Ingestion

Source files:

- `frontends/desktop-electron/main.js`
- `frontends/desktop-electron/renderer.js`
- `frontends/desktop-electron/audio-processor.js`
- `frontends/desktop-electron/preload.js`
- `frontends/desktop-electron/index.html`

Responsibilities:

- request microphone audio
- keep Electron desktop/window audio capture as diagnostic only
- encode audio as stereo PCM16
- left channel: Customer
- right channel: Worker/Manager
- send frames to the backend websocket stream server

Production desktop customer audio should come from `native-audio-capture`, not
Electron `desktopCapturer`.

## Native Desktop Audio Capture

Source files:

- `native-audio-capture/*`

Responsibilities:

- capture default microphone as Worker
- capture default speaker/output audio as Customer
- validate built-in and external output devices
- eventually emit the same stereo PCM16 backend contract used by the desktop app

## Extension Audio Ingestion

Source files:

- `frontends/chrome-extension/*`

Responsibilities:

- capture Google Meet tab audio and microphone audio
- send stereo PCM16 frames to the same backend websocket server

## Transcription, Translation, Buffering

Source files:

- `backend/live_assist/audio/websocket_server.py`
- `backend/live_assist/audio/pcm.py`
- `backend/live_assist/audio/buffering.py`
- `backend/live_assist/providers/asr/sarvam.py`

Responsibilities:

- receive websocket audio
- split stereo PCM into customer/worker mono channels
- send speech chunks to ASR/translation
- merge partial transcripts into meaningful utterances
- forward final utterances to the Live Assist API

## Live Assist Cycle

Source files:

- `backend/live_assist/api/routes/live_feedback.py`
- `backend/live_assist/live_cycle/service.py`
- `backend/live_assist/live_cycle/graph.py`

Responsibilities:

- store transcript turns
- route customer turns into Live Assist
- route worker turns into context update only
- enrich query
- retrieve RAG context
- generate answer
- store assistant answer

## RAG, LLM, Storage

Source files:

- `backend/live_assist/providers/rag/chroma.py`
- `backend/live_assist/providers/llm/groq.py`
- `backend/live_assist/storage/sqlite.py`
- `backend/live_assist/storage/context_store.py`
- `backend/live_assist/storage/transcript_store.py`

Responsibilities:

- Chroma retrieval
- Groq-compatible LLM calls
- transcript persistence
- session/context persistence
