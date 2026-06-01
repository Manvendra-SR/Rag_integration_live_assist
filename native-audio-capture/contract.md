# Native Audio Capture Contract

The native helper will eventually stream stereo PCM chunks to the existing
backend websocket.

## PCM Contract

```text
sample_rate: 16000 Hz
sample_format: signed 16-bit little-endian PCM
channels: 2 interleaved stereo
left: Customer / system output audio
right: Worker / microphone audio
chunking: compatible with backend PYTHON_WS_CHUNK_SIZE and PYTHON_WS_BUFFER_CHUNKS
```

## Diagnostic Contract

The macOS streamer prints:

```text
[Native Devices] default_input="..." default_output="..."
[Native Stream] connected=true
[Native Energy] system_rms=... mic_rms=...
[Native WS] sent_chunk=...
[Native Backend Event] {"type":"TRANSCRIPT",...}
[Native Error] source=system type=... message="..."
```

No provider keys or secrets should be printed.
