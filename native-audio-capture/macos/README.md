# macOS Native Audio Capture

This contains the macOS native audio probe and streamer used by the desktop
Live Assist MVP.

It uses:

- `AVFoundation` / `AVAudioEngine` for default microphone capture
- `ScreenCaptureKit` for system-output audio capture

## Probe Run

```bash
cd native-audio-capture/macos
swift run native-audio-probe
```

Speak into the mic and play meeting/audio through the current default output.

Expected logs:

```text
[Native Devices] default_input="MacBook Air Microphone" default_output="MacBook Air Speakers"
[Native Mic] rms=0.01234 peak=0.12345
[Native System] rms=0.01876 peak=0.20123
```

For the current system-audio diagnosis, use the verbose format mode:

```bash
swift run native-audio-probe --debug-format
```

To test whether excluding the helper's own process audio is related to silent
buffers:

```bash
swift run native-audio-probe --include-own-audio --debug-format
```

The probe also accepts a capture-mode flag:

```bash
swift run native-audio-probe --capture-mode display --debug-format
```

`display` is the active implementation. Other values are logged and safely
fall back to display capture while we validate which ScreenCaptureKit filter is
best for production.

## Diagnostic Logs

The probe prints:

- macOS version and probe options
- default input and output device names
- ScreenCaptureKit shareable content counts: displays, apps, and windows
- selected display id and size
- ScreenCaptureKit audio format metadata when `--debug-format` is enabled
- system sample byte count and parsed format even when RMS is zero

Important lines:

```text
[Native Shareable Content] displays=1 apps=12 windows=30
[Native Display] selected_id=1 width=1440 height=900
[Native System Format] sample_count=... channel_count=... sample_rate=... format_id=... data_ready=true byte_count=...
[Native System] rms=0.00000 peak=0.00000 byte_count=... parsed_format=float32
```

If `byte_count` is greater than zero but RMS stays zero while YouTube or meeting
audio is playing, ScreenCaptureKit is likely delivering silent system-audio
buffers because of permissions, routing, or filter behavior.

## Streamer Run

The streamer sends native mic + system audio into the existing Live Assist
websocket contract:

```text
left channel  = Customer / system audio
right channel = Worker / mic audio
format        = 16kHz PCM16 stereo
```

Run it manually:

```bash
cd native-audio-capture/macos
swift run native-audio-streamer \
  --ws-url ws://127.0.0.1:8089 \
  --call-id electron-test \
  --session-id electron-session-1
```

Expected streamer logs:

```text
[Native Stream] connected=true
[Native Energy] system_rms=... mic_rms=...
[Native WS] sent_chunk=... bytes=16384
[Native Backend Event] {"type":"TRANSCRIPT",...}
```

The desktop Electron app now starts this streamer in native mode by default and
relays `[Native Backend Event]` messages into the existing Live Transcription
and Live Assist UI.

If the backend websocket is unavailable or drops, the streamer logs:

```text
[Native WS Reconnect] reason=... retry_seconds=2
```

Keep `DESKTOP_NATIVE_INCLUDE_OWN_AUDIO=true` for local MVP testing. On this
machine, ScreenCaptureKit produced silent system-audio buffers when current
process audio was excluded.

## Permissions

macOS may require:

- Microphone permission
- Screen & System Audio Recording permission

Check macOS Settings > Privacy & Security and grant permission to the host that
is running the probe, for example Terminal, iTerm, VS Code, or the packaged
helper. After changing permissions, quit and reopen that host app before testing
again.

For a production app, this helper must be packaged inside a signed/notarized
desktop app with a clear permission onboarding screen.

## Risk

ScreenCaptureKit behavior varies by macOS version and permission state. This
spike is intentionally separate from Electron so we can validate native capture
before touching the working Live Assist stream.
