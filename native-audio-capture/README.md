# Native Audio Capture Spike

This folder is the native desktop audio capture path for the desktop app.

The backend contract stays unchanged:

```text
left channel  = Customer / system output audio
right channel = Worker / microphone audio
PCM16 stereo  -> ws://127.0.0.1:8089
```

## Why This Exists

Electron `desktopCapturer` is not reliable enough for production customer audio
capture on macOS. It can briefly return a `System Audio` track and then end the
track, which also destabilizes the microphone graph.

The scalable desktop app should keep Electron as the UI shell and use a native
helper for audio capture.

## Current MVP

The macOS helper now streams audio to the existing Live Assist websocket
contract. Electron starts it by default in stable mode.

- default input device becomes Worker/right channel
- default/system output capture becomes Customer/left channel
- both channels are resampled to 16 kHz and sent as PCM16 stereo
- packet-level metadata is sent only in diagnostic mode
- if the default input changes, the mic engine restarts

Stable mode is the product baseline. Diagnostic mode is for debugging packet
loss, format issues, and permission problems.

## Platform Folders

- `macos`: Swift helper using AVFoundation for mic and ScreenCaptureKit for
  system audio.
- `windows`: WASAPI loopback implementation notes and acceptance checklist.

## Acceptance Criteria

- Built-in mic produces `mic_rms > 0` while speaking.
- Built-in speaker output produces `system_rms > 0` while meeting/audio plays.
- External speaker/headphones as default output also produce `system_rms > 0`.
- If system capture fails, mic capture remains alive.
- External headset/AirPods selected as the macOS default input produces right
  channel audio and Worker transcript.
