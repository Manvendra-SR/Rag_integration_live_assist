# Windows Native Audio Capture Plan

Windows should use WASAPI:

- microphone: default capture endpoint
- customer/system audio: default render endpoint in loopback mode

## Spike Acceptance

The first Windows helper should print:

```text
[Native Devices] default_input="..." default_output="..."
[Native Mic] rms=...
[Native System] rms=...
```

## Implementation Notes

Recommended implementation choices:

- C#/.NET helper using NAudio for fastest spike, or
- Rust/C++ helper using native WASAPI for production packaging.

WASAPI loopback should follow the default render/output device, so built-in
speakers, headphones, and external speakers are captured as long as they are the
active default output route.

## Production Risks

- Exclusive-mode devices can interfere with capture.
- Bluetooth devices can change sample rates and latency.
- Device changes must restart or rebind the capture graph.
- Installer should include microphone permission guidance and diagnostic logs.

