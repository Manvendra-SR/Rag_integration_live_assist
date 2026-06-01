from __future__ import annotations

import base64
import io
import math
import wave

PCM_SAMPLE_WIDTH_BYTES = 2
STEREO_FRAME_WIDTH_BYTES = PCM_SAMPLE_WIDTH_BYTES * 2


def pcm_to_wav_b64(pcm_bytes: bytes, sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def calculate_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < PCM_SAMPLE_WIDTH_BYTES:
        return 0.0

    samples = len(pcm_bytes) // PCM_SAMPLE_WIDTH_BYTES
    total = 0
    for i in range(0, len(pcm_bytes), PCM_SAMPLE_WIDTH_BYTES):
        sample = int.from_bytes(
            pcm_bytes[i : i + PCM_SAMPLE_WIDTH_BYTES],
            "little",
            signed=True,
        )
        total += sample * sample
    return math.sqrt(total / samples)


def split_stereo_to_mono(pcm_bytes: bytes) -> tuple[bytes, bytes]:
    trim = len(pcm_bytes) % STEREO_FRAME_WIDTH_BYTES
    if trim:
        pcm_bytes = pcm_bytes[:-trim]

    left = bytearray()
    right = bytearray()
    for i in range(0, len(pcm_bytes), STEREO_FRAME_WIDTH_BYTES):
        left.extend(pcm_bytes[i : i + PCM_SAMPLE_WIDTH_BYTES])
        right.extend(pcm_bytes[i + PCM_SAMPLE_WIDTH_BYTES : i + STEREO_FRAME_WIDTH_BYTES])
    return bytes(left), bytes(right)


def target_buffer_bytes(chunk_size: int, buffer_chunks: int) -> int:
    # Match the legacy stream behavior: process after CHUNK_SIZE * 2 * BUFFER_CHUNKS
    # bytes of stereo PCM, then split that chunk into left/right mono channels.
    return chunk_size * buffer_chunks * PCM_SAMPLE_WIDTH_BYTES
