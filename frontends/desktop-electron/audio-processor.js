// audio-processor.js — AudioWorklet processor (dual-channel)
//
// Receives two separate inputs:
//   Input 0 = Tab/system audio (Customer — left channel)
//   Input 1 = Microphone audio (Manager — right channel)
//
// Accumulates samples in a typed-array ring buffer and posts
// { micChunk, tabChunk } to the renderer thread every CHUNK_SIZE frames.
//
// Chunk size: 1024 samples @ 16kHz = 64ms per WebSocket send.
// (Previous value was 4096 = 256ms — caused visible audio delivery lag.)

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunkSize = 1024;    // 64ms @ 16kHz

    // Pre-allocated Float32 ring buffers — avoids GC pressure in the audio thread
    this._micBuf  = new Float32Array(this._chunkSize * 4);
    this._tabBuf  = new Float32Array(this._chunkSize * 4);
    this._writePos = 0;        // how many samples are currently buffered
  }

  process(inputs) {
    // inputs[0] = tab audio (Customer / left channel)
    // inputs[1] = mic audio (Manager  / right channel)
    const tabData = inputs[0]?.[0] ?? null;
    const micData = inputs[1]?.[0] ?? null;

    const frameCount = micData?.length ?? tabData?.length ?? 128;

    // Append this quantum into the ring buffers
    for (let i = 0; i < frameCount; i++) {
      this._micBuf[this._writePos + i] = micData ? micData[i] : 0.0;
      this._tabBuf[this._writePos + i] = tabData ? tabData[i] : 0.0;
    }
    this._writePos += frameCount;

    // Flush complete chunks to the renderer
    while (this._writePos >= this._chunkSize) {
      const micChunk = new Float32Array(this._chunkSize);
      const tabChunk = new Float32Array(this._chunkSize);

      micChunk.set(this._micBuf.subarray(0, this._chunkSize));
      tabChunk.set(this._tabBuf.subarray(0, this._chunkSize));

      // Shift remaining samples to the front of the buffer
      this._micBuf.copyWithin(0, this._chunkSize, this._writePos);
      this._tabBuf.copyWithin(0, this._chunkSize, this._writePos);
      this._writePos -= this._chunkSize;

      // Transfer ownership — zero-copy send to renderer thread
      this.port.postMessage(
        { micChunk, tabChunk },
        [micChunk.buffer, tabChunk.buffer]
      );
    }

    return true; // keep processor alive
  }
}

registerProcessor('mic-processor', MicProcessor);
