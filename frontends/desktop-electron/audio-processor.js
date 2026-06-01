// audio-processor.js — AudioWorklet processor (dual-channel)
//
// Receives two separate inputs:
//   Input 0 = Tab/system audio (Customer — left channel)
//   Input 1 = Microphone audio (Manager — right channel)
//
// Accumulates samples in a buffer until a full chunk is ready,
// then posts { micChunk, tabChunk } back to the renderer thread.

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._micBuffer = [];
    this._tabBuffer = [];
    this._chunkSize = 4096;
  }

  process(inputs) {
    // inputs[0] = tab audio source (Customer)
    // inputs[1] = mic audio source (Manager)
    const tabInput = inputs[0];
    const micInput = inputs[1];

    const tabData = (tabInput && tabInput[0]) ? tabInput[0] : null;
    const micData = (micInput && micInput[0]) ? micInput[0] : null;

    // Accumulate tab samples (or zeros if no tab source)
    const frameCount = (micData || tabData)?.length ?? 128;
    for (let i = 0; i < frameCount; i++) {
      this._tabBuffer.push(tabData ? tabData[i] : 0.0);
      this._micBuffer.push(micData ? micData[i] : 0.0);
    }

    // Once we have a full chunk, post it to the renderer
    while (this._micBuffer.length >= this._chunkSize) {
      const micChunk = new Float32Array(this._micBuffer.splice(0, this._chunkSize));
      const tabChunk = new Float32Array(this._tabBuffer.splice(0, this._chunkSize));
      this.port.postMessage(
        { micChunk, tabChunk },
        [micChunk.buffer, tabChunk.buffer]   // transfer ownership (zero-copy)
      );
    }

    return true; // keep processor alive
  }
}

registerProcessor('mic-processor', MicProcessor);
