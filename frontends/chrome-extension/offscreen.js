const TARGET_SAMPLE_RATE = 16000;

let captureSession = null;

function sendStatus(status, detail = "") {
  chrome.runtime.sendMessage({
    type: "OFFSCREEN_STATUS",
    status,
    detail,
  });
}

function buildWebsocketUrl(baseUrl, callId) {
  if (!callId) {
    return baseUrl;
  }

  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}call_id=${encodeURIComponent(callId)}`;
}

function downsampleBuffer(input, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return input;
  }

  if (outputSampleRate > inputSampleRate) {
    throw new Error("Output sample rate must be less than or equal to input sample rate.");
  }

  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.round(input.length / sampleRateRatio));
  const output = new Float32Array(outputLength);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < output.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;

    for (
      let i = offsetBuffer;
      i < nextOffsetBuffer && i < input.length;
      i += 1
    ) {
      accum += input[i];
      count += 1;
    }

    output[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return output;
}

function interleaveToPcm16(leftChannel, rightChannel) {
  const frameCount = Math.min(leftChannel.length, rightChannel.length);
  const pcm = new Int16Array(frameCount * 2);

  for (let i = 0; i < frameCount; i += 1) {
    const leftSample = Math.max(-1, Math.min(1, leftChannel[i]));
    const rightSample = Math.max(-1, Math.min(1, rightChannel[i]));

    pcm[i * 2] = leftSample < 0 ? leftSample * 0x8000 : leftSample * 0x7fff;
    pcm[i * 2 + 1] =
      rightSample < 0 ? rightSample * 0x8000 : rightSample * 0x7fff;
  }

  return pcm.buffer;
}

async function createTabStream(streamId) {
  const audioConstraints = {
    mandatory: {
      chromeMediaSource: "tab",
      chromeMediaSourceId: streamId,
    },
  };

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: audioConstraints,
    video: false,
  });

  const supportedConstraints = navigator.mediaDevices.getSupportedConstraints();
  const tabAudioTrack = stream.getAudioTracks()[0];

  if (
    supportedConstraints.suppressLocalAudioPlayback &&
    tabAudioTrack &&
    typeof tabAudioTrack.applyConstraints === "function"
  ) {
    try {
      await tabAudioTrack.applyConstraints({
        suppressLocalAudioPlayback: false,
      });
    } catch (error) {
      console.debug("Unable to disable local audio suppression on tab track", error);
    }
  }

  return stream;
}

async function createMicStream() {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
}

function normalizeDeviceLabel(label) {
  return (label || "")
    .toLowerCase()
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function resolvePlaybackOutputDevice({ preferredSinkId, micStream }) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const audioOutputs = devices.filter((device) => device.kind === "audiooutput");

  if (audioOutputs.length === 0) {
    return {
      deviceId: "",
      label: "",
      strategy: "no-audio-output-devices",
    };
  }

  if (preferredSinkId) {
    const preferredDevice = audioOutputs.find(
      (device) => device.deviceId === preferredSinkId,
    );

    if (preferredDevice) {
      return {
        deviceId: preferredDevice.deviceId,
        label: preferredDevice.label || "selected output",
        strategy: "preferred",
      };
    }
  }

  const micTrack = micStream?.getAudioTracks?.()[0];
  const micSettings =
    typeof micTrack?.getSettings === "function" ? micTrack.getSettings() : {};

  if (micSettings.groupId) {
    const groupedOutput = audioOutputs.find(
      (device) => device.groupId && device.groupId === micSettings.groupId,
    );

    if (groupedOutput) {
      return {
        deviceId: groupedOutput.deviceId,
        label: groupedOutput.label || "matched output",
        strategy: "group-id",
      };
    }
  }

  const micLabel = normalizeDeviceLabel(micTrack?.label || "");
  if (micLabel) {
    const labelTokens = micLabel.split(" ").filter((token) => token.length >= 3);
    const labelMatchedOutput = audioOutputs.find((device) => {
      const outputLabel = normalizeDeviceLabel(device.label);
      return (
        outputLabel &&
        labelTokens.length > 0 &&
        labelTokens.every((token) => outputLabel.includes(token))
      );
    });

    if (labelMatchedOutput) {
      return {
        deviceId: labelMatchedOutput.deviceId,
        label: labelMatchedOutput.label || "matched output",
        strategy: "label-match",
      };
    }
  }

  const communicationsOutput = audioOutputs.find(
    (device) => device.deviceId === "communications",
  );
  if (communicationsOutput) {
    return {
      deviceId: communicationsOutput.deviceId,
      label: communicationsOutput.label || "communications output",
      strategy: "communications",
    };
  }

  const defaultOutput =
    audioOutputs.find((device) => device.deviceId === "default") || audioOutputs[0];
  return {
    deviceId: defaultOutput.deviceId,
    label: defaultOutput.label || "default output",
    strategy: defaultOutput.deviceId === "default" ? "default" : "first-output",
  };
}

async function configurePlaybackSink(playbackContext, playbackDevice) {
  if (!playbackContext) {
    return {
      applied: false,
      detail: "no playback context",
    };
  }

  if (!playbackDevice?.deviceId) {
    return {
      applied: false,
      detail: "using browser default output",
    };
  }

  if (typeof playbackContext.setSinkId !== "function") {
    return {
      applied: false,
      detail: `AudioContext.setSinkId unavailable, using browser default output instead of ${playbackDevice.label}`,
    };
  }

  try {
    await playbackContext.setSinkId(playbackDevice.deviceId);
    return {
      applied: true,
      detail: `playback output: ${playbackDevice.label} (${playbackDevice.strategy})`,
    };
  } catch (error) {
    console.debug("Unable to set playback sink", error);
    return {
      applied: false,
      detail: `could not switch playback output to ${playbackDevice.label}: ${error.message || String(error)}`,
    };
  }
}

async function stopCapture() {
  if (!captureSession) {
    sendStatus("idle", "");
    return;
  }

  const {
    websocket,
    processor,
    silentGain,
    merger,
    tabSplitter,
    micSplitter,
    processingContext,
    playbackContext,
    tabSourceNode,
    micSourceNode,
    playbackSourceNode,
    playbackGain,
    tabStream,
    micStream,
  } = captureSession;

  try {
    processor.disconnect();
    silentGain.disconnect();
    merger.disconnect();
    tabSplitter.disconnect();
    micSplitter.disconnect();
    tabSourceNode.disconnect();
    micSourceNode.disconnect();
    if (playbackSourceNode) {
      playbackSourceNode.disconnect();
    }
    if (playbackGain) {
      playbackGain.disconnect();
    }
  } catch (error) {
    console.debug("Node disconnect warning", error);
  }

  if (tabStream) {
    tabStream.getTracks().forEach((track) => track.stop());
  }

  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
  }

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.close();
  }

  if (playbackContext && playbackContext.state !== "closed") {
    await playbackContext.close();
  }

  if (processingContext && processingContext.state !== "closed") {
    await processingContext.close();
  }

  captureSession = null;
  sendStatus("idle", "");
}

async function startCapture({
  streamId,
  websocketUrl,
  callId,
  tabTitle,
  playbackDeviceId,
}) {
  await stopCapture();
  sendStatus("starting", "Opening tab and microphone streams...");

  const tabStream = await createTabStream(streamId);
  const micStream = await createMicStream();
  const micTrack = micStream.getAudioTracks()[0];
  const playbackDevice = await resolvePlaybackOutputDevice({
    preferredSinkId: playbackDeviceId,
    micStream,
  });
  const tabAudioTrack = tabStream.getAudioTracks()[0];
  const tabTrackSettings =
    typeof tabAudioTrack?.getSettings === "function"
      ? tabAudioTrack.getSettings()
      : {};
  const needsPlaybackLoopback =
    tabTrackSettings.suppressLocalAudioPlayback !== false;

  const processingContext = new AudioContext({
    sampleRate: TARGET_SAMPLE_RATE,
    latencyHint: "interactive",
  });
  await processingContext.resume();

  let playbackContext = null;
  let playbackSourceNode = null;
  let playbackGain = null;
  let playbackSinkResult = {
    applied: false,
    detail: "loopback playback not needed",
  };

  if (needsPlaybackLoopback) {
    playbackContext = new AudioContext({
      latencyHint: "interactive",
    });
    playbackSinkResult = await configurePlaybackSink(playbackContext, playbackDevice);
    await playbackContext.resume();
  }

  const websocket = new WebSocket(buildWebsocketUrl(websocketUrl, callId));
  websocket.binaryType = "arraybuffer";

  const tabSourceNode = processingContext.createMediaStreamSource(tabStream);
  const micSourceNode = processingContext.createMediaStreamSource(micStream);
  const tabSplitter = processingContext.createChannelSplitter(2);
  const micSplitter = processingContext.createChannelSplitter(2);
  const merger = processingContext.createChannelMerger(2);
  const processor = processingContext.createScriptProcessor(4096, 2, 2);
  const silentGain = processingContext.createGain();
  silentGain.gain.value = 0;

  tabSourceNode.connect(tabSplitter);
  micSourceNode.connect(micSplitter);
  tabSplitter.connect(merger, 0, 0);
  micSplitter.connect(merger, 0, 1);
  merger.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(processingContext.destination);

  if (needsPlaybackLoopback) {
    playbackSourceNode = playbackContext.createMediaStreamSource(tabStream);
    playbackGain = playbackContext.createGain();
    playbackGain.gain.value = 1;
    playbackSourceNode.connect(playbackGain);
    playbackGain.connect(playbackContext.destination);
  }

  websocket.onopen = () => {
    const playbackDetail = needsPlaybackLoopback
      ? playbackSinkResult.detail || "loopback playback enabled"
      : "local tab playback preserved by Chrome";
    const micDetail = micTrack?.label ? `mic: ${micTrack.label}` : "mic: default";
    sendStatus(
      "running",
      `Streaming "${tabTitle || "active tab"}" and microphone to ${websocketUrl}${needsPlaybackLoopback ? " (using audio loopback fallback)" : ""}\n${micDetail}\n${playbackDetail}`,
    );
  };

  websocket.onerror = () => {
    sendStatus("error", "WebSocket error. Check that ws://127.0.0.1:8089 is running.");
  };

  websocket.onclose = () => {
    if (captureSession) {
      sendStatus("idle", "WebSocket connection closed.");
    }
  };

  processor.onaudioprocess = (event) => {
    if (websocket.readyState !== WebSocket.OPEN) {
      return;
    }

    const leftInput = event.inputBuffer.getChannelData(0);
    const rightInput = event.inputBuffer.getChannelData(1);
    const sampleRate = processingContext.sampleRate;

    const leftChannel =
      sampleRate === TARGET_SAMPLE_RATE
        ? leftInput
        : downsampleBuffer(leftInput, sampleRate, TARGET_SAMPLE_RATE);
    const rightChannel =
      sampleRate === TARGET_SAMPLE_RATE
        ? rightInput
        : downsampleBuffer(rightInput, sampleRate, TARGET_SAMPLE_RATE);

    const pcmBuffer = interleaveToPcm16(leftChannel, rightChannel);
    websocket.send(pcmBuffer);
  };

  captureSession = {
    websocket,
    processor,
    silentGain,
    merger,
    tabSplitter,
    micSplitter,
    processingContext,
    playbackContext,
    tabSourceNode,
    micSourceNode,
    playbackSourceNode,
    playbackGain,
    tabStream,
    micStream,
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target !== "offscreen") {
    return false;
  }

  if (message.type === "START_CAPTURE") {
    startCapture(message)
      .then(() => sendResponse({ ok: true }))
      .catch(async (error) => {
        await stopCapture();
        sendStatus("error", error.message || String(error));
        sendResponse({
          ok: false,
          error: error.message || String(error),
        });
      });
    return true;
  }

  if (message.type === "STOP_CAPTURE") {
    stopCapture()
      .then(() => sendResponse({ ok: true }))
      .catch((error) => {
        sendStatus("error", error.message || String(error));
        sendResponse({
          ok: false,
          error: error.message || String(error),
        });
      });
    return true;
  }

  return false;
});
