const websocketUrlInput = document.getElementById("websocketUrl");
const callIdInput = document.getElementById("callId");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusElement = document.getElementById("status");

function renderStatus(state) {
  statusElement.textContent = `Status: ${state.status}\n${state.detail || ""}`.trim();
}

async function loadSavedConfig() {
  const stored = await chrome.storage.local.get(["websocketUrl", "callId"]);

  if (stored.websocketUrl) {
    websocketUrlInput.value = stored.websocketUrl;
  }

  if (stored.callId) {
    callIdInput.value = stored.callId;
  }
}

async function refreshStatus() {
  const state = await chrome.runtime.sendMessage({ type: "GET_STATUS" });
  renderStatus(state);
}

async function saveConfig() {
  await chrome.storage.local.set({
    websocketUrl: websocketUrlInput.value.trim(),
    callId: callIdInput.value.trim(),
  });
}

async function ensureMicrophonePermission() {
  let permissionStream;
  try {
    permissionStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
  } catch (error) {
    throw new Error(
      "Microphone permission was blocked or dismissed. Allow mic access for this extension and try again.",
    );
  } finally {
    if (permissionStream) {
      permissionStream.getTracks().forEach((track) => track.stop());
    }
  }
}

startButton.addEventListener("click", async () => {
  await saveConfig();
  renderStatus({
    status: "starting",
    detail: "Requesting microphone permission...",
  });

  try {
    await ensureMicrophonePermission();
  } catch (error) {
    renderStatus({
      status: "error",
      detail: error.message || "Microphone permission failed.",
    });
    return;
  }

  const response = await chrome.runtime.sendMessage({
    type: "START_CAPTURE_FROM_POPUP",
    websocketUrl: websocketUrlInput.value.trim(),
    callId: callIdInput.value.trim(),
  });

  if (!response.ok) {
    renderStatus({
      status: "error",
      detail: response.error || "Failed to start capture.",
    });
    return;
  }

  await refreshStatus();
});

stopButton.addEventListener("click", async () => {
  const response = await chrome.runtime.sendMessage({
    type: "STOP_CAPTURE_FROM_POPUP",
  });

  if (!response.ok) {
    renderStatus({
      status: "error",
      detail: response.error || "Failed to stop capture.",
    });
    return;
  }

  await refreshStatus();
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadSavedConfig();
  await refreshStatus();
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.type !== "OFFSCREEN_STATUS") {
    return false;
  }

  renderStatus({
    status: message.status,
    detail: message.detail || "",
  });
  return false;
});
