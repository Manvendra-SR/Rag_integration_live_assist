const OFFSCREEN_DOCUMENT_PATH = "offscreen.html";

let extensionState = {
  status: "idle",
  detail: "",
  websocketUrl: "ws://127.0.0.1:8089",
  callId: "",
};

async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [offscreenUrl],
  });

  if (contexts.length > 0) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture tab audio and microphone audio for live STT while keeping tab audio audible.",
  });
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({
    active: true,
    lastFocusedWindow: true,
  });
  return tabs[0];
}

function setState(patch) {
  extensionState = {
    ...extensionState,
    ...patch,
  };
}

async function startCapture(message) {
  const activeTab = await getActiveTab();
  if (!activeTab || !activeTab.id) {
    throw new Error("No active tab found. Open Google Meet and try again.");
  }

  await ensureOffscreenDocument();

  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: activeTab.id,
  });

  setState({
    status: "starting",
    detail: "Preparing tab and microphone capture...",
    websocketUrl: message.websocketUrl,
    callId: message.callId,
  });

  await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "START_CAPTURE",
    streamId,
    websocketUrl: message.websocketUrl,
    callId: message.callId,
    tabTitle: activeTab.title || "",
  });
}

async function stopCapture() {
  await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "STOP_CAPTURE",
  });

  setState({
    status: "idle",
    detail: "",
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "GET_STATUS") {
    sendResponse(extensionState);
    return true;
  }

  if (message.type === "START_CAPTURE_FROM_POPUP") {
    startCapture(message)
      .then(() => sendResponse({ ok: true }))
      .catch((error) => {
        setState({
          status: "error",
          detail: error.message || String(error),
        });
        sendResponse({
          ok: false,
          error: error.message || String(error),
        });
      });
    return true;
  }

  if (message.type === "STOP_CAPTURE_FROM_POPUP") {
    stopCapture()
      .then(() => sendResponse({ ok: true }))
      .catch((error) => {
        setState({
          status: "error",
          detail: error.message || String(error),
        });
        sendResponse({
          ok: false,
          error: error.message || String(error),
        });
      });
    return true;
  }

  if (message.type === "OFFSCREEN_STATUS") {
    setState({
      status: message.status,
      detail: message.detail || "",
    });
    return false;
  }

  return false;
});
