// renderer.js — Audio capture + WebSocket client (dual-channel)
//
// WHAT THIS DOES:
//   1. On "Start Listening": shows a window/screen picker so user selects the
//      tab/app where the customer call is happening.
//   2. Captures BOTH:
//        • Mic (right channel)  = Manager / You
//        • System audio (left channel) = Customer (from selected window/screen)
//   3. Interleaves as stereo 16-bit PCM → Live Assist stream server:8089
//      stream server → ASR/translation → Live Assist API → transcript/assist UI

'use strict';

// ── Config ────────────────────────────────────────────────────────────────
const DEFAULT_WS_URL = 'ws://127.0.0.1:8089';
const DEFAULT_API_URL = 'http://127.0.0.1:8000';
let runtimeConfig = {};
let PROXY_URL = window.liveAssistConfig?.wsUrl || DEFAULT_WS_URL;
let API_URL = window.liveAssistConfig?.apiUrl || DEFAULT_API_URL;
const SAMPLE_RATE = 16000;
let FORCE_MIC_ONLY_DIAGNOSTIC = window.liveAssistConfig?.forceMicOnlyDiagnostic ?? false;
let AUDIO_CAPTURE_MODE = window.liveAssistConfig?.audioCaptureMode || 'desktop_native_stable';
let NATIVE_INCLUDE_OWN_AUDIO = true;
let NATIVE_CHUNK_FRAMES = 4096;
let DIAGNOSTICS_ENABLED = false;

// ── State ─────────────────────────────────────────────────────────────────
let proxySocket  = null;
let proxyReady   = false;
let audioCtx     = null;
let micStream    = null;
let tabStream    = null;   // system/tab audio stream from desktopCapturer
let micSourceNode = null;
let tabSourceNode = null;
let processor    = null;
let isRecording  = false;
let sentAudioChunks = 0;
let skippedWorkletChunks = 0;
let capturePacketSeq = 0;
let trackStatusTimer = null;
let nativeAudioRunning = false;
let nativeMicActive = false;
let nativeSystemActive = false;
const liveTranscriptEntries = new Map();
const lastFinalTranscript = new Map();
const activeTranscriptSegments = new Map();
let legacyUtteranceSeq = 0;
let transcriptSegmentSeq = 0;
let currentCallId = '';
let currentSessionId = '';
const pendingAssistEntries = new Map();

// ── DOM refs ──────────────────────────────────────────────────────────────
const btnStart     = document.getElementById('btn-start');
const btnStop      = document.getElementById('btn-stop');
const btnClear     = document.getElementById('btn-clear');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const errorBox     = document.getElementById('error-box');
const transcript   = document.getElementById('transcript');
const assistFeed   = document.getElementById('assist-feed');
const assistInput  = document.getElementById('assist-input');
const btnAssistSend = document.getElementById('btn-assist-send');
const versionLabel = document.getElementById('version-label');
const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
const assistPanel = document.getElementById('assist-panel');
const transcriptPanel = document.getElementById('transcript-panel');
let activeTab = 'assist';

// Source picker modal elements
const modal          = document.getElementById('source-modal');
const sourceGrid     = document.getElementById('source-grid');
const btnPickConfirm = document.getElementById('btn-pick-confirm');
const btnPickCancel  = document.getElementById('btn-pick-cancel');
const btnSkipTab     = document.getElementById('btn-skip-tab');

if (window.electronAPI?.version) {
  versionLabel.textContent = `Electron v${window.electronAPI.version} · loading config`;
}

async function loadRuntimeConfig() {
  try {
    runtimeConfig = await window.electronAPI?.getRuntimeConfig?.() || {};
  } catch (err) {
    console.warn('[Runtime Config] failed, using defaults:', err);
    runtimeConfig = {};
  }
  PROXY_URL = window.liveAssistConfig?.wsUrl || runtimeConfig.wsUrl || DEFAULT_WS_URL;
  API_URL = window.liveAssistConfig?.apiUrl || runtimeConfig.apiUrl || DEFAULT_API_URL;
  AUDIO_CAPTURE_MODE = window.liveAssistConfig?.audioCaptureMode || runtimeConfig.audioCaptureMode || 'desktop_native_stable';
  NATIVE_INCLUDE_OWN_AUDIO = runtimeConfig.includeOwnAudio !== false;
  NATIVE_CHUNK_FRAMES = Number(runtimeConfig.nativeChunkFrames || 4096);
  DIAGNOSTICS_ENABLED = Boolean(runtimeConfig.diagnosticsEnabled || AUDIO_CAPTURE_MODE === 'desktop_native_diagnostic');
  FORCE_MIC_ONLY_DIAGNOSTIC = Boolean(window.liveAssistConfig?.forceMicOnlyDiagnostic ?? runtimeConfig.forceMicOnlyDiagnostic ?? false);
  if (window.electronAPI?.version) {
    versionLabel.textContent = `Electron v${window.electronAPI.version} · ${AUDIO_CAPTURE_MODE}`;
  }
  console.log('[Runtime Config]', JSON.stringify(runtimeConfig));
}

function createRuntimeId(prefix) {
  const randomPart = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${randomPart}`;
}

function ensureRecordingSession() {
  if (!currentCallId) currentCallId = createRuntimeId('desktop-call');
  if (!currentSessionId) currentSessionId = createRuntimeId('desktop-session');
}

window.addEventListener('error', (event) => {
  console.error(
    '[Renderer Error]',
    event.message,
    event.filename,
    event.lineno,
    event.colno,
    event.error?.stack || ''
  );
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('[Renderer Unhandled Rejection]', event.reason?.stack || event.reason);
});

// ── UI helpers ────────────────────────────────────────────────────────────
function setStatus(msg, state = '') {
  statusText.textContent = msg;
  statusDot.className = 'status-dot' + (state ? ' ' + state : '');
}

function showError(msg) {
  errorBox.style.display = 'block';
  errorBox.textContent = '⚠ ' + msg;
}

function hideError() {
  errorBox.style.display = 'none';
}

function setActiveTab(tabName) {
  activeTab = tabName;
  const showAssist = tabName === 'assist';
  assistPanel.hidden = !showAssist;
  transcriptPanel.hidden = showAssist;

  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });
}

function speakerInfo(channel) {
  const role = channel === 1 ? 'manager' : 'customer';
  const label = channel === 1 ? 'You (Manager)' : 'Customer';
  return { role, label };
}

function createTranscriptEntry(text, channel, isLive = false, utteranceId = '') {
  if (!text?.trim()) return;
  const currentEmptyState = document.getElementById('empty-state');
  if (currentEmptyState) currentEmptyState.style.display = 'none';

  const { role, label } = speakerInfo(channel);

  const entry = document.createElement('div');
  entry.className = 'entry' + (isLive ? ' live-entry' : '');
  entry.dataset.channel = String(channel);
  if (utteranceId) entry.dataset.utteranceId = utteranceId;
  entry.innerHTML = `
    <span class="entry-speaker ${role}">${label}</span>
    <div class="entry-text ${role}">${escapeHtml(text)}</div>
  `;
  transcript.appendChild(entry);
  transcript.scrollTop = transcript.scrollHeight;
  return entry;
}

function updateEntryText(entry, text) {
  const textEl = entry?.querySelector('.entry-text');
  if (!textEl) return;
  textEl.innerHTML = escapeHtml(text);
  transcript.scrollTop = transcript.scrollHeight;
}

function getEntryText(entry) {
  return entry?.querySelector('.entry-text')?.textContent?.trim() || '';
}

function isTranscriptExpansion(previousText, nextText) {
  const previous = previousText.trim();
  const next = nextText.trim();
  if (!previous || !next) return true;
  return next === previous || next.startsWith(previous) || previous.startsWith(next);
}

function baseTranscriptKey(channel, utteranceId) {
  if (utteranceId) return utteranceId;
  const existing = Array.from(liveTranscriptEntries.keys()).find((key) => key.startsWith(`legacy:${channel}:`));
  if (existing) return existing;
  legacyUtteranceSeq += 1;
  return `legacy:${channel}:${legacyUtteranceSeq}`;
}

function nextTranscriptSegmentKey(baseKey) {
  transcriptSegmentSeq += 1;
  return `${baseKey}:segment:${transcriptSegmentSeq}`;
}

function addOrUpdateTranscriptEntry(text, channel, isFinal = false, utteranceId = '', metadata = {}) {
  const cleanText = text?.trim();
  if (!cleanText) return;
  const baseKey = baseTranscriptKey(channel, utteranceId);
  let key = activeTranscriptSegments.get(baseKey) || baseKey;
  if (metadata?.possible_bleed) {
    console.warn(`[Transcript Bleed Suppressed?] utterance=${key} channel=${channel} text=${cleanText}`);
  }

  if (isFinal) {
    const liveEntry = liveTranscriptEntries.get(key);
    if (liveEntry) {
      const currentText = getEntryText(liveEntry);
      if (isTranscriptExpansion(currentText, cleanText)) {
        updateEntryText(liveEntry, cleanText);
      }
      liveEntry.classList.remove('live-entry');
      liveTranscriptEntries.delete(key);
      activeTranscriptSegments.delete(baseKey);
      lastFinalTranscript.set(key, getEntryText(liveEntry) || cleanText);
      return;
    }

    if (lastFinalTranscript.get(key) === cleanText) {
      activeTranscriptSegments.delete(baseKey);
      return;
    } else {
      createTranscriptEntry(cleanText, channel, false, key);
    }

    activeTranscriptSegments.delete(baseKey);
    lastFinalTranscript.set(key, cleanText);
    return;
  }

  if (lastFinalTranscript.get(key) === cleanText) return;

  const liveEntry = liveTranscriptEntries.get(key);
  if (liveEntry) {
    const currentText = getEntryText(liveEntry);
    if (isTranscriptExpansion(currentText, cleanText)) {
      updateEntryText(liveEntry, cleanText);
      return;
    }

    liveEntry.classList.remove('live-entry');
    liveTranscriptEntries.delete(key);
    lastFinalTranscript.set(key, currentText);
    key = nextTranscriptSegmentKey(baseKey);
    activeTranscriptSegments.set(baseKey, key);
    const entry = createTranscriptEntry(cleanText, channel, true, key);
    liveTranscriptEntries.set(key, entry);
    return;
  }

  const entry = createTranscriptEntry(cleanText, channel, true, key);
  activeTranscriptSegments.set(baseKey, key);
  liveTranscriptEntries.set(key, entry);
}

function createAssistEntry({ label, text, role = 'assistant', pending = false }) {
  if (!text?.trim()) return null;
  const currentAssistEmptyState = document.getElementById('assist-empty-state');
  if (currentAssistEmptyState) currentAssistEmptyState.style.display = 'none';

  const entry = document.createElement('div');
  entry.className = 'entry';
  if (pending) entry.classList.add('pending-entry');
  entry.innerHTML = `
    <span class="entry-speaker ${role}">${escapeHtml(label)}</span>
    <div class="entry-text ${role}${pending ? ' pending' : ''}">${escapeHtml(text)}</div>
  `;
  assistFeed.appendChild(entry);
  assistFeed.scrollTop = assistFeed.scrollHeight;
  return entry;
}

function updateAssistEntry(entry, { label, text, role = 'assistant', pending = false }) {
  if (!entry) return;
  entry.classList.toggle('pending-entry', pending);
  entry.innerHTML = `
    <span class="entry-speaker ${role}">${escapeHtml(label)}</span>
    <div class="entry-text ${role}${pending ? ' pending' : ''}">${escapeHtml(text)}</div>
  `;
  assistFeed.scrollTop = assistFeed.scrollHeight;
}

function addAssistEntry(answer, label = 'Live Assist') {
  createAssistEntry({
    label,
    text: answer,
    role: 'assistant',
  });
}

function pendingAssistKey(parsed) {
  return parsed.utterance_id || `${parsed.call_id || 'call'}:${parsed.utterance || ''}`;
}

function setPendingAssist(parsed) {
  const key = pendingAssistKey(parsed);
  const entry = createAssistEntry({
    label: 'Live Assist',
    text: 'Thinking...',
    role: 'assistant',
    pending: true,
  });
  if (entry) pendingAssistEntries.set(key, entry);
}

function resolvePendingAssist(parsed, text, role = 'assistant') {
  const key = pendingAssistKey(parsed);
  const entry = pendingAssistEntries.get(key);
  if (entry) {
    updateAssistEntry(entry, {
      label: 'Live Assist',
      text,
      role,
      pending: false,
    });
    pendingAssistEntries.delete(key);
    return;
  }
  createAssistEntry({
    label: 'Live Assist',
    text,
    role,
  });
}

function handleLiveAssistPayload(parsed) {
  if (parsed.channel?.alternatives) {
    const text    = parsed.channel.alternatives[0].transcript || '';
    const channel = parsed.channel_index ? parsed.channel_index[0] : 0;
    if (text.trim()) {
      console.log(`[Transcript] ch${channel}:`, text);
      addOrUpdateTranscriptEntry(
        text,
        channel,
        Boolean(parsed.is_final),
        parsed.utterance_id || '',
        parsed.metadata || {}
      );
    }
  } else if (parsed.type === 'LIVE_ASSIST_RESULT') {
    const action = parsed.result?.action_result || {};
    if (action.answer) {
      addAssistEntry(action.answer);
    }
  } else if (parsed.type === 'LIVE_ASSIST_ERROR') {
    resolvePendingAssist(parsed, parsed.error || 'Live Assist backend error.', 'assistant');
    showError(parsed.error || 'Live Assist backend error.');
  }
}

function isNativeCaptureMode() {
  return [
    'desktop_native_stable',
    'desktop_native_diagnostic',
    'desktop_native',
    'native',
  ].includes(AUDIO_CAPTURE_MODE);
}

function parseNativeEnergy(line) {
  const systemMatch = line.match(/system_rms=([0-9.]+)/);
  const micMatch = line.match(/mic_rms=([0-9.]+)/);
  if (!systemMatch && !micMatch) return null;

  const systemRms = systemMatch ? Number(systemMatch[1]) : 0;
  const micRms = micMatch ? Number(micMatch[1]) : 0;
  return {
    systemRms,
    micRms,
    systemActive: systemRms > 0.003,
    micActive: micRms > 0.003,
  };
}

if (window.electronAPI?.onNativeAudioLog) {
  window.electronAPI.onNativeAudioLog((line) => {
    console.log(line);
    if (line.includes('[Native Stream] connected=true')) {
      setStatus('Native stream connected — waiting for audio…', 'connecting');
    }
    if (line.includes('[Native WS Reconnect]') || line.includes('[Native WS Receive Error]')) {
      setStatus('Native backend disconnected — retrying…', 'connecting');
    }
    if (line.includes('[Native Error]')) {
      showError(line);
      setStatus('Native capture error.', 'error');
    }
    if (line.includes('[Native Energy]')) {
      const energy = parseNativeEnergy(line);
      if (!energy) {
        setStatus('Native capture running — listening…', 'listening');
        return;
      }
      nativeMicActive = energy.micActive;
      nativeSystemActive = energy.systemActive;
      const micLabel = nativeMicActive ? 'mic active' : 'mic quiet';
      const systemLabel = nativeSystemActive ? 'system active' : 'system quiet';
      setStatus(`Native capture running — ${systemLabel}, ${micLabel}`, 'listening');
    }
  });
}

if (window.electronAPI?.onNativeBackendEvent) {
  window.electronAPI.onNativeBackendEvent((payload) => {
    try {
      handleLiveAssistPayload(JSON.parse(payload));
    } catch (err) {
      console.warn('[Native Backend Event] Invalid JSON:', err, payload);
    }
  });
}

if (window.electronAPI?.onNativeAudioExit) {
  window.electronAPI.onNativeAudioExit(({ code, signal, expected }) => {
    nativeAudioRunning = false;
    if (isRecording && !expected) {
      isRecording = false;
      btnStart.disabled = false;
      btnStop.disabled = true;
      setStatus('Native audio helper stopped.', 'error');
      showError(
        `Native audio helper stopped. code=${code ?? 'null'} signal=${signal ?? 'null'}. ` +
        'Check microphone permission, Screen & System Audio Recording permission, default headset input, and whether the stream server is running.'
      );
    }
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Source picker modal ───────────────────────────────────────────────────
let selectedSourceId = null;

/**
 * Opens the source picker modal and resolves with:
 *   - A source ID string → user picked a window/screen
 *   - null              → user chose "Skip / Mic only"
 *   - undefined         → user cancelled (don't start recording)
 */
function openSourcePicker() {
  return new Promise(async (resolve) => {
    selectedSourceId = null;
    sourceGrid.innerHTML = '<div class="source-loading">Loading windows…</div>';
    modal.style.display = 'flex';

    // Fetch sources from main process
    let sources = [];
    try {
      sources = await window.electronAPI.getDesktopSources();
    } catch (err) {
      console.error('[Sources] Failed:', err);
    }

    // Render source tiles
    sourceGrid.innerHTML = '';
    if (sources.length === 0) {
      sourceGrid.innerHTML = '<div class="source-loading">No sources found.</div>';
    }

    sources.forEach(src => {
      const tile = document.createElement('div');
      tile.className = 'source-tile';
      tile.dataset.id = src.id;
      tile.innerHTML = `
        <img class="source-thumb" src="${src.thumbnail}" alt="${escapeHtml(src.name)}" />
        <span class="source-name">${escapeHtml(src.name)}</span>
      `;
      tile.addEventListener('click', () => {
        document.querySelectorAll('.source-tile').forEach(t => t.classList.remove('selected'));
        tile.classList.add('selected');
        selectedSourceId = src.id;
        btnPickConfirm.disabled = false;
      });
      sourceGrid.appendChild(tile);
    });

    // Button handlers (one-shot — remove after use)
    function onConfirm() { cleanup(); resolve(selectedSourceId); }
    function onCancel()  { cleanup(); resolve(undefined); }
    function onSkip()    { cleanup(); resolve(null); }

    function cleanup() {
      modal.style.display = 'none';
      btnPickConfirm.removeEventListener('click', onConfirm);
      btnPickCancel.removeEventListener('click', onCancel);
      btnSkipTab.removeEventListener('click', onSkip);
    }

    btnPickConfirm.disabled = true;
    btnPickConfirm.addEventListener('click', onConfirm);
    btnPickCancel.addEventListener('click', onCancel);
    btnSkipTab.addEventListener('click', onSkip);
  });
}

// ── WebSocket ─────────────────────────────────────────────────────────────
function initProxySocket() {
  ensureRecordingSession();
  if (proxySocket && proxySocket.readyState === WebSocket.OPEN) return;

  const wsUrl = PROXY_URL + `?call_id=${encodeURIComponent(currentCallId)}&session_id=${encodeURIComponent(currentSessionId)}`;
  console.log('[Live Assist] Connecting to:', wsUrl);
  setStatus('Connecting to Live Assist stream…', 'connecting');

  try {
    proxySocket = new WebSocket(wsUrl);
  } catch (err) {
    console.error('[Live Assist] WebSocket create failed:', err);
    showError('Could not connect to the Live Assist stream server. Is it running on port 8089?');
    setStatus('Connection failed.', 'error');
    return;
  }

  proxySocket.binaryType = 'arraybuffer';

  proxySocket.onopen = () => {
    console.log('[Live Assist] Connected ✓');
    proxyReady = true;
    hideError();
    setStatus('Connected — listening…', 'listening');
  };

  proxySocket.onmessage = (ev) => {
    let data = ev.data;
    if (typeof data !== 'string') {
      try { data = new TextDecoder().decode(new Uint8Array(data)); } catch {}
    }
    try {
      handleLiveAssistPayload(JSON.parse(data));
    } catch (err) {
      console.warn('[Live Assist] Non-JSON message:', err, data);
    }
  };

  proxySocket.onclose = (ev) => {
    console.warn('[Live Assist] Closed. Code:', ev?.code, 'Reason:', ev?.reason);
    proxyReady = false;
    if (isRecording) {
      setStatus('Connection dropped — reconnecting in 2s…', 'connecting');
      setTimeout(() => initProxySocket(), 2000);
    } else {
      setStatus('Disconnected.', '');
    }
  };

  proxySocket.onerror = (err) => {
    console.error('[Live Assist] WebSocket error:', err);
    proxyReady = false;
    showError('WebSocket error. Make sure the Live Assist stream server is running on port 8089.');
    setStatus('WebSocket error.', 'error');
  };
}

async function sendManualAssistQuestion() {
  const question = assistInput.value.trim();
  if (!question) return;

  ensureRecordingSession();
  hideError();
  assistInput.value = '';
  btnAssistSend.disabled = true;
  setActiveTab('assist');

  createAssistEntry({
    label: 'You',
    text: question,
    role: 'agent',
  });
  const pendingEntry = createAssistEntry({
    label: 'Live Assist',
    text: 'Thinking...',
    role: 'assistant',
    pending: true,
  });

  try {
    const response = await fetch(`${API_URL}/livefeedback/manual_question`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        call_id: currentCallId,
        question,
        source: 'agent_manual_question',
        metadata: { ui_source: 'desktop_live_assist_tab' },
      }),
    });

    if (!response.ok) {
      throw new Error(`Manual question failed (${response.status})`);
    }

    const payload = await response.json();
    const action = payload.action_result || {};
    const answer = action.answer || action.metadata?.error || 'NO_MATCH';
    updateAssistEntry(pendingEntry, {
      label: 'Live Assist',
      text: answer,
      role: 'assistant',
      pending: false,
    });
  } catch (err) {
    console.error('[Live Assist] Manual question failed:', err);
    updateAssistEntry(pendingEntry, {
      label: 'Live Assist',
      text: err.message || 'Manual question failed.',
      role: 'assistant',
      pending: false,
    });
    showError(err.message || 'Manual Live Assist question failed.');
  } finally {
    btnAssistSend.disabled = false;
    assistInput.focus();
  }
}

function closeProxy() {
  if (proxySocket) {
    proxySocket.close(1000, 'Recording stopped');
    proxySocket = null;
  }
  proxyReady = false;
}

function socketStateLabel(socket) {
  if (!socket) return 'none';
  if (socket.readyState === WebSocket.CONNECTING) return 'connecting';
  if (socket.readyState === WebSocket.OPEN) return 'open';
  if (socket.readyState === WebSocket.CLOSING) return 'closing';
  if (socket.readyState === WebSocket.CLOSED) return 'closed';
  return String(socket.readyState);
}

function formatTrackState(track) {
  if (!track) return 'missing';
  return [
    `enabled=${track.enabled}`,
    `muted=${track.muted}`,
    `readyState=${track.readyState}`,
    `label=${track.label || 'unknown'}`,
  ].join(' ');
}

function logTrackState(prefix, stream) {
  const tracks = stream?.getAudioTracks?.() || [];
  if (tracks.length === 0) {
    console.log(`[${prefix} Track] missing`);
    return;
  }
  tracks.forEach((track, index) => {
    console.log(`[${prefix} Track] index=${index} ${formatTrackState(track)}`);
  });
}

function attachTrackDiagnostics(prefix, stream) {
  const tracks = stream?.getAudioTracks?.() || [];
  tracks.forEach((track, index) => {
    track.addEventListener('mute', () => {
      console.warn(`[${prefix} Track] muted index=${index} ${formatTrackState(track)}`);
    });
    track.addEventListener('unmute', () => {
      console.log(`[${prefix} Track] unmuted index=${index} ${formatTrackState(track)}`);
    });
    track.addEventListener('ended', () => {
      console.warn(`[${prefix} Track] ended index=${index} ${formatTrackState(track)}`);
      if (isRecording) {
        console.warn(`[${prefix} Track] capture ended during active recording`);
        logCaptureBreak(prefix, 'ended');
        if (prefix === 'Tab') {
          detachTabCapture('track_ended');
          return;
        }
        const message = `${prefix} capture track ended while recording. Try mic-only mode, or restart the selected audio source.`;
        showError(message);
        setStatus(`${prefix} audio capture ended.`, 'error');
      }
    });
  });
  logTrackState(prefix, stream);
}

function startTrackStatusLogging() {
  if (trackStatusTimer) clearInterval(trackStatusTimer);
  trackStatusTimer = setInterval(() => {
    if (!isRecording) return;
    logTrackState('Mic', micStream);
    logTrackState('Tab', tabStream);
  }, 5000);
}

function stopTrackStatusLogging() {
  if (!trackStatusTimer) return;
  clearInterval(trackStatusTimer);
  trackStatusTimer = null;
}

function calculateFloatRms(samples) {
  if (!samples || samples.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSquares += samples[i] * samples[i];
  }
  return Math.sqrt(sumSquares / samples.length);
}

function calculateFloatPeak(samples) {
  if (!samples || samples.length === 0) return 0;
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = Math.abs(samples[i]);
    if (value > peak) peak = value;
  }
  return peak;
}

function firstAudioTrackState(stream) {
  const track = stream?.getAudioTracks?.()[0];
  return track ? track.readyState : 'missing';
}

function countTracks(stream, kind) {
  if (!stream) return 0;
  if (kind === 'audio') return stream.getAudioTracks?.().length || 0;
  if (kind === 'video') return stream.getVideoTracks?.().length || 0;
  return stream.getTracks?.().length || 0;
}

function logAllStreamTracks(prefix, stream) {
  if (!stream) {
    console.log(`[${prefix} Stream] missing`);
    return;
  }

  console.log(
    `[${prefix} Stream] active=${stream.active} ` +
    `audio_tracks=${countTracks(stream, 'audio')} ` +
    `video_tracks=${countTracks(stream, 'video')}`
  );

  stream.getTracks().forEach((track, index) => {
    console.log(
      `[${prefix} Stream Track] index=${index} kind=${track.kind} ` +
      `${formatTrackState(track)}`
    );
  });
}

async function logMediaDevices(reason) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    console.log(`[Media Devices] reason=${reason} unavailable`);
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    devices.forEach((device, index) => {
      console.log(
        `[Media Device] reason=${reason} index=${index} ` +
        `kind=${device.kind} label=${device.label || 'unlabeled'}`
      );
    });
  } catch (err) {
    console.warn(`[Media Devices Error] reason=${reason} ${err.name}: ${err.message}`);
  }
}

function logCaptureBreak(prefix, eventName) {
  console.warn(
    `[Capture Break] source=${prefix} event=${eventName} recording=${isRecording} ` +
    `audio_context=${audioCtx?.state || 'missing'} socket=${socketStateLabel(proxySocket)} ` +
    `sent_chunks=${sentAudioChunks} skipped_chunks=${skippedWorkletChunks} ` +
    `packet_seq=${capturePacketSeq} mic_stream_active=${micStream?.active ?? 'missing'} ` +
    `tab_stream_active=${tabStream?.active ?? 'missing'}`
  );
  logAllStreamTracks('Mic', micStream);
  logAllStreamTracks('Tab', tabStream);
  logMediaDevices(`capture_break_${prefix}_${eventName}`);
}

function detachTabCapture(reason) {
  console.warn(`[Tab Capture Detached] reason=${reason}`);
  if (tabSourceNode) {
    try {
      tabSourceNode.disconnect();
    } catch (err) {
      console.warn(`[Tab Capture Detach Error] disconnect_source=${err.message}`);
    }
    tabSourceNode = null;
  }
  if (tabStream) {
    tabStream.getTracks().forEach((track) => {
      if (track.readyState !== 'ended') {
        track.stop();
      }
    });
    tabStream = null;
  }
  console.warn('[Tab Capture Detached] continuing mic-only capture');
  showError('System audio capture ended. Continuing mic-only transcription.');
  setStatus('System audio ended — continuing mic-only.', 'listening');
}

if (navigator.mediaDevices) {
  navigator.mediaDevices.ondevicechange = () => {
    console.warn('[Media Devices] devicechange');
    logMediaDevices('devicechange');
  };
}

// ── Start recording ────────────────────────────────────────────────────────
async function startRecording() {
  if (isRecording) return;
  hideError();
  if (!runtimeConfig.audioCaptureMode) {
    await loadRuntimeConfig();
  }
  ensureRecordingSession();

  if (isNativeCaptureMode()) {
    if (!window.electronAPI?.startNativeAudio) {
      showError('Native audio capture is not available in this desktop build.');
      setStatus('Native audio unavailable.', 'error');
      return;
    }

    setStatus('Starting native desktop audio capture…', 'connecting');
    try {
      const result = await window.electronAPI.startNativeAudio({
        wsUrl: PROXY_URL,
        callId: currentCallId,
        sessionId: currentSessionId,
        includeOwnAudio: NATIVE_INCLUDE_OWN_AUDIO,
        chunkFrames: NATIVE_CHUNK_FRAMES,
        diagnostics: DIAGNOSTICS_ENABLED,
      });
      if (!result?.ok) {
        throw new Error(result?.error || 'Native audio helper did not start.');
      }

      isRecording = true;
      nativeAudioRunning = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      setStatus('Native capture starting…', 'connecting');
      console.log('[Recorder] Started ✓ — mode: Native Mic + Native System Audio');
    } catch (err) {
      console.error('[Native Audio] start failed:', err);
      showError(err.message || 'Native audio capture failed to start.');
      setStatus('Native audio failed.', 'error');
    }
    return;
  }

  // ── Step 1: Show source picker ─────────────────────────────────────────
  let sourceId = null;
  if (FORCE_MIC_ONLY_DIAGNOSTIC) {
    console.warn('[Capture Mode] FORCE_MIC_ONLY_DIAGNOSTIC=true — bypassing source picker and desktop capture');
    setStatus('Starting forced mic-only diagnostic…', 'connecting');
  } else {
    setStatus('Select the customer audio source…', 'connecting');
    sourceId = await openSourcePicker();
  }

  if (sourceId === undefined) {
    // User cancelled
    setStatus('Ready. Click "Start Listening" to begin.', '');
    return;
  }

  // ── Step 2: Request microphone ─────────────────────────────────────────
  setStatus('Requesting microphone access…', 'connecting');
  const micConstraints = {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl:  true,
    }
  };
  try {
    await logMediaDevices('before_mic_getUserMedia');
    console.log(`[Mic Capture Start] constraints=${JSON.stringify(micConstraints)}`);
    micStream = await navigator.mediaDevices.getUserMedia(micConstraints);
    console.log(
      `[Mic Capture OK] audio_tracks=${countTracks(micStream, 'audio')} ` +
      `video_tracks=${countTracks(micStream, 'video')}`
    );
    console.log('[Mic] Stream acquired ✓', micStream.getAudioTracks()[0].getSettings());
    logAllStreamTracks('Mic', micStream);
    attachTrackDiagnostics('Mic', micStream);
  } catch (err) {
    console.error('[Mic] getUserMedia failed:', err);
    showError(`Microphone access denied: ${err.name} — ${err.message}`);
    setStatus('Microphone access denied.', 'error');
    return;
  }

  // ── Step 3: Capture tab/system audio (if user selected a source) ───────
    if (sourceId) {
    setStatus('Capturing system audio…', 'connecting');
    const tabConstraints = {
      audio: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: sourceId,
        }
      },
      video: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: sourceId,
        }
      }
    };
    try {
      // Some Electron/macOS desktop captures require the video track to stay
      // alive for the paired system-audio track to keep streaming.
      console.log(`[Tab Capture Start] source_id=${sourceId} constraints=${JSON.stringify(tabConstraints)}`);
      tabStream = await navigator.mediaDevices.getUserMedia(tabConstraints);

      console.log(
        `[Tab Capture OK] audio_tracks=${countTracks(tabStream, 'audio')} ` +
        `video_tracks=${countTracks(tabStream, 'video')}`
      );
      console.log('[Tab] System audio stream acquired ✓');
      logAllStreamTracks('Tab', tabStream);
      attachTrackDiagnostics('Tab', tabStream);
    } catch (err) {
      console.warn('[Tab] System audio capture failed:', err.name, err.message);
      // Non-fatal: fall back to mic-only mode
      showError(
        `System audio capture failed (${err.name}). Running mic-only mode. ` +
        'On macOS, enable Screen & System Audio Recording permission for this app/Terminal, then restart the desktop app.'
      );
      tabStream = null;
    }
  } else {
    console.log(`[Tab] Skipped — ${FORCE_MIC_ONLY_DIAGNOSTIC ? 'forced diagnostic mic-only mode' : 'mic-only mode'}`);
    tabStream = null;
    logTrackState('Tab', tabStream);
  }

  if (!FORCE_MIC_ONLY_DIAGNOSTIC && tabStream) {
    logTrackState('Mic', micStream);
    logTrackState('Tab', tabStream);
  }

  // ── Step 4: AudioContext ───────────────────────────────────────────────
  try {
    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioCtx.onstatechange = () => {
      console.log('[Audio] AudioContext state changed:', audioCtx?.state || 'closed');
    };
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    console.log('[Audio] AudioContext state:', audioCtx.state, '@ sampleRate:', audioCtx.sampleRate);
  } catch (err) {
    console.error('[Audio] AudioContext error:', err);
    showError(`AudioContext failed: ${err.message}`);
    setStatus('Audio setup failed.', 'error');
    return;
  }

  // ── Step 5: Load AudioWorklet ─────────────────────────────────────────
  try {
    await audioCtx.audioWorklet.addModule('audio-processor.js');
    console.log('[Audio] AudioWorklet loaded ✓');
  } catch (err) {
    console.error('[Audio] AudioWorklet load failed:', err);
    showError(`AudioWorklet failed: ${err.message}`);
    setStatus('Audio worklet error.', 'error');
    return;
  }

  // ── Step 6: Build audio graph ─────────────────────────────────────────
  //
  //  tabSource  ──► processor (input 0)  ← Customer (left  ch)
  //  micSource  ──► processor (input 1)  ← Manager  (right ch)
  //
  // The worklet merges them and posts { tabChunk, micChunk } back here.

  micSourceNode = audioCtx.createMediaStreamSource(micStream);

  // Manvendra-compatible capture topology: two inputs, no outputs.
  processor = new AudioWorkletNode(audioCtx, 'mic-processor', {
    numberOfInputs:  2,
    numberOfOutputs: 0,
    channelCount:    1,
    channelCountMode: 'explicit',
  });
  processor.onprocessorerror = (event) => {
    console.error('[Audio] AudioWorklet processor error:', event);
    showError('Audio processor crashed. Stop and start listening again.');
    setStatus('Audio processor error.', 'error');
  };

  // Connect mic to input 1 (Manager / right channel)
  micSourceNode.connect(processor, 0, 1);

  // Connect tab audio to input 0 (Customer / left channel) if available
  if (tabStream && tabStream.getAudioTracks().length > 0) {
    tabSourceNode = audioCtx.createMediaStreamSource(tabStream);
    tabSourceNode.connect(processor, 0, 0);
    console.log('[Audio] Tab audio connected to left channel ✓');
  } else {
    tabSourceNode = null;
    console.log('[Audio] No tab audio — left channel will be silence');
  }

  // ── Step 7: Handle PCM chunks from the worklet ───────────────────────
  processor.port.onmessage = (ev) => {
    const message = ev.data || {};
    const { micChunk, tabChunk } = message;
    if (!micChunk || !tabChunk) {
      console.warn('[Audio] Worklet sent malformed chunk', message);
      return;
    }

    capturePacketSeq += 1;
    const socketState = socketStateLabel(proxySocket);
    const micRms = calculateFloatRms(micChunk);
    const tabRms = calculateFloatRms(tabChunk);
    const micPeak = calculateFloatPeak(micChunk);
    const tabPeak = calculateFloatPeak(tabChunk);
    console.log(
      `[Capture Packet] seq=${capturePacketSeq} socket=${socketState} ` +
      `mic_rms=${micRms.toFixed(4)} mic_peak=${micPeak.toFixed(4)} ` +
      `tab_rms=${tabRms.toFixed(4)} tab_peak=${tabPeak.toFixed(4)} ` +
      `mic_state=${firstAudioTrackState(micStream)} ` +
      `tab_state=${firstAudioTrackState(tabStream)}`
    );

    if (!proxySocket || proxySocket.readyState !== WebSocket.OPEN) {
      skippedWorkletChunks += 1;
      console.warn(`[WS Skip] seq=${capturePacketSeq} socket=${socketState}`);
      return;
    }

    // Build stereo interleaved PCM: [L0, R0, L1, R1, …]
    // L = Customer (tab/system audio), R = Manager (mic)
    const interleaved = new Int16Array(micChunk.length * 2);
    for (let i = 0; i < micChunk.length; i++) {
      const tabS  = Math.max(-1, Math.min(1, tabChunk[i]));
      const micS  = Math.max(-1, Math.min(1, micChunk[i]));
      interleaved[i * 2]     = tabS < 0 ? tabS * 0x8000 : tabS * 0x7FFF; // Left  = Customer
      interleaved[i * 2 + 1] = micS < 0 ? micS * 0x8000 : micS * 0x7FFF; // Right = Manager
    }

    try {
      proxySocket.send(interleaved.buffer);
      sentAudioChunks += 1;
      console.log(`[WS Send] seq=${capturePacketSeq} chunk=${sentAudioChunks} bytes=${interleaved.byteLength}`);
    } catch (err) {
      console.error(`[WS Send Error] seq=${capturePacketSeq} error=${err.message}`);
    }
  };

  // ── Step 8: Connect WebSocket ─────────────────────────────────────────
  initProxySocket();

  isRecording = true;
  startTrackStatusLogging();
  btnStart.disabled = true;
  btnStop.disabled  = false;

  const mode = tabStream ? 'Mic + System Audio (dual-channel)' : 'Mic only (single-channel)';
  console.log(`[Recorder] Started ✓ — mode: ${mode}`);
}

// ── Stop recording ─────────────────────────────────────────────────────────
function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  stopTrackStatusLogging();

  if (nativeAudioRunning) {
    nativeAudioRunning = false;
    window.electronAPI?.stopNativeAudio?.().catch((err) => {
      console.warn('[Native Audio] stop failed:', err);
    });
  }

  if (processor) {
    processor.port.onmessage = null;
    processor.disconnect();
    processor = null;
  }
  if (micSourceNode) {
    micSourceNode.disconnect();
    micSourceNode = null;
  }
  if (tabSourceNode) {
    tabSourceNode.disconnect();
    tabSourceNode = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (tabStream) {
    tabStream.getTracks().forEach(t => t.stop());
    tabStream = null;
  }

  closeProxy();
  sentAudioChunks = 0;
  skippedWorkletChunks = 0;
  capturePacketSeq = 0;

  btnStart.disabled = false;
  btnStop.disabled  = true;
  setStatus('Stopped. Click "Start Listening" to begin again.', '');
  console.log('[Recorder] Stopped ✓');
  currentCallId = '';
  currentSessionId = '';
}

// ── Button handlers ───────────────────────────────────────────────────────
btnStart.addEventListener('click', startRecording);
btnStop.addEventListener('click', stopRecording);
tabButtons.forEach((button) => {
  button.addEventListener('click', () => setActiveTab(button.dataset.tab));
});
btnAssistSend.addEventListener('click', sendManualAssistQuestion);
assistInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendManualAssistQuestion();
  }
});

btnClear.addEventListener('click', () => {
  if (activeTab === 'assist') {
    assistFeed.innerHTML = '';
    assistFeed.appendChild(Object.assign(document.createElement('div'), {
      className: 'empty-state',
      id: 'assist-empty-state',
      innerHTML: 'Live suggestions will appear here<br/><span style="font-size:12px;opacity:0.6">Real-time prompts, objection handlers, and answers — newest at the bottom.</span>'
    }));
    return;
  }

  transcript.innerHTML = '';
  liveTranscriptEntries.clear();
  lastFinalTranscript.clear();
  activeTranscriptSegments.clear();
  transcript.appendChild(Object.assign(document.createElement('div'), {
    className: 'empty-state',
    id:        'empty-state',
    innerHTML: 'Transcript cleared.<br/><span style="font-size:12px;opacity:0.6">Listening continues…</span>'
  }));
});

loadRuntimeConfig();
