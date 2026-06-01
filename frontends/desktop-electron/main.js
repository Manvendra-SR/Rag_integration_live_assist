// main.js — Electron Main Process

const { app, BrowserWindow, session, ipcMain, desktopCapturer } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

require('dotenv').config({
  path: path.resolve(__dirname, '../../.env'),
});

let nativeAudioProcess = null;
let nativeAudioStopping = false;

function runtimeConfig() {
  const audioCaptureMode = process.env.DESKTOP_AUDIO_CAPTURE_MODE || process.env.INPUT_SOURCE || 'desktop_native_stable';
  return {
    inputSource: process.env.INPUT_SOURCE || 'desktop_native_stable',
    audioCaptureMode,
    wsUrl: process.env.DESKTOP_STREAM_WS_URL || `ws://127.0.0.1:${process.env.PYTHON_WS_PORT || '8089'}`,
    apiUrl: process.env.DESKTOP_API_URL || `http://127.0.0.1:${process.env.API_PORT || '8000'}`,
    includeOwnAudio: process.env.DESKTOP_NATIVE_INCLUDE_OWN_AUDIO !== 'false',
    nativeChunkFrames: Number(process.env.DESKTOP_NATIVE_CHUNK_FRAMES || '4096'),
    diagnosticsEnabled:
      process.env.LIVE_ASSIST_DIAGNOSTICS_ENABLED === 'true' ||
      audioCaptureMode === 'desktop_native_diagnostic',
  };
}

function stopNativeAudio() {
  if (!nativeAudioProcess) return;
  console.log('[Native Audio] stopping helper');
  nativeAudioStopping = true;
  nativeAudioProcess.kill('SIGTERM');
}

function createWindow() {
  const win = new BrowserWindow({
    width: 900,
    height: 700,
    title: 'Sales Assistant — Prototype',
    backgroundColor: '#0f0f13',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });

  win.once('ready-to-show', () => {
    win.show();
  });

  // ── Grant microphone + desktop-capture permission automatically ──────
  session.defaultSession.setPermissionRequestHandler(
    (webContents, permission, callback) => {
      // Allow microphone and display-media (desktop capture)
      if (permission === 'media' || permission === 'display-media') {
        console.log(`[Main] Granting permission: ${permission}`);
        callback(true);
      } else {
        callback(false);
      }
    }
  );

  // ── IPC: return list of capturable windows/screens to the renderer ───
  // desktopCapturer is a main-process-only API; we expose it via IPC.
  ipcMain.handle('get-runtime-config', async () => runtimeConfig());

  ipcMain.handle('get-desktop-sources', async () => {
    try {
      const sources = await desktopCapturer.getSources({
        types: ['window', 'screen'],
        thumbnailSize: { width: 240, height: 135 },
        fetchWindowIcons: true,
      });

      // Serialise only the fields the renderer needs
      // (Electron NativeImage objects can't cross the context bridge directly)
      return sources.map(src => ({
        id:        src.id,
        name:      src.name,
        thumbnail: src.thumbnail.toDataURL(),  // base64 PNG → safe to send
        appIcon:   src.appIcon ? src.appIcon.toDataURL() : null,
      }));
    } catch (err) {
      console.error('[Main] desktopCapturer error:', err);
      return [];
    }
  });

  ipcMain.handle('start-native-audio', async (event, options = {}) => {
    if (nativeAudioProcess) {
      return { ok: true, alreadyRunning: true };
    }

    const nativeAudioDir = path.resolve(__dirname, '../../native-audio-capture/macos');
    const wsUrl = options.wsUrl || 'ws://127.0.0.1:8089';
    const callId = options.callId || `desktop-call-${Date.now()}`;
    const sessionId = options.sessionId || `desktop-session-${Date.now()}`;
    const includeOwnAudio = options.includeOwnAudio !== false;
    const chunkFrames = Number(options.chunkFrames || process.env.DESKTOP_NATIVE_CHUNK_FRAMES || '4096');
    const diagnostics = options.diagnostics === true;
    const args = [
      'run',
      'native-audio-streamer',
      '--ws-url',
      wsUrl,
      '--call-id',
      callId,
      '--session-id',
      sessionId,
      '--chunk-frames',
      String(chunkFrames),
    ];
    if (!includeOwnAudio) {
      args.push('--exclude-own-audio');
    }
    if (diagnostics) {
      args.push('--diagnostics');
    }

    console.log('[Native Audio] starting:', `swift ${args.join(' ')}`);
    nativeAudioStopping = false;
    nativeAudioProcess = spawn('swift', args, {
      cwd: nativeAudioDir,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    const sendLine = (line) => {
      if (!line.trim()) return;
      console.log('[Native Audio]', line);
      win.webContents.send('native-audio-log', line);

      const backendPrefix = '[Native Backend Event] ';
      if (line.startsWith(backendPrefix)) {
        win.webContents.send('native-backend-event', line.slice(backendPrefix.length));
      }
    };

    let stdoutBuffer = '';
    nativeAudioProcess.stdout.on('data', (chunk) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || '';
      lines.forEach(sendLine);
    });

    let stderrBuffer = '';
    nativeAudioProcess.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() || '';
      lines.forEach((line) => {
        if (!line.trim()) return;
        console.error('[Native Audio stderr]', line);
        win.webContents.send('native-audio-log', `[stderr] ${line}`);
      });
    });

    nativeAudioProcess.on('exit', (code, signal) => {
      console.log('[Native Audio] exited:', { code, signal });
      win.webContents.send('native-audio-exit', { code, signal, expected: nativeAudioStopping });
      nativeAudioProcess = null;
      nativeAudioStopping = false;
    });

    nativeAudioProcess.on('error', (err) => {
      console.error('[Native Audio] failed:', err);
      win.webContents.send('native-audio-log', `[Native Error] ${err.message}`);
      nativeAudioProcess = null;
    });

    return { ok: true };
  });

  ipcMain.handle('stop-native-audio', async () => {
    stopNativeAudio();
    return { ok: true };
  });

  win.loadFile(path.join(__dirname, 'index.html'));

  win.webContents.on('render-process-gone', (event, details) => {
    console.error('Renderer crashed:', details);
  });

  win.webContents.on('did-fail-load', (event, code, desc) => {
    console.error('Load failed:', code, desc);
  });

  win.webContents.on('console-message', (e, level, message) => {
    console.log('[Renderer]', message);
  });

  // Open DevTools for debugging — comment out when done
  win.webContents.openDevTools({ mode: 'detach' });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  stopNativeAudio();
  app.quit();
});
