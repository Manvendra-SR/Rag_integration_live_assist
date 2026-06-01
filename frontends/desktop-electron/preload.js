// preload.js — Bridge between Main Process and Renderer

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // App version for the UI label
  version: process.versions.electron,
  getRuntimeConfig: () => ipcRenderer.invoke('get-runtime-config'),

  // Ask main process for the list of capturable windows/screens.
  // Returns: Array<{ id, name, thumbnail (dataURL), appIcon (dataURL|null) }>
  getDesktopSources: () => ipcRenderer.invoke('get-desktop-sources'),

  startNativeAudio: (options) => ipcRenderer.invoke('start-native-audio', options),
  stopNativeAudio: () => ipcRenderer.invoke('stop-native-audio'),
  onNativeAudioLog: (callback) => ipcRenderer.on('native-audio-log', (_event, line) => callback(line)),
  onNativeBackendEvent: (callback) => ipcRenderer.on('native-backend-event', (_event, payload) => callback(payload)),
  onNativeAudioExit: (callback) => ipcRenderer.on('native-audio-exit', (_event, payload) => callback(payload)),
});
