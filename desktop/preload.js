"use strict";
const { contextBridge, ipcRenderer } = require("electron");

function on(channel, cb) {
  const listener = (_evt, payload) => cb(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld("desktop", {
  getVersion: () => ipcRenderer.invoke("desktop:get-version"),
  checkForUpdates: () => ipcRenderer.invoke("desktop:check-for-updates"),
  downloadUpdate: () => ipcRenderer.invoke("desktop:download-update"),
  quitAndInstall: () => ipcRenderer.send("desktop:quit-and-install"),
  onUpdateAvailable: (cb) => on("update:available", cb),
  onDownloadProgress: (cb) => on("update:progress", cb),
  onUpdateDownloaded: (cb) => on("update:downloaded", cb),
  onUpdateError: (cb) => on("update:error", cb),
});
