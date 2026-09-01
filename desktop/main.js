"use strict";
/**
 * AutoLabelFlow desktop shell.
 *
 * Boot sequence:
 *   1. spawn the bundled Python backend (uvicorn) on a free 127.0.0.1 port,
 *      rooted at %APPDATA%/AutoLabelFlow, `local` task runtime, serving the
 *      built SPA itself
 *   2. poll /api/v1/health until {"status":"ok"}
 *   3. open the window at http://127.0.0.1:<port>/
 *   4. on quit, terminate the backend child
 *
 * Updates are manual: the renderer calls desktop:check-for-updates, then
 * desktop:download-update, then desktop:quit-and-install (electron-updater,
 * public GitHub Releases).
 */
const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { autoUpdater } = require("electron-updater");
const { spawn } = require("node:child_process");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");
const fs = require("node:fs");

const isPackaged = app.isPackaged;
// In a packaged build, extraResources land in process.resourcesPath.
// In dev (`npm start` in desktop/), point at the sibling repo checkout.
const resRoot = isPackaged ? process.resourcesPath : path.join(__dirname, "..");

const PATHS = isPackaged
  ? {
      python: path.join(resRoot, "python", process.platform === "win32" ? "python.exe" : "bin/python3"),
      backendCwd: path.join(resRoot, "python", "app-root"),
      frontendDist: path.join(resRoot, "frontend"),
      binDir: path.join(resRoot, "bin"),
    }
  : {
      python: path.join(resRoot, "backend", "venv", "Scripts", "python.exe"),
      backendCwd: path.join(resRoot, "backend"),
      frontendDist: path.join(resRoot, "frontend", "dist"),
      binDir: path.join(resRoot, "desktop", "build", "bin"),
    };

const DATA_DIR = app.getPath("userData");
let backendProc = null;
let backendPort = 0;
let win = null;

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function startBackend(port) {
  const env = {
    ...process.env,
    ALF_DATA_DIR: DATA_DIR,
    ALF_TASK_QUEUE: "local",
    ALF_APP_VERSION: app.getVersion(),
    FRONTEND_DIST_DIR: PATHS.frontendDist,
    PYTHONUNBUFFERED: "1",
    PYTHONDONTWRITEBYTECODE: "1",
  };
  if (fs.existsSync(PATHS.binDir)) {
    env.PATH = PATHS.binDir + path.delimiter + (env.PATH || "");
  }

  const args = ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port)];
  backendProc = spawn(PATHS.python, args, { cwd: PATHS.backendCwd, env, windowsHide: true });

  const logDir = path.join(DATA_DIR, "logs");
  fs.mkdirSync(logDir, { recursive: true });
  const logStream = fs.createWriteStream(path.join(logDir, "backend.log"), { flags: "a" });
  backendProc.stdout.pipe(logStream);
  backendProc.stderr.pipe(logStream);

  backendProc.on("exit", (code) => {
    backendProc = null;
    if (code && code !== 0 && !app.isQuitting) {
      dialog.showErrorBox("AutoLabelFlow", `The backend process exited unexpectedly (code ${code}).`);
      app.quit();
    }
  });
}

function waitForHealth(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(
        { host: "127.0.0.1", port, path: "/api/v1/health", timeout: 2000 },
        (res) => {
          let body = "";
          res.on("data", (c) => (body += c));
          res.on("end", () => {
            try {
              if (res.statusCode === 200 && JSON.parse(body).status === "ok") return resolve();
            } catch (_) {}
            retry();
          });
        },
      );
      req.on("error", retry);
      req.on("timeout", () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() > deadline) return reject(new Error("backend did not become healthy in time"));
      setTimeout(tick, 400);
    };
    tick();
  });
}

function createWindow(port) {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#ffffff",
    icon: path.join(__dirname, "icon.ico"),
    show: false,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false },
  });
  win.removeMenu();
  win.loadURL(`http://127.0.0.1:${port}/`);
  win.once("ready-to-show", () => win.show());
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

// --- auto-update (manual trigger from the renderer) ---
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = true;
function wireUpdaterEvents() {
  const send = (ch, payload) => win && win.webContents.send(ch, payload);
  autoUpdater.on("update-available", (i) =>
    send("update:available", { version: i.version, releaseNotes: i.releaseNotes, releaseName: i.releaseName }),
  );
  autoUpdater.on("download-progress", (p) =>
    send("update:progress", {
      percent: p.percent,
      transferred: p.transferred,
      total: p.total,
      bytesPerSecond: p.bytesPerSecond,
    }),
  );
  autoUpdater.on("update-downloaded", (i) => send("update:downloaded", { version: i.version }));
  autoUpdater.on("error", (e) => send("update:error", String(e && e.message ? e.message : e)));
}

ipcMain.handle("desktop:get-version", () => app.getVersion());
ipcMain.handle("desktop:check-for-updates", async () => {
  const r = await autoUpdater.checkForUpdates();
  const available = !!(r && r.updateInfo && r.updateInfo.version !== app.getVersion());
  return {
    updateAvailable: available,
    info: available
      ? { version: r.updateInfo.version, releaseNotes: r.updateInfo.releaseNotes, releaseName: r.updateInfo.releaseName }
      : undefined,
  };
});
ipcMain.handle("desktop:download-update", () => autoUpdater.downloadUpdate());
ipcMain.on("desktop:quit-and-install", () => {
  app.isQuitting = true;
  autoUpdater.quitAndInstall();
});

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(async () => {
    try {
      backendPort = await findFreePort();
      startBackend(backendPort);
      await waitForHealth(backendPort);
      wireUpdaterEvents();
      createWindow(backendPort);
    } catch (err) {
      dialog.showErrorBox("AutoLabelFlow", `Failed to start:\n${err && err.message ? err.message : err}`);
      app.quit();
    }
  });

  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", () => {
    app.isQuitting = true;
    if (backendProc) {
      try {
        backendProc.kill();
      } catch (_) {}
    }
  });
}
