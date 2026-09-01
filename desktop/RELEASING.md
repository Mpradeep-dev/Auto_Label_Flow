# Releasing the desktop app

The desktop build is a Windows-only, unsigned NSIS installer that bundles the
FastAPI backend (as an embedded CPython 3.12 + site-packages), the built SPA,
and a static `ffmpeg.exe`. The in-app **Settings → Application update → Check
for updates** button pulls new versions from **public GitHub Releases** on
`github.com/Mpradeep-dev/Auto_Label_Flow` via `electron-updater`.

## One-time setup on the build machine

- Windows 10/11 x64
- Node 20+ and npm
- `curl` and `tar` on `PATH` (built into Windows 10 1803+)
- `cd desktop && npm install`
- Put a static LGPL `ffmpeg.exe` at `desktop/build/bin/ffmpeg.exe`
  (https://www.gyan.dev/ffmpeg/builds/ — "release essentials"). OpenCV bundles
  its own FFmpeg DLLs; this is a fallback for codecs/containers it misses.
- Bundle the MSVC runtime: copy `vc_redist.x64.exe` (or the `vcruntime140*.dll`
  set) into `desktop/build/bin/` so torch's DLLs load on a clean machine.
- `GH_TOKEN` env var with `repo` scope, for `--publish always`.

## Cut a release

1. Bump the version in **`desktop/package.json`** (single source of truth —
   it drives `electron-updater` and, via `ALF_APP_VERSION`, the backend's
   `/api/v1/health` + `/api/v1/system/info`).
2. Commit, then tag: `git tag vX.Y.Z && git push --tags` (convention matches
   the existing `v0.2.0`).
3. `cd desktop`
4. `node scripts/build.mjs` — assembles `desktop/build/` (frontend with
   `VITE_DESKTOP=1`, embedded Python + `requirements-desktop.txt`, backend
   source, version stamp). ~10–20 min the first time (downloads
   python-build-standalone + CPU torch).
5. `npx electron-builder --win nsis --publish always` — builds
   `desktop/dist/AutoLabelFlow-Setup.exe` (versionless on purpose, so the
   README's `releases/latest/download/AutoLabelFlow-Setup.exe` button keeps
   working) and uploads it plus `latest.yml` to the GitHub Release for the tag.

`npm run dist` does steps 4–5 with `--publish never` (local artifact only);
`npm run release` does them with `--publish always`.

## Optional add-on packs

The GPU-training and cloud-integrations packs are **not** part of the
installer — the base install is CPU-only. They are downloaded on demand from
**Settings → Desktop app** and installed with `pip install --target` into
`%APPDATA%/AutoLabelFlow/packs/<name>/`. Their pinned specs live in
`backend/app/workers/tasks/packs.py`; no separate release artifact is needed.

## What ships where

| Path in installer | Source |
|---|---|
| `resources/python/` | `desktop/build/python/` (CPython + site-packages + `app-root/` backend) |
| `resources/frontend/` | `desktop/build/frontend/` (Vite `dist`) |
| `resources/bin/` | `desktop/build/bin/` (`ffmpeg.exe`, vcredist) |
| `app.asar` | `main.js`, `preload.js`, `package.json` |

All user data (SQLite DB, media, model artifacts, logs, add-on packs) lives
under `%APPDATA%/AutoLabelFlow/` and survives updates and uninstall
(`deleteAppDataOnUninstall: false`).

## Observed build output (v0.2.0, first build)

| | |
|---|---|
| `AutoLabelFlow-Setup.exe` | ~438 MB |
| `win-unpacked/` (installed footprint) | ~2.3 GB |
| Bundled Python (`build/python`) | ~1.9 GB — torch CPU, OpenCV (×2: headless + the copy ultralytics pulls), matplotlib, polars, numpy, sympy |

Trim levers if the installer needs to be smaller: use the
`install_only_stripped` python-build-standalone asset; drop `matplotlib`
(only ultralytics training plots need it); carefully remove `opencv-python`
and keep `opencv-python-headless`. The `.pdb` / `__pycache__` strip is
already in `build.mjs`.

## Notes

- **Unsigned** — `win.signAndEditExecutable: false` in `electron-builder.yml`
  skips the exe resource-edit/sign pass; this also sidesteps electron-builder's
  `winCodeSign` helper, whose archive carries macOS symlinks that fail to
  extract on Windows without Developer Mode / an elevated shell. Windows
  SmartScreen warns on first run ("More info → Run anyway"). Auto-update still
  works. To sign later, remove that line and add `win.certificateFile` /
  `certificatePassword` (or an EV token config).
- **No app icon** — electron-builder falls back to the default Electron icon.
  Add `desktop/build/icon.ico` (256×256) to brand it.
- To move updates off public GitHub Releases (the all-rights-reserved LICENSE
  makes the installer publicly downloadable): switch `publish.provider` to
  `generic` with a `url` you host, and point `autoUpdater` at it. Config-only.
