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

Versioning is automated by **release-please** ([`.github/workflows/release-please.yml`](../.github/workflows/release-please.yml)).
Commit messages on `main` must follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:` bumps the minor, `fix:` bumps the patch, `feat!:` / `BREAKING CHANGE:`
bumps the minor while pre-1.0. `chore/docs/refactor/test/ci` do not release.

1. **Merge feat/fix PRs.** release-please keeps one open "release PR" that
   accumulates the pending `CHANGELOG.md` entry and version bump.
2. **Merge the release PR.** That commits the changelog, bumps
   `version.txt` + **`desktop/package.json`** (the version
   `electron-updater` and, via `ALF_APP_VERSION`, the backend's
   `/api/v1/health` + `/api/v1/system/info` read), and creates the
   `vX.Y.Z` tag + a GitHub Release — **notes only, no installer yet**.
3. **Build + publish the installer for that tag.** Actions → *Release
   desktop app* → **Run workflow**, tag = `vX.Y.Z`. It runs
   `scripts/build.mjs` then `electron-builder --win nsis --publish always`
   on `windows-latest` and uploads `AutoLabelFlow-Setup.exe` (versionless
   on purpose, so the README's `releases/latest/download/...` button keeps
   working) plus `latest.yml` to the Release release-please just made.
   Only now does the in-app **Check for updates** button see the new
   version.

`.release-please-manifest.json` and `version.txt` at the repo root are
release-please's version anchors; don't hand-edit them. To build a local
artifact without any of this: `cd desktop && npm run dist` runs
`scripts/build.mjs` + `electron-builder … --publish never`.

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

- **Unsigned** — there is no `win.certificateFile`, so the exe/installer are
  not code-signed. Windows SmartScreen warns on first run ("More info → Run
  anyway"). Auto-update still works. To sign, add `win.certificateFile` /
  `certificatePassword` (or an EV token config).
- **exe icon / rcedit** — `win.icon: icon.ico` is set, so the release build
  embeds the app icon and version strings into `AutoLabelFlow.exe` via
  electron-builder's rcedit pass. That pass pulls the `winCodeSign` helper,
  whose archive carries macOS symlinks that only extract on Windows with
  admin / **Developer Mode**. The *Release desktop app* workflow runs on
  `windows-latest` (admin) so it just works; for a **local** `npm run dist`,
  turn on Windows Developer Mode first (Settings → Privacy & security → For
  developers) or the build fails unpacking `winCodeSign`.
- **App icon source** — `desktop/icon.ico` is generated from
  `frontend/public/favicon.png` by `scripts/make-icon.py` (7 frames,
  16→256 px). Re-run that script if the favicon changes.
- To move updates off public GitHub Releases (the all-rights-reserved LICENSE
  makes the installer publicly downloadable): switch `publish.provider` to
  `generic` with a `url` you host, and point `autoUpdater` at it. Config-only.
