/**
 * Assemble desktop/build/ — the payload electron-builder packs as
 * extraResources. Run on a Windows x64 machine before `electron-builder`.
 *
 *   node scripts/build.mjs            # full assembly
 *   node scripts/build.mjs --skip-python   # reuse an existing build/python
 *
 * Produces:
 *   build/python/          relocatable CPython 3.12 + venv site-packages
 *   build/python/app-root/ a copy of ../backend (app/, alembic/, requirements*)
 *   build/frontend/        ../frontend/dist built with VITE_DESKTOP=1
 *   build/bin/ffmpeg.exe   static LGPL ffmpeg (see RELEASING.md)
 *
 * python-build-standalone is preferred over PyInstaller here: torch /
 * ultralytics / OpenCV on Windows ship hundreds of DLLs and do dynamic
 * imports that PyInstaller needs per-package hooks for. Shipping "a Python
 * install + site-packages" is exactly what the repo already runs and tests.
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, cpSync, writeFileSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import process from "node:process";

const ROOT = resolve(import.meta.dirname, "..");
const REPO = resolve(ROOT, "..");
const BUILD = join(ROOT, "build");
const args = new Set(process.argv.slice(2));

// Pin the python-build-standalone release + CPython version used for the bundle.
const PBS_TAG = "20241016";
const PY_VER = "3.12.7";
const PBS_URL = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PY_VER}+${PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz`;

const version = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")).version;
console.log(`▶ assembling desktop payload for v${version}`);

function run(cmd, opts = {}) {
  console.log(`  $ ${cmd}`);
  execSync(cmd, { stdio: "inherit", ...opts });
}

// --- 1. version stamp (backend reads app/_version.py; config.py falls back if absent) ---
const versionPy = `__version__ = "${version}"\n`;
writeFileSync(join(REPO, "backend", "app", "_version.py"), versionPy);

// --- 2. frontend ---
if (!args.has("--skip-frontend")) {
  const FE = join(REPO, "frontend");
  run("npm ci", { cwd: FE });
  run("npm run build", { cwd: FE, env: { ...process.env, VITE_DESKTOP: "1" } });
  rmSync(join(BUILD, "frontend"), { recursive: true, force: true });
  mkdirSync(BUILD, { recursive: true });
  cpSync(join(FE, "dist"), join(BUILD, "frontend"), { recursive: true });
  console.log("  ✓ frontend -> build/frontend");
}

// --- 3. python runtime ---
if (!args.has("--skip-python")) {
  rmSync(join(BUILD, "python"), { recursive: true, force: true });
  mkdirSync(BUILD, { recursive: true });

  const tgz = join(tmpdir(), `cpython-${PY_VER}-${PBS_TAG}.tar.gz`);
  if (!existsSync(tgz)) {
    console.log(`  ⬇ ${PBS_URL}`);
    run(`curl -L --fail -o "${tgz}" "${PBS_URL}"`);
  }
  // python-build-standalone "install_only" archives contain a top-level python/.
  // Use the Windows system bsdtar (handles C:\ drive paths; GNU tar reads the
  // colon as a remote host) and extract with cwd rather than -C.
  const tarExe =
    process.platform === "win32" ? `${process.env.SystemRoot}\\System32\\tar.exe` : "tar";
  run(`"${tarExe}" -xf "${tgz}"`, { cwd: BUILD }); // -> build/python/
  const py = join(BUILD, "python", "python.exe");
  if (!existsSync(py)) throw new Error(`expected ${py} after extract`);

  run(`"${py}" -m pip install --upgrade pip`);
  run(
    `"${py}" -m pip install --no-warn-script-location ` +
      `-r "${join(REPO, "backend", "requirements-desktop.txt")}" ` +
      `--extra-index-url https://download.pytorch.org/whl/cpu`,
  );

  // backend source alongside the interpreter
  const appRoot = join(BUILD, "python", "app-root");
  rmSync(appRoot, { recursive: true, force: true });
  mkdirSync(appRoot, { recursive: true });
  for (const item of ["app", "alembic", "alembic.ini", "requirements.txt", "requirements-desktop.txt"]) {
    const src = join(REPO, "backend", item);
    if (existsSync(src)) cpSync(src, join(appRoot, item), { recursive: true });
  }
  writeFileSync(join(appRoot, "app", "_version.py"), versionPy);
  console.log("  ✓ python runtime -> build/python");
}

// --- 4. ffmpeg (belt-and-braces for OpenCV video decode; see RELEASING.md) ---
const bin = join(BUILD, "bin");
mkdirSync(bin, { recursive: true });
if (!existsSync(join(bin, "ffmpeg.exe"))) {
  console.warn(
    "  ⚠ build/bin/ffmpeg.exe is missing.\n" +
      "    Drop a static LGPL Windows ffmpeg.exe there (https://www.gyan.dev/ffmpeg/builds/,\n" +
      "    ffmpeg-release-essentials). OpenCV bundles its own FFmpeg DLLs so this is a\n" +
      "    fallback, but ship it — some containers/codecs need the full binary.",
  );
}

console.log("✔ done. Next: npx electron-builder --win nsis --publish never (or `npm run release`).");
