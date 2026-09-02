#!/usr/bin/env node
// Launches the Electron shell in "dev-frontend" mode: the backend runs
// API-only on 127.0.0.1:8000 (matching frontend/vite.config.ts's default
// proxy target) and the window loads the Vite dev server instead of
// frontend/dist — no `npm run build` needed to test in the desktop shell.
//
// Start `npm run dev` in frontend/ first, then this in desktop/.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import electronBin from "electron";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const env = {
  ...process.env,
  ALF_DEV_FRONTEND_URL: process.env.ALF_DEV_FRONTEND_URL || "http://127.0.0.1:5173",
  ALF_DEV_BACKEND_PORT: process.env.ALF_DEV_BACKEND_PORT || "8000",
};

const child = spawn(electronBin, ["."], { stdio: "inherit", cwd: path.join(__dirname, ".."), env });
child.on("exit", (code) => process.exit(code ?? 0));
