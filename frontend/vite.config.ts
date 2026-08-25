/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Where the dev-server proxy sends /api and /media. Must be overridable:
// on the host (local dev without Docker, per README) 127.0.0.1:8000 reaches
// a locally-run backend; inside the `frontend` container that same address
// is the container's OWN loopback, where nothing listens — the backend is
// only reachable there via the Docker Compose service name. docker-compose.yml
// sets API_PROXY_TARGET=http://backend:8000 for exactly this reason.
const apiProxyTarget = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiProxyTarget, changeOrigin: true },
      "/media": { target: apiProxyTarget, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
