/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" when built for the Electron desktop shell (see desktop/scripts/build.mjs). */
  readonly VITE_DESKTOP?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
