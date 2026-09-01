/**
 * Bridge to the Electron shell (exposed on `window.desktop` by the preload
 * script via `contextBridge`). Every method degrades to a safe no-op in a
 * plain browser, so the same build runs in dev with `npm run dev` and packaged
 * in Electron with no branching at the call sites beyond `isDesktop()`.
 */

export type UpdateInfo = { version: string; releaseNotes?: string | null; releaseName?: string | null };
export type DownloadProgress = { percent: number; transferred: number; total: number; bytesPerSecond: number };

type DesktopBridge = {
  getVersion(): Promise<string>;
  checkForUpdates(): Promise<{ updateAvailable: boolean; info?: UpdateInfo }>;
  downloadUpdate(): Promise<void>;
  quitAndInstall(): void;
  onUpdateAvailable(cb: (info: UpdateInfo) => void): () => void;
  onDownloadProgress(cb: (p: DownloadProgress) => void): () => void;
  onUpdateDownloaded(cb: (info: UpdateInfo) => void): () => void;
  onUpdateError(cb: (message: string) => void): () => void;
};

declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}

export function isDesktop(): boolean {
  return typeof window !== "undefined" && !!window.desktop;
}

const noopUnsub = () => {};

export const desktop: DesktopBridge = {
  getVersion: () => window.desktop?.getVersion() ?? Promise.resolve(""),
  checkForUpdates: () => window.desktop?.checkForUpdates() ?? Promise.resolve({ updateAvailable: false }),
  downloadUpdate: () => window.desktop?.downloadUpdate() ?? Promise.resolve(),
  quitAndInstall: () => window.desktop?.quitAndInstall(),
  onUpdateAvailable: (cb) => window.desktop?.onUpdateAvailable(cb) ?? noopUnsub,
  onDownloadProgress: (cb) => window.desktop?.onDownloadProgress(cb) ?? noopUnsub,
  onUpdateDownloaded: (cb) => window.desktop?.onUpdateDownloaded(cb) ?? noopUnsub,
  onUpdateError: (cb) => window.desktop?.onUpdateError(cb) ?? noopUnsub,
};
