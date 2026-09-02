import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { desktop } from "@/services/desktop";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { FieldError } from "@/components/layout/FieldError";

// Desktop-only Settings section: the manual "Check for updates" button and
// the two optional add-on packs (GPU training support, cloud integrations).
// Rendered by IntegrationsPage only when running inside the Electron shell.

function fmtSize(bytes: number | null): string {
  if (!bytes) return "";
  const mb = bytes / 1_000_000;
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

type UpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "current" }
  | { kind: "available"; version: string; notes?: string | null }
  | { kind: "downloading"; percent: number }
  | { kind: "downloaded"; version: string }
  | { kind: "error"; message: string; detail?: string };

// electron-updater rejects (check OR download) with the raw HttpError text —
// 404 body, response headers and a Node stack trace all concatenated into one
// string. Collapse the shapes we recognise into a single sentence; the
// untouched original is kept behind a "Technical details" toggle.
function humanizeUpdateError(raw: string): string {
  const s = raw.replace(/^Error invoking remote method '[^']*':\s*/, "").trim();
  if (/latest\.yml|Cannot find .*\.ya?ml|HttpError: 404|\b404\b/i.test(s))
    return "A newer version is available, but its installer files aren't published on the release yet. Try again later.";
  if (/ENOTFOUND|EAI_AGAIN|ETIMEDOUT|ECONNREFUSED|ECONNRESET|getaddrinfo|net::|ERR_INTERNET/i.test(s))
    return "Couldn't reach the update server. Check your internet connection and try again.";
  if (/No published versions|Unable to find latest version/i.test(s))
    return "No published releases are available yet.";
  if (/sha512|checksum|integrity|signature/i.test(s))
    return "The downloaded update failed its integrity check. Try again.";
  if (/ENOSPC|no space/i.test(s)) return "Not enough disk space to download the update.";
  return "The update couldn't be completed. See technical details below.";
}

function AppUpdateCard() {
  const infoQuery = useQuery({ queryKey: ["system-info"], queryFn: () => api.systemInfo() });
  const [state, setState] = useState<UpdateState>({ kind: "idle" });

  useEffect(() => {
    const offAvail = desktop.onUpdateAvailable((i) =>
      setState({ kind: "available", version: i.version, notes: i.releaseNotes }),
    );
    const offProg = desktop.onDownloadProgress((p) => setState({ kind: "downloading", percent: p.percent }));
    const offDone = desktop.onUpdateDownloaded((i) => setState({ kind: "downloaded", version: i.version }));
    const offErr = desktop.onUpdateError((m) =>
      setState({ kind: "error", message: humanizeUpdateError(m), detail: m }),
    );
    return () => {
      offAvail();
      offProg();
      offDone();
      offErr();
    };
  }, []);

  async function check() {
    setState({ kind: "checking" });
    try {
      const r = await desktop.checkForUpdates();
      if (r.updateAvailable && r.info) {
        setState({ kind: "available", version: r.info.version, notes: r.info.releaseNotes });
      } else {
        setState({ kind: "current" });
      }
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e);
      setState({ kind: "error", message: humanizeUpdateError(raw), detail: raw });
    }
  }

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">Application update</p>
        <span className="tabular px-2 py-1 text-[10px] font-bold uppercase tracking-widest bg-muted text-ink/60">
          v{infoQuery.data?.app_version ?? "…"}
        </span>
      </div>
      <p className="mb-4 text-sm text-ink/60">
        Downloads and installs the latest release, then restarts. Your projects, datasets and models are kept.
      </p>

      <div className="flex flex-wrap items-center gap-3 border-t border-ink/20 pt-4">
        <button
          onClick={check}
          disabled={state.kind === "checking" || state.kind === "downloading"}
          className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange hover:text-ink disabled:opacity-40"
        >
          {state.kind === "checking" ? "Checking…" : "Check for updates"}
        </button>

        {state.kind === "current" && <span className="text-xs uppercase tracking-widest text-ink/60">Up to date</span>}

        {state.kind === "available" && (
          <button
            onClick={() => {
              setState({ kind: "downloading", percent: 0 });
              // Belt-and-braces: electron-updater also emits `error`, but the
              // returned promise can reject on its own — don't leave it unhandled.
              desktop.downloadUpdate().catch((e) => {
                const raw = e instanceof Error ? e.message : String(e);
                setState({ kind: "error", message: humanizeUpdateError(raw), detail: raw });
              });
            }}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:border-orange"
          >
            Download v{state.version}
          </button>
        )}

        {state.kind === "downloading" && (
          <span className="tabular text-xs uppercase tracking-widest text-ink/60">
            Downloading… {Math.round(state.percent)}%
          </span>
        )}

        {state.kind === "downloaded" && (
          <button
            onClick={() => desktop.quitAndInstall()}
            className="border-2 border-ink bg-orange px-4 py-2 text-xs font-bold uppercase tracking-widest"
          >
            Restart &amp; install v{state.version}
          </button>
        )}
      </div>

      {state.kind === "error" && (
        <div role="alert" className="mt-4 border-2 border-accent-ink p-3">
          <p className="text-xs font-bold uppercase tracking-widest text-accent-ink">Update failed</p>
          <p className="mt-1 text-xs text-ink/70">{state.message}</p>
          {state.detail && (
            <details className="mt-2">
              <summary className="cursor-pointer text-[10px] font-bold uppercase tracking-widest text-ink/50 hover:text-ink">
                Technical details
              </summary>
              <pre className="mt-2 max-h-40 overflow-auto border border-ink/20 bg-muted p-3 text-[11px] whitespace-pre-wrap break-words text-ink/60">
                {state.detail}
              </pre>
            </details>
          )}
        </div>
      )}

      {state.kind === "available" && state.notes && (
        <pre className="mt-4 max-h-40 overflow-auto border border-ink/20 bg-muted p-3 text-xs whitespace-pre-wrap text-ink/70">
          {state.notes}
        </pre>
      )}
    </div>
  );
}

function PackRow({
  name,
  title,
  blurb,
}: {
  name: "gpu" | "integrations";
  title: string;
  blurb: string;
}) {
  const queryClient = useQueryClient();
  const packsQuery = useQuery({ queryKey: ["system-packs"], queryFn: () => api.listPacks() });
  const pack = packsQuery.data?.packs.find((p) => p.name === name);
  const [log, setLog] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  // Set once the stream reaches a terminal state so the `error` event that
  // fires on our own es.close() isn't mistaken for a dropped connection.
  const settledRef = useRef(false);

  useEffect(() => () => esRef.current?.close(), []);

  const install = useMutation({
    mutationFn: () => api.installPack(name),
    onMutate: () => setStreamError(null),
    onSuccess: () => {
      setRunning(true);
      setLog("");
      settledRef.current = false;
      const es = new EventSource(`/api/v1/system/packs/${name}/stream`);
      esRef.current = es;
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data) as { state: string; detail: string; lines: string[] };
        setLog(d.lines.slice(-8).join("\n"));
        if (d.state === "done" || d.state === "failed") {
          settledRef.current = true;
          es.close();
          setRunning(false);
          if (d.state === "failed")
            setStreamError(d.detail || "The installation failed — see the log below.");
          queryClient.invalidateQueries({ queryKey: ["system-packs"] });
          queryClient.invalidateQueries({ queryKey: ["system-info"] });
          queryClient.invalidateQueries({ queryKey: ["integrations"] });
        }
      };
      es.onerror = () => {
        es.close();
        setRunning(false);
        if (!settledRef.current)
          setStreamError("Lost the connection to the installer. It may still be running — reopen this page to check.");
      };
    },
  });

  const remove = useMutation({
    mutationFn: () => api.removePack(name),
    onMutate: () => setStreamError(null),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["system-packs"] });
      queryClient.invalidateQueries({ queryKey: ["system-info"] });
    },
  });

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">{title}</p>
        <span
          className={`px-2 py-1 text-[10px] font-bold uppercase tracking-widest ${
            pack?.installed ? "bg-ink text-paper" : "bg-muted text-ink/60"
          }`}
        >
          {pack?.installed ? "Installed" : "Not installed"}
        </span>
      </div>
      <p className="mb-4 text-sm text-ink/60">{blurb}</p>

      <div className="flex flex-wrap items-center gap-3 border-t border-ink/20 pt-4">
        {pack?.installed ? (
          <>
            <span className="tabular text-xs uppercase tracking-widest text-ink/60">
              {fmtSize(pack.size_bytes)}
            </span>
            <button
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
              className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:border-orange disabled:opacity-40"
            >
              Remove
            </button>
          </>
        ) : (
          <button
            onClick={() => install.mutate()}
            disabled={running || install.isPending}
            className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange hover:text-ink disabled:opacity-40"
          >
            {running ? "Downloading…" : "Download"}
          </button>
        )}
      </div>

      <FieldError error={install.error ?? remove.error} />
      {streamError && (
        <p role="alert" className="mt-2 text-xs text-accent-ink">
          {streamError}
        </p>
      )}

      {log && (
        <pre className="mt-4 max-h-40 overflow-auto border border-ink/20 bg-muted p-3 text-[11px] whitespace-pre-wrap break-words text-ink/70">
          {log}
        </pre>
      )}
    </div>
  );
}

function fmtBytes(bytes: number): string {
  const mb = bytes / 1_000_000;
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

// SAM checkpoints — a separate mechanism from PackRow above: not a pip
// install, just a weights file (ultralytics already ships SAM/MobileSAM
// support). Same card shape, different progress payload (byte counts, not
// a pip log) and available on the server profile too, not desktop-only.
function SamModelRow({ name }: { name: "sam-lite" | "sam-full" }) {
  const queryClient = useQueryClient();
  const modelsQuery = useQuery({ queryKey: ["system-sam-models"], queryFn: () => api.listSamModels() });
  const model = modelsQuery.data?.find((m) => m.name === name);
  const [progress, setProgress] = useState<{ downloaded: number; total: number } | null>(null);
  const [running, setRunning] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const settledRef = useRef(false);

  useEffect(() => () => esRef.current?.close(), []);

  const install = useMutation({
    mutationFn: () => api.installSamModel(name),
    onMutate: () => setStreamError(null),
    onSuccess: () => {
      setRunning(true);
      setProgress({ downloaded: 0, total: 0 });
      settledRef.current = false;
      const es = new EventSource(`/api/v1/system/sam-models/${name}/stream`);
      esRef.current = es;
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data) as { state: string; detail: string; downloaded: number; total: number };
        setProgress({ downloaded: d.downloaded, total: d.total });
        if (d.state === "done" || d.state === "failed") {
          settledRef.current = true;
          es.close();
          setRunning(false);
          if (d.state === "failed") setStreamError(d.detail || "The download failed.");
          queryClient.invalidateQueries({ queryKey: ["system-sam-models"] });
        }
      };
      es.onerror = () => {
        es.close();
        setRunning(false);
        if (!settledRef.current)
          setStreamError("Lost the connection to the download. It may still be running — reopen this page to check.");
      };
    },
  });

  const remove = useMutation({
    mutationFn: () => api.removeSamModel(name),
    onMutate: () => setStreamError(null),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["system-sam-models"] }),
  });

  const pct = progress && progress.total > 0 ? Math.round((progress.downloaded / progress.total) * 100) : null;

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">{model?.label ?? name}</p>
        <span
          className={`px-2 py-1 text-[10px] font-bold uppercase tracking-widest ${
            model?.installed ? "bg-ink text-paper" : "bg-muted text-ink/60"
          }`}
        >
          {model?.installed ? "Installed" : "Not installed"}
        </span>
      </div>
      <p className="mb-4 text-sm text-ink/60">{model?.blurb}</p>

      <div className="flex flex-wrap items-center gap-3 border-t border-ink/20 pt-4">
        {model?.installed ? (
          <>
            <span className="tabular text-xs uppercase tracking-widest text-ink/60">
              {model.size_bytes ? fmtBytes(model.size_bytes) : ""}
            </span>
            <button
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
              className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:border-orange disabled:opacity-40"
            >
              Remove
            </button>
          </>
        ) : (
          <button
            onClick={() => install.mutate()}
            disabled={running || install.isPending}
            className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange hover:text-ink disabled:opacity-40"
          >
            {running ? (pct != null ? `Downloading… ${pct}%` : "Downloading…") : "Download"}
          </button>
        )}
      </div>

      <FieldError error={install.error ?? remove.error} />
      {streamError && (
        <p role="alert" className="mt-2 text-xs text-accent-ink">
          {streamError}
        </p>
      )}
    </div>
  );
}

export function DesktopPanel({ sectionIndex = 2 }: { sectionIndex?: number }) {
  return (
    <section className="mb-16">
      <SectionLabel index={sectionIndex}>Desktop app</SectionLabel>
      <div className="grid max-w-5xl grid-cols-1 gap-6 lg:grid-cols-2">
        <AppUpdateCard />
        <PackRow
          name="gpu"
          title="GPU training support"
          blurb="Adds CUDA PyTorch so you can train YOLO models on a local NVIDIA GPU. ~2.5 GB. Without it, training runs on Kaggle or Modal instead."
        />
        <PackRow
          name="integrations"
          title="Cloud integrations"
          blurb="Adds the Kaggle, Modal and Roboflow SDKs so those connections become available below. ~80 MB."
        />
        <SamModelRow name="sam-lite" />
        <SamModelRow name="sam-full" />
      </div>
    </section>
  );
}
