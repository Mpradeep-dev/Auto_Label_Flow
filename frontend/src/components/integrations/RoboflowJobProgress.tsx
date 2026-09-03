import { useEffect, useRef, useState } from "react";
import { api } from "@/services/api";
import type { RoboflowJob, RoboflowJobStatus } from "@/types";

// Follows a Roboflow import/export job over SSE — same shape as
// AutoAnnotationPage's inference progress bar (PLAN "SSE over WebSockets
// because the flow is one-directional and reconnects itself"), pulled out
// into its own component since both the import (Datasets page) and export
// (Export page) flows need an identical bar. Render with `key={job.id}` so
// a new job resets this component's state instead of reusing stale one.
interface StreamPayload {
  current: number;
  total: number;
  status: string;
  error: string | null;
  // Attempts that failed. `current` is successes only, so an export where
  // every image is rejected stays at current=0 with failed climbing —
  // the bar never advances on work that didn't reach Roboflow.
  failed?: number;
}

const SETTLED: RoboflowJobStatus[] = ["COMPLETED", "FAILED", "CANCELLED"];

export function RoboflowJobProgress({
  job,
  onSettled,
}: {
  job: RoboflowJob;
  onSettled?: (status: RoboflowJobStatus) => void;
}) {
  const [progress, setProgress] = useState<StreamPayload>({
    current: job.processed_items,
    total: job.total_items,
    status: job.status,
    error: job.error,
    failed: job.failed_count,
  });
  const [cancelling, setCancelling] = useState(false);
  const settledRef = useRef(false);

  useEffect(() => {
    if (SETTLED.includes(job.status)) {
      onSettled?.(job.status);
      return;
    }

    const source = new EventSource(`/api/v1/integrations/roboflow/jobs/${job.id}/stream`);
    source.onmessage = (e) => {
      const data: StreamPayload = JSON.parse(e.data);
      setProgress(data);
      if (SETTLED.includes(data.status as RoboflowJobStatus) && !settledRef.current) {
        settledRef.current = true;
        source.close();
        onSettled?.(data.status as RoboflowJobStatus);
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  const failed = progress.failed ?? 0;
  // Bar tracks confirmed successes only — never failed attempts.
  const pct = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;
  const running = progress.status === "RUNNING" || progress.status === "QUEUED";
  // Running, but every attempt so far has bounced and nothing has landed —
  // don't imply forward motion; say plainly that it isn't reaching Roboflow.
  const stalled = running && failed > 0 && progress.current === 0;

  async function cancel() {
    setCancelling(true);
    try {
      await api.cancelRoboflowJob(job.id);
      // Not setting local status here — the running task notices the flag
      // at its next per-item check and the SSE stream reports CANCELLED
      // from there, same as every other status transition in this bar.
    } finally {
      setCancelling(false);
    }
  }

  return (
    <div className="mt-3">
      <div className={`h-2 w-full border ${stalled ? "border-accent" : "border-ink"}`}>
        <div className="h-full bg-ink transition-all duration-150" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 flex items-center justify-between gap-3">
        <p className="tabular text-xs text-ink/60">
          {failed > 0 ? (
            <>
              {progress.current} uploaded · <span className="text-accent-ink">{failed} failed</span> ·{" "}
              {progress.current + failed} / {progress.total}
            </>
          ) : (
            <>
              {progress.current} / {progress.total} images
            </>
          )}
          {running && !stalled && " · working…"}
          {!running && (
            <>
              {" · "}
              <span className="font-bold uppercase">{progress.status}</span>
            </>
          )}
        </p>
        {running && (
          <button
            onClick={cancel}
            disabled={cancelling}
            className="shrink-0 border border-ink px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:bg-orange hover:text-ink hover:border-orange disabled:opacity-40"
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>
      {stalled && !progress.error && (
        <p className="mt-1 text-xs text-accent-ink">
          Not reaching Roboflow — {failed} upload{failed === 1 ? "" : "s"} rejected so far, nothing
          stored yet. If this keeps up the export stops and shows why (usually an out-of-quota or
          expired Roboflow plan).
        </p>
      )}
      {progress.error && <p className="mt-1 text-xs text-accent-ink">{progress.error}</p>}
    </div>
  );
}
