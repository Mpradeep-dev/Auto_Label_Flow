import { useEffect, useRef, useState } from "react";
import type { RoboflowJobStatus } from "@/types";

// Generic SSE-followed progress bar for a background import job — the
// same shape as RoboflowJobProgress, but parameterised by stream URL and
// cancel callback so the Azure-Blob import (and any future import source)
// can reuse it without copying the EventSource/settle wiring.
interface StreamPayload {
  current: number;
  total: number;
  status: string;
  error: string | null;
}

const SETTLED: RoboflowJobStatus[] = ["COMPLETED", "FAILED", "CANCELLED"];

export function ImportJobProgress({
  jobId,
  initialStatus,
  initialProcessed,
  initialTotal,
  initialError,
  streamUrl,
  onCancel,
  onSettled,
  unit = "images",
}: {
  jobId: string;
  initialStatus: RoboflowJobStatus;
  initialProcessed: number;
  initialTotal: number;
  initialError: string | null;
  streamUrl: string;
  onCancel: () => Promise<unknown>;
  onSettled?: (status: RoboflowJobStatus) => void;
  unit?: string;
}) {
  const [progress, setProgress] = useState<StreamPayload>({
    current: initialProcessed,
    total: initialTotal,
    status: initialStatus,
    error: initialError,
  });
  const [cancelling, setCancelling] = useState(false);
  const settledRef = useRef(false);

  useEffect(() => {
    if (SETTLED.includes(initialStatus)) {
      onSettled?.(initialStatus);
      return;
    }

    const source = new EventSource(streamUrl);
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
  }, [jobId]);

  const pct = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;
  const running = progress.status === "RUNNING" || progress.status === "QUEUED";

  async function cancel() {
    setCancelling(true);
    try {
      await onCancel();
    } finally {
      setCancelling(false);
    }
  }

  return (
    <div className="mt-3">
      <div className="h-2 w-full border border-ink">
        <div className="h-full bg-ink transition-all duration-150" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 flex items-center justify-between gap-3">
        <p className="tabular text-xs text-ink/60">
          {progress.current} / {progress.total} {unit}
          {running && " · working…"}
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
      {progress.error && <p className="mt-1 text-xs text-accent-ink">{progress.error}</p>}
    </div>
  );
}
