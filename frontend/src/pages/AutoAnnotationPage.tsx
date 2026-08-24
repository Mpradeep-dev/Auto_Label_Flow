import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { JobProgressEvent } from "@/types";

/**
 * Dispatches a real background job (Celery, `gpu` queue) and follows its
 * progress over SSE — see PLAN "Jobs: Celery + Redis" / "SSE over
 * WebSockets because the flow is one-directional and reconnects itself".
 * The browser tab is free to navigate away; the job keeps running.
 */
export function AutoAnnotationPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const eventSourceRef = useRef<EventSource | null>(null);

  const [datasetId, setDatasetId] = useState("");
  const [modelId, setModelId] = useState("");
  const [conf, setConf] = useState(0.2);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobProgressEvent | null>(null);

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const detectorModels = (modelsQuery.data ?? []).filter((m) => m.kind === "DETECTOR");

  const running = progress != null && progress.status === "RUNNING";
  const finished = progress != null && progress.status !== "RUNNING";

  useEffect(() => {
    return () => eventSourceRef.current?.close();
  }, []);

  async function run() {
    if (!datasetId || !modelId) return;
    setProgress(null);
    const job = await api.createInferenceJob({ dataset_id: datasetId, model_id: modelId, conf });
    setJobId(job.id);

    eventSourceRef.current?.close();
    const source = new EventSource(`/api/v1/inference/jobs/${job.id}/stream`);
    eventSourceRef.current = source;
    source.onmessage = (e) => {
      const data: JobProgressEvent = JSON.parse(e.data);
      setProgress(data);
      if (data.status !== "RUNNING") {
        source.close();
        queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
        queryClient.invalidateQueries({ queryKey: ["dataset-stats", datasetId] });
      }
    };
    source.onerror = () => source.close();
  }

  if (!projectId) return null;

  const pct = progress && progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Auto annotation</SectionLabel>
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Auto Annotation
      </h1>

      <div className="max-w-2xl space-y-6">
        <div>
          <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">Dataset</label>
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
          >
            <option value="">Select a dataset…</option>
            {(datasetsQuery.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">Model</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
          >
            <option value="">Select a detector…</option>
            {detectorModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.class_config.map((c) => c.name).join(", ")})
              </option>
            ))}
          </select>
          {detectorModels.length === 0 && (
            <p className="mt-2 text-xs text-ink/50">
              No detector registered yet.{" "}
              <Link to={`/projects/${projectId}/models`} className="underline">
                Register one
              </Link>
              .
            </p>
          )}
        </div>

        <div>
          <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Confidence floor: <span className="tabular">{conf.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            className="w-full accent-accent"
          />
          <p className="mt-1 text-[10px] text-ink/40">
            Kept low by default — low-confidence detections are review candidates, not noise. See
            the documented foot/cone confusion: the known false positive scores well below a
            typical 0.5+ cutoff.
          </p>
        </div>

        <button
          onClick={run}
          disabled={!datasetId || !modelId || running}
          className="w-full border-2 border-ink bg-ink py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {running ? `Processing… ${progress?.current} / ${progress?.total}` : "Run auto-annotation"}
        </button>

        {progress && (
          <div>
            <div className="h-2 w-full border border-ink">
              <div className="h-full bg-ink transition-all duration-150" style={{ width: `${pct}%` }} />
            </div>
            <p className="tabular mt-2 text-xs text-ink/60">
              {progress.current} / {progress.total} images · {progress.predictions} predictions
              {running && progress.fps > 0 && (
                <> · {progress.fps.toFixed(1)} img/s · ETA {Math.ceil(progress.eta_s ?? 0)}s</>
              )}
              {finished && (
                <>
                  {" "}
                  · <span className="font-bold uppercase">{progress.status}</span>
                  {" · "}
                  <Link to={`/projects/${projectId}/datasets`} className="underline">
                    Review results →
                  </Link>
                </>
              )}
            </p>
            {progress.error && <p className="mt-1 text-xs text-accent">{progress.error}</p>}
          </div>
        )}
        {jobId && (
          <p className="text-[10px] uppercase tracking-widest text-ink/30">
            Job <span className="tabular">{jobId}</span> runs in the background — this page can be
            left safely.
          </p>
        )}
      </div>
    </div>
  );
}
