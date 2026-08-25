import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { Dataset, InferenceJob, JobProgressEvent, MLModel } from "@/types";

const STATUS_STYLE: Record<string, string> = {
  QUEUED: "bg-muted text-ink/70",
  RUNNING: "bg-ink text-paper",
  COMPLETED: "bg-ink text-paper",
  FAILED: "bg-accent text-paper",
  CANCELLED: "bg-muted text-ink/50",
};

// Converts a finished/history job row into the same shape the live SSE
// stream produces, so History can reuse the one progress-bar renderer
// instead of a second display for "job I'm not currently following".
function jobToProgress(job: InferenceJob): JobProgressEvent {
  return {
    current: job.processed_images,
    total: job.total_images,
    predictions: job.total_predictions,
    fps: 0,
    eta_s: null,
    status: job.status,
    error: job.error,
  };
}

/**
 * Dispatches a real background job (Celery, `gpu` queue) and follows its
 * progress over SSE — see PLAN "Jobs: Celery + Redis" / "SSE over
 * WebSockets because the flow is one-directional and reconnects itself".
 * The browser tab is free to navigate away; the job keeps running.
 */
// Unlike the Roboflow reattach (scoped by the project id already in the
// URL), this page's dataset/model choice is plain form state with nothing
// to restore it from on a fresh mount — persisting the last selection is
// what makes "navigate away, come back" actually reattach on its own
// instead of requiring you to re-pick the same dataset first.
function selectionStorageKey(projectId: string): string {
  return `auto-annotate-selection:${projectId}`;
}

// Project-scoped run history — independent of whichever job the page is
// currently following (or not). Polls on its own timer so a job that
// finished (or failed) while the user was on another page still shows up
// here instead of just disappearing, which is what the single-dataset
// /latest reattach above this component can't do on its own.
function HistorySection({
  projectId,
  datasets,
  models,
  activeJobId,
  onSelect,
}: {
  projectId: string;
  datasets: Dataset[];
  models: MLModel[];
  activeJobId: string | null;
  onSelect: (job: InferenceJob) => void;
}) {
  const queryClient = useQueryClient();
  const jobsQuery = useQuery({
    queryKey: ["inference-jobs", projectId],
    queryFn: () => api.listInferenceJobs(projectId),
    refetchInterval: 5000,
  });

  const cancelMutation = useMutation({
    mutationFn: (id: string) => api.cancelInferenceJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["inference-jobs", projectId] }),
  });

  const jobs = jobsQuery.data ?? [];
  if (jobs.length === 0) return null;

  return (
    <div className="mt-16 max-w-3xl">
      <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/50">History</p>
      <div className="border-t-2 border-ink">
        {jobs.map((j) => {
          const dataset = datasets.find((d) => d.id === j.dataset_id);
          const model = models.find((m) => m.id === j.model_id);
          const cancellable = j.status === "RUNNING" || j.status === "QUEUED";
          return (
            <div
              key={j.id}
              className={`flex items-center justify-between gap-4 border-b-2 border-ink py-3 ${
                j.id === activeJobId ? "bg-muted" : ""
              }`}
            >
              <button onClick={() => onSelect(j)} className="flex-1 text-left hover:text-accent">
                <span className="text-xs font-semibold">
                  {dataset?.name ?? "Unknown dataset"} · {model?.name ?? "Unknown model"}
                </span>
                <span className="tabular ml-2 text-[10px] text-ink/50">
                  {j.processed_images}/{j.total_images} images · {j.total_predictions} predictions ·{" "}
                  {new Date(j.created_at).toLocaleString()}
                </span>
              </button>
              <span className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${STATUS_STYLE[j.status]}`}>
                {j.status}
              </span>
              {cancellable && (
                <button
                  onClick={() => cancelMutation.mutate(j.id)}
                  disabled={cancelMutation.isPending}
                  className="border border-ink/30 px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:border-accent hover:text-accent disabled:opacity-40"
                >
                  Cancel
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AutoAnnotationPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const eventSourceRef = useRef<EventSource | null>(null);

  const [datasetId, setDatasetId] = useState(() => {
    if (!projectId) return "";
    try {
      return JSON.parse(localStorage.getItem(selectionStorageKey(projectId)) ?? "{}").datasetId ?? "";
    } catch {
      return "";
    }
  });
  const [modelId, setModelId] = useState(() => {
    if (!projectId) return "";
    try {
      return JSON.parse(localStorage.getItem(selectionStorageKey(projectId)) ?? "{}").modelId ?? "";
    } catch {
      return "";
    }
  });
  const [conf, setConf] = useState(0.2);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobProgressEvent | null>(null);
  // Distinct from `running` (SSE-reported job status): covers the gap
  // between clicking the button and the job actually starting to report
  // progress — without it, a slow or failing `createInferenceJob` call
  // left the button looking completely inert (no spinner, no error, just
  // nothing) for however long that took.
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

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

  useEffect(() => {
    if (!projectId) return;
    try {
      localStorage.setItem(selectionStorageKey(projectId), JSON.stringify({ datasetId, modelId }));
    } catch {
      /* private-browsing or storage-disabled — losing the remembered selection is harmless */
    }
  }, [projectId, datasetId, modelId]);

  // Reattaches to a job this page kicked off before a navigation away or a
  // reload wiped `progress`/`jobId` above — otherwise a still-running batch
  // just disappears from the UI even though it keeps going server-side.
  // Only fires once a dataset is (re-)selected, same limitation the
  // Roboflow reattach doesn't have (its scope comes from the URL; this
  // page's dataset choice is local form state with nothing to restore it
  // from on a fresh mount).
  const latestJobQuery = useQuery({
    queryKey: ["latest-inference-job", datasetId],
    queryFn: () => api.getLatestInferenceJob(datasetId),
    enabled: !!datasetId,
  });
  useEffect(() => {
    const latest = latestJobQuery.data;
    if (latest && (latest.status === "RUNNING" || latest.status === "QUEUED") && jobId !== latest.id) {
      setJobId(latest.id);
      followJob(latest.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestJobQuery.data]);

  function followJob(id: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`/api/v1/inference/jobs/${id}/stream`);
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
    source.onerror = () => {
      source.close();
      // Only surface this as an error if the job never got anywhere —
      // once terminal progress has already rendered, a stream drop after
      // the fact isn't something the user needs to see as a failure.
      setProgress((current) =>
        current && current.status !== "RUNNING"
          ? current
          : { current: 0, total: 0, predictions: 0, fps: 0, eta_s: null, status: "FAILED", error: "Lost connection to the progress stream." },
      );
    };
  }

  // Clicking a History row: still-live jobs reattach over SSE like a normal
  // follow; terminal ones just render their final DB state directly — no
  // need to open a stream for a job that's already done.
  function selectJob(job: InferenceJob) {
    setJobId(job.id);
    if (job.status === "RUNNING" || job.status === "QUEUED") {
      followJob(job.id);
    } else {
      eventSourceRef.current?.close();
      setProgress(jobToProgress(job));
    }
  }

  async function run() {
    if (!datasetId || !modelId) return;
    setProgress(null);
    setSubmitError(null);
    setSubmitting(true);
    try {
      const job: InferenceJob = await api.createInferenceJob({ dataset_id: datasetId, model_id: modelId, conf });
      setJobId(job.id);
      followJob(job.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to start auto-annotation.");
    } finally {
      setSubmitting(false);
    }
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
          disabled={!datasetId || !modelId || submitting || running}
          className="w-full border-2 border-ink bg-ink py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {submitting
            ? "Starting…"
            : running
              ? `Processing… ${progress?.current} / ${progress?.total}`
              : "Run auto-annotation"}
        </button>
        {submitError && <p className="text-xs text-accent">{submitError}</p>}

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
                  <Link to={`/projects/${projectId}/review?datasetId=${datasetId}`} className="underline">
                    Start reviewing →
                  </Link>
                </>
              )}
            </p>
            {progress.error && <p className="mt-1 text-xs text-accent">{progress.error}</p>}
          </div>
        )}
        {jobId && !finished && (
          <p className="text-[10px] uppercase tracking-widest text-ink/30">
            Job <span className="tabular">{jobId}</span> runs in the background — this page can be
            left safely.
          </p>
        )}
      </div>

      <HistorySection
        projectId={projectId}
        datasets={datasetsQuery.data ?? []}
        models={modelsQuery.data ?? []}
        activeJobId={jobId}
        onSelect={selectJob}
      />
    </div>
  );
}
