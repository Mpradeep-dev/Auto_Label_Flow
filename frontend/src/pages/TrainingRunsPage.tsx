import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { TrainingJob, TrainingProviderName } from "@/types";

const STATUS_STYLE: Record<string, string> = {
  QUEUED: "bg-muted text-ink/70",
  RUNNING: "bg-ink text-paper",
  COMPLETED: "bg-ink text-paper",
  FAILED: "bg-accent text-paper",
  CANCELLED: "bg-muted text-ink/50",
};

function MetricCell({ label, value }: { label: string; value: number | null | undefined }) {
  return (
    <div>
      <p className="text-[9px] font-bold uppercase tracking-widest text-ink/40">{label}</p>
      <p className="tabular text-lg font-bold">{value != null ? value.toFixed(3) : "—"}</p>
    </div>
  );
}

function JobDetail({ job }: { job: TrainingJob }) {
  const epochsQuery = useQuery({
    queryKey: ["training-epochs", job.id],
    queryFn: () => api.getTrainingJobEpochs(job.id),
    refetchInterval: job.status === "RUNNING" ? 2000 : false,
  });
  const jobQuery = useQuery({
    queryKey: ["training-job", job.id],
    queryFn: () => api.getTrainingJob(job.id),
    initialData: job,
    refetchInterval: job.status === "RUNNING" || job.status === "QUEUED" ? 2000 : false,
  });
  const current = jobQuery.data ?? job;
  const queryClient = useQueryClient();

  const cancelMutation = useMutation({
    mutationFn: () => api.cancelTrainingJob(job.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["training-job", job.id] }),
  });

  // Ultralytics can fire one extra on_fit_epoch_end during its final
  // validation pass after the last training epoch, so current_epoch can
  // exceed the requested total by one — clamp the bar rather than
  // overflow its border.
  const pct = current.epochs > 0 ? Math.min(100, Math.round((current.current_epoch / current.epochs) * 100)) : 0;

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-sm font-bold uppercase tracking-widest text-ink/50">
            {current.provider} · device {current.device}
          </p>
          <p className="tabular text-2xl font-black">
            Epoch {current.current_epoch} / {current.epochs}
          </p>
        </div>
        <span className={`px-3 py-1 text-xs font-bold uppercase tracking-widest ${STATUS_STYLE[current.status]}`}>
          {current.status}
        </span>
      </div>

      <div className="mb-4 h-2 w-full border border-ink">
        <div className="h-full bg-ink transition-all duration-150" style={{ width: `${pct}%` }} />
      </div>

      <div className="mb-4 grid grid-cols-4 gap-4 border-y-2 border-ink/10 py-4">
        <MetricCell label="mAP50" value={current.metrics.map50} />
        <MetricCell label="mAP50-95" value={current.metrics.map50_95} />
        <MetricCell label="Precision" value={current.metrics.precision} />
        <MetricCell label="Recall" value={current.metrics.recall} />
      </div>

      {current.error && <p className="mb-4 text-xs text-accent">{current.error}</p>}

      {(current.status === "RUNNING" || current.status === "QUEUED") && (
        <button
          onClick={() => cancelMutation.mutate()}
          className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-accent hover:text-paper hover:border-accent"
        >
          Cancel
        </button>
      )}
      {current.status === "COMPLETED" && current.result_model_id && (
        <p className="text-xs text-ink/60">
          New model registered — go to{" "}
          <a href="../models" className="underline">
            Models
          </a>{" "}
          or{" "}
          <a href="../auto-annotation" className="underline">
            Auto Annotation
          </a>{" "}
          to use it.
        </p>
      )}

      {epochsQuery.data && epochsQuery.data.length > 0 && (
        <table className="tabular mt-4 w-full text-xs">
          <thead>
            <tr className="border-b-2 border-ink text-left text-[10px] uppercase tracking-widest text-ink/50">
              <th className="py-1">Epoch</th>
              <th>Box</th>
              <th>Cls</th>
              <th>DFL</th>
              <th>P</th>
              <th>R</th>
              <th>mAP50</th>
              <th>mAP50-95</th>
            </tr>
          </thead>
          <tbody>
            {epochsQuery.data.map((e) => (
              <tr key={e.epoch} className="border-b border-ink/10">
                <td className="py-1">{e.epoch}</td>
                <td>{e.box_loss?.toFixed(3) ?? "—"}</td>
                <td>{e.cls_loss?.toFixed(3) ?? "—"}</td>
                <td>{e.dfl_loss?.toFixed(4) ?? "—"}</td>
                <td>{e.precision?.toFixed(3) ?? "—"}</td>
                <td>{e.recall?.toFixed(3) ?? "—"}</td>
                <td>{e.map50?.toFixed(3) ?? "—"}</td>
                <td>{e.map50_95?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function TrainingRunsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();

  const [datasetId, setDatasetId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [baseModelId, setBaseModelId] = useState("");
  const [provider, setProvider] = useState<TrainingProviderName>("LOCAL");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(8);
  const [imageSize, setImageSize] = useState(640);
  const [learningRate, setLearningRate] = useState("");
  const [device, setDevice] = useState("0");
  // Free-form passthrough for any other Ultralytics `YOLO.train()` kwarg
  // (optimizer, patience, dropout, augmentation knobs, ...) — the backend
  // has a typed column per the handful of settings above, but Ultralytics
  // documents ~100 train() arguments in total; a JSON box covers the rest
  // without a dedicated field for each one. Server-side, these always lose
  // to the typed fields above on conflict.
  const [extraArgsJson, setExtraArgsJson] = useState("");
  const [extraArgsError, setExtraArgsError] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const providersQuery = useQuery({ queryKey: ["training-providers"], queryFn: api.getTrainingProviders });
  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });
  const versionsQuery = useQuery({
    queryKey: ["versions", datasetId],
    queryFn: () => api.listDatasetVersions(datasetId),
    enabled: !!datasetId,
  });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const detectorModels = (modelsQuery.data ?? []).filter((m) => m.kind === "DETECTOR");
  const jobsQuery = useQuery({
    queryKey: ["training-jobs", projectId],
    queryFn: () => api.listTrainingJobs(projectId!),
    enabled: !!projectId,
    refetchInterval: 5000,
  });

  const createMutation = useMutation({
    mutationFn: () => {
      let extra_args: Record<string, unknown> | undefined;
      if (extraArgsJson.trim()) {
        try {
          extra_args = JSON.parse(extraArgsJson);
        } catch {
          throw new Error("Advanced parameters isn't valid JSON — fix it or clear the box.");
        }
      }
      return api.createTrainingJob({
        dataset_version_id: versionId,
        base_model_id: baseModelId,
        provider,
        epochs,
        batch_size: batchSize,
        image_size: imageSize,
        learning_rate: learningRate.trim() ? parseFloat(learningRate) : undefined,
        device: device.trim() || undefined,
        extra_args,
      });
    },
    onSuccess: (job) => {
      setSelectedJobId(job.id);
      queryClient.invalidateQueries({ queryKey: ["training-jobs", projectId] });
    },
  });

  function onExtraArgsChange(value: string) {
    setExtraArgsJson(value);
    if (!value.trim()) {
      setExtraArgsError(null);
      return;
    }
    try {
      const parsed = JSON.parse(value);
      setExtraArgsError(
        parsed && typeof parsed === "object" && !Array.isArray(parsed) ? null : "Must be a JSON object, e.g. {\"patience\": 20}",
      );
    } catch {
      setExtraArgsError("Not valid JSON");
    }
  }

  if (!projectId) return null;

  const gpu = providersQuery.data?.gpu;
  const kaggleAvailable = providersQuery.data?.available.includes("KAGGLE") ?? false;
  const jobs = jobsQuery.data ?? [];
  const activeJob = jobs.find((j) => j.id === selectedJobId) ?? jobs[0];

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Training runs</SectionLabel>
      <h1 className="mb-8 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Training Runs
      </h1>

      {gpu && (
        <div className="mb-8 max-w-3xl border-2 border-ink p-4">
          <p className="text-[10px] font-bold uppercase tracking-widest text-ink/50">GPU</p>
          <p className="tabular text-sm font-semibold">
            {gpu.cuda_available ? gpu.device_name : "No CUDA GPU detected — will train on CPU"}
            {gpu.vram_total_mb && ` · ${(gpu.vram_total_mb / 1024).toFixed(1)} GB VRAM`}
            {gpu.cuda_version && ` · CUDA ${gpu.cuda_version}`} · torch {gpu.torch_version}
          </p>
        </div>
      )}

      <div className="mb-8 max-w-3xl space-y-4 border-2 border-ink p-6">
        <div className="flex gap-6">
          {(["LOCAL", "KAGGLE"] as const).map((p) => (
            <label key={p} className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
              <input
                type="radio"
                checked={provider === p}
                disabled={p === "KAGGLE" && !kaggleAvailable}
                onChange={() => setProvider(p)}
                className="accent-accent"
              />
              {p}
              {p === "KAGGLE" && !kaggleAvailable && (
                <span className="text-[9px] font-normal normal-case text-ink/40">
                  (disabled — no Kaggle credentials configured)
                </span>
              )}
            </label>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => {
                setDatasetId(e.target.value);
                setVersionId("");
              }}
              className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
            >
              <option value="">Select…</option>
              {(datasetsQuery.data ?? []).map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">
              Dataset version
            </label>
            <select
              value={versionId}
              onChange={(e) => setVersionId(e.target.value)}
              disabled={!datasetId}
              className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent disabled:opacity-40"
            >
              <option value="">Select…</option>
              {(versionsQuery.data ?? []).map((v) => (
                <option key={v.id} value={v.id}>
                  v{v.version_number} ({v.total_images} images)
                </option>
              ))}
            </select>
            {/* A dataset version is a separate, deliberate step (Export
                page → "Create version") — it's not created for you just by
                approving images in Review, so an empty dropdown here isn't
                a bug, it's "you haven't done that step yet." Say so instead
                of leaving a silently-empty <select>. */}
            {datasetId && !versionsQuery.isLoading && versionsQuery.data?.length === 0 && (
              <p className="mt-1 text-[10px] text-ink/50">
                No versions yet for this dataset —{" "}
                <Link to={`/projects/${projectId}/export`} className="underline hover:text-accent">
                  create one on the Export page
                </Link>{" "}
                first (approve some images in Review, then "Create version").
              </p>
            )}
          </div>
        </div>

        <div>
          <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Base model
          </label>
          <select
            value={baseModelId}
            onChange={(e) => setBaseModelId(e.target.value)}
            className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
          >
            <option value="">Select…</option>
            {detectorModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} {m.version}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Epochs
            <input
              type="number"
              min={1}
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
              className="tabular mt-1 w-full border border-ink/30 px-2 py-1 text-sm"
            />
          </label>
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Batch size
            <input
              type="number"
              min={1}
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
              className="tabular mt-1 w-full border border-ink/30 px-2 py-1 text-sm"
            />
          </label>
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Image size
            <input
              type="number"
              min={32}
              step={32}
              value={imageSize}
              onChange={(e) => setImageSize(parseInt(e.target.value) || 640)}
              className="tabular mt-1 w-full border border-ink/30 px-2 py-1 text-sm"
            />
          </label>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Learning rate (lr0)
            <input
              type="number"
              min={0}
              step={0.0001}
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
              placeholder="auto (0.01)"
              className="tabular mt-1 w-full border border-ink/30 px-2 py-1 text-sm placeholder:normal-case placeholder:text-ink/30"
            />
          </label>
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Device
            <input
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              placeholder="0, 0,1, or cpu"
              className="tabular mt-1 w-full border border-ink/30 px-2 py-1 text-sm placeholder:normal-case placeholder:text-ink/30"
            />
          </label>
        </div>

        <div>
          <label className="mb-1 flex items-baseline justify-between text-[10px] font-bold uppercase tracking-widest text-ink/50">
            <span>Advanced YOLO parameters (JSON)</span>
            <a
              href="https://docs.ultralytics.com/modes/train/#train-settings"
              target="_blank"
              rel="noreferrer"
              className="font-normal normal-case text-ink/40 underline"
            >
              Ultralytics train() reference →
            </a>
          </label>
          <textarea
            value={extraArgsJson}
            onChange={(e) => onExtraArgsChange(e.target.value)}
            placeholder='{"optimizer": "AdamW", "patience": 20, "dropout": 0.1, "mosaic": 0.5}'
            rows={3}
            spellCheck={false}
            className="tabular w-full border border-ink/30 bg-paper px-2 py-1.5 text-xs outline-none placeholder:text-ink/30 focus:border-accent"
          />
          <p className="mt-1 text-[10px] text-ink/40">
            Any other <span className="tabular">YOLO.train()</span> keyword — optimizer, patience, dropout,
            augmentation knobs, and everything else not already a field above. Epochs/batch/image
            size/learning rate/device here always win over the same key typed in this box.
          </p>
          {extraArgsError && <p className="mt-1 text-xs text-accent">{extraArgsError}</p>}
        </div>

        <button
          onClick={() => createMutation.mutate()}
          disabled={!versionId || !baseModelId || !!extraArgsError || createMutation.isPending}
          className="w-full border-2 border-ink bg-ink py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {createMutation.isPending ? "Starting…" : `Start ${provider === "LOCAL" ? "local" : "Kaggle"} training`}
        </button>
        {createMutation.isError && (
          <p className="text-xs text-accent">{(createMutation.error as Error).message}</p>
        )}
      </div>

      {activeJob && (
        <div className="mb-8 max-w-3xl">
          <JobDetail key={activeJob.id} job={activeJob} />
        </div>
      )}

      {jobs.length > 1 && (
        <div className="max-w-3xl">
          <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/50">Previous runs</p>
          {jobs
            .filter((j) => j.id !== activeJob?.id)
            .map((j) => (
              <button
                key={j.id}
                onClick={() => setSelectedJobId(j.id)}
                className="flex w-full items-center justify-between border-b-2 border-ink py-3 text-left hover:bg-muted"
              >
                <span className="tabular text-xs">
                  Epoch {j.current_epoch}/{j.epochs} · {new Date(j.created_at).toLocaleString()}
                </span>
                <span className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${STATUS_STYLE[j.status]}`}>
                  {j.status}
                </span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
