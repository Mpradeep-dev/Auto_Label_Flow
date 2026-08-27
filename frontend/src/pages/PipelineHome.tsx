import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";

type StageState = "done" | "in_progress" | "blocked" | "not_started";

interface Stage {
  label: string;
  state: StageState;
  detail: string;
  to?: string;
}

const STATE_STYLE: Record<StageState, string> = {
  done: "border-ink bg-ink text-paper",
  in_progress: "border-ink bg-paper text-ink",
  blocked: "border-ink/30 bg-muted text-ink/50",
  not_started: "border-ink/30 bg-paper text-ink/40",
};

function StageRow({ index, stage }: { index: number; stage: Stage }) {
  const dot = (
    <span
      className={`flex h-8 w-8 shrink-0 items-center justify-center border-2 text-xs font-bold ${STATE_STYLE[stage.state]}`}
    >
      {index}
    </span>
  );

  const body = (
    <div className="flex flex-1 items-center justify-between border-b-2 border-ink/10 py-6">
      <div>
        <p className="text-xl font-bold uppercase tracking-tight sm:text-2xl">{stage.label}</p>
        <p className="mt-1 text-sm text-ink/60">{stage.detail}</p>
      </div>
      {stage.to && stage.state !== "blocked" && (
        <Link
          to={stage.to}
          className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-paper"
        >
          {stage.state === "done" || stage.state === "in_progress" ? "Continue →" : "Start →"}
        </Link>
      )}
    </div>
  );

  return (
    <div className="flex gap-4">
      {dot}
      {body}
    </div>
  );
}

export function PipelineHome() {
  const { projectId } = useParams<{ projectId: string }>();

  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId!),
    enabled: !!projectId,
  });

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });

  const statsQuery = useQuery({
    queryKey: ["dataset-stats", projectId, datasetsQuery.data?.map((d) => d.id)],
    queryFn: async () => {
      const datasets = datasetsQuery.data ?? [];
      const all = await Promise.all(datasets.map((d) => api.getDatasetStats(d.id)));
      return all.reduce(
        (acc, s) => ({
          total_images: acc.total_images + s.total_images,
          pending_images: acc.pending_images + s.pending_images,
          approved_images: acc.approved_images + s.approved_images,
          total_videos: acc.total_videos + s.total_videos,
        }),
        { total_images: 0, pending_images: 0, approved_images: 0, total_videos: 0 },
      );
    },
    enabled: !!datasetsQuery.data,
  });

  // Versions live per-dataset, not per-project — aggregate the same way as
  // stats above so the Version stage can react to "does any dataset in
  // this project have an exported version" rather than being hardcoded.
  const versionsQuery = useQuery({
    queryKey: ["dataset-versions", projectId, datasetsQuery.data?.map((d) => d.id)],
    queryFn: async () => {
      const datasets = datasetsQuery.data ?? [];
      const all = await Promise.all(datasets.map((d) => api.listDatasetVersions(d.id)));
      return all.flat();
    },
    enabled: !!datasetsQuery.data,
  });

  const inferenceJobsQuery = useQuery({
    queryKey: ["inference-jobs", projectId],
    queryFn: () => api.listInferenceJobs(projectId!),
    enabled: !!projectId,
  });

  const trainingJobsQuery = useQuery({
    queryKey: ["training-jobs", projectId],
    queryFn: () => api.listTrainingJobs(projectId!),
    enabled: !!projectId,
  });

  if (!projectId) return null;

  const project = projectQuery.data;
  const stats = statsQuery.data;
  const hasImages = (stats?.total_images ?? 0) > 0;
  const reviewed = stats?.approved_images ?? 0;
  const total = stats?.total_images ?? 0;
  const pct = total > 0 ? Math.round((reviewed / total) * 100) : 0;

  const inferenceJobs = inferenceJobsQuery.data ?? [];
  const hasCompletedInference = inferenceJobs.some((j) => j.status === "COMPLETED");
  const inferenceRunning = inferenceJobs.some((j) => j.status === "RUNNING" || j.status === "QUEUED");
  const totalPredictions = inferenceJobs.reduce((n, j) => n + j.total_predictions, 0);
  const hasPredictions = hasCompletedInference || totalPredictions > 0;

  const versions = versionsQuery.data ?? [];
  const exportedVersion = versions.find((v) => v.status === "EXPORTED");
  const hasVersion = versions.length > 0;

  const trainingJobs = trainingJobsQuery.data ?? [];
  const trainingRunning = trainingJobs.some((j) => j.status === "RUNNING" || j.status === "QUEUED");
  const completedTraining = trainingJobs.find((j) => j.status === "COMPLETED");
  const trainedModelId = trainingJobs.find((j) => j.status === "COMPLETED" && j.result_model_id)?.result_model_id;

  const autoAnnotateState: StageState = !hasImages
    ? "blocked"
    : inferenceRunning
      ? "in_progress"
      : hasCompletedInference
        ? "done"
        : "not_started";

  const reviewState: StageState = !hasPredictions
    ? "blocked"
    : total > 0 && reviewed === total
      ? "done"
      : reviewed > 0
        ? "in_progress"
        : "not_started";

  const versionState: StageState = hasVersion ? "done" : reviewed > 0 ? "not_started" : "blocked";

  const trainState: StageState = !hasVersion
    ? "blocked"
    : trainingRunning
      ? "in_progress"
      : completedTraining
        ? "done"
        : "not_started";

  const modelState: StageState = trainedModelId ? "done" : trainingRunning ? "in_progress" : "blocked";

  const stages: Stage[] = [
    {
      label: "Import",
      state: hasImages ? "done" : "in_progress",
      detail: hasImages
        ? `${total} image${total === 1 ? "" : "s"} · ${stats?.total_videos ?? 0} video${stats?.total_videos === 1 ? "" : "s"}`
        : "Upload images or video to get started",
      to: `/projects/${projectId}/datasets`,
    },
    {
      label: "Auto-annotate",
      state: autoAnnotateState,
      detail: !hasImages
        ? "Import media first"
        : inferenceRunning
          ? "Running…"
          : hasCompletedInference
            ? `${totalPredictions} prediction${totalPredictions === 1 ? "" : "s"} generated`
            : "Run the registered detector over this dataset",
      to: hasImages ? `/projects/${projectId}/auto-annotation` : undefined,
    },
    {
      label: "Review",
      state: reviewState,
      detail: !hasPredictions
        ? "Blocked until predictions exist"
        : total > 0
          ? `${reviewed} / ${total} reviewed`
          : "Predictions ready to review",
      to: hasPredictions ? `/projects/${projectId}/review` : undefined,
    },
    {
      label: "Version",
      state: versionState,
      detail: hasVersion
        ? `v${Math.max(...versions.map((v) => v.version_number))} · ${exportedVersion ? "exported" : "draft"}`
        : reviewed > 0
          ? "Create a dataset version once review is underway"
          : "Blocked until images are reviewed",
      to: reviewed > 0 ? `/projects/${projectId}/export` : undefined,
    },
    {
      label: "Train",
      state: trainState,
      detail: !hasVersion
        ? "Local (RTX) or Kaggle, once a version exists"
        : trainingRunning
          ? `Epoch ${trainingJobs.find((j) => j.status === "RUNNING")?.current_epoch ?? 0}…`
          : completedTraining
            ? "Training run completed"
            : "Ready to start a training run",
      to: hasVersion ? `/projects/${projectId}/training` : undefined,
    },
    {
      label: "New model",
      state: modelState,
      detail: trainedModelId
        ? "Registered and available for auto-annotation"
        : trainingRunning
          ? "Registers once training completes"
          : "Registers and becomes available for auto-annotation",
      to: trainedModelId ? `/projects/${projectId}/models` : undefined,
    },
  ];

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Pipeline</SectionLabel>
      <div className="mb-12 flex flex-wrap items-end justify-between gap-4 border-b-4 border-ink pb-8">
        <h1 className="text-5xl font-black uppercase tracking-tightest sm:text-7xl">
          {project?.name ?? "…"}
        </h1>
        <div className="text-right">
          <p className="tabular text-4xl font-black">
            {total > 0 ? `${reviewed}⁄${total}` : "—"}
          </p>
          <p className="text-xs font-bold uppercase tracking-widest text-ink/50">
            {total > 0 ? `${pct}% reviewed` : "no images yet"}
          </p>
        </div>
      </div>

      <div className="max-w-4xl">
        {stages.map((stage, i) => (
          <StageRow key={stage.label} index={i + 1} stage={stage} />
        ))}
      </div>
    </div>
  );
}
