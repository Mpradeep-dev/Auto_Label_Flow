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
          className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:bg-ink hover:text-paper"
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

  if (!projectId) return null;

  const project = projectQuery.data;
  const stats = statsQuery.data;
  const hasImages = (stats?.total_images ?? 0) > 0;
  const reviewed = stats?.approved_images ?? 0;
  const total = stats?.total_images ?? 0;
  const pct = total > 0 ? Math.round((reviewed / total) * 100) : 0;

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
      state: hasImages ? "not_started" : "blocked",
      detail: hasImages ? "Run the registered detector over this dataset" : "Import media first",
      to: hasImages ? `/projects/${projectId}/auto-annotation` : undefined,
    },
    {
      label: "Review",
      state: "blocked",
      detail:
        total > 0
          ? `${reviewed} / ${total} reviewed`
          : "Blocked until predictions exist",
      to: hasImages ? `/projects/${projectId}/review` : undefined,
    },
    { label: "Version", state: "blocked", detail: "Create a dataset version once review is underway" },
    { label: "Train", state: "blocked", detail: "Local (RTX) or Kaggle, once a version exists" },
    { label: "New model", state: "blocked", detail: "Registers and becomes available for auto-annotation" },
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
