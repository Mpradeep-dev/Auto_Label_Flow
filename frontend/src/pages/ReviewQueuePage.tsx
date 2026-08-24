import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { FlagType } from "@/types";

const FLAG_TYPES: FlagType[] = [
  "SUSPICIOUS_CONE",
  "CONE_NEAR_PLAYER",
  "VERY_SMALL_CONE",
  "LOW_CONFIDENCE",
  "POSSIBLE_DUPLICATE",
  "ISOLATED_DETECTION",
  "TEMPORAL_ANOMALY",
];

export function ReviewQueuePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [flagType, setFlagType] = useState<FlagType | "">("");
  const [analyzeDatasetId, setAnalyzeDatasetId] = useState("");
  const queryClient = useQueryClient();

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });

  const queueQuery = useQuery({
    queryKey: ["review-queue", projectId, flagType],
    queryFn: () =>
      api.getReviewQueue({ project_id: projectId!, flag_type: flagType || undefined, limit: 60 }),
    enabled: !!projectId,
  });

  const analyzeMutation = useMutation({
    mutationFn: () => api.analyzeDatasetQuality(analyzeDatasetId),
    onSuccess: () => {
      // Background job — give it a moment, then refresh. Not exact, but
      // this is a manual trigger the reviewer can re-run if it's early.
      setTimeout(() => queryClient.invalidateQueries({ queryKey: ["review-queue", projectId] }), 3000);
    },
  });

  const datasetById = new Map((datasetsQuery.data ?? []).map((d) => [d.id, d]));

  if (!projectId) return null;

  const items = queueQuery.data?.items ?? [];

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Review queue</SectionLabel>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b-4 border-ink pb-8">
        <h1 className="text-5xl font-black uppercase tracking-tightest sm:text-7xl">Review Queue</h1>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={analyzeDatasetId}
            onChange={(e) => setAnalyzeDatasetId(e.target.value)}
            className="border-2 border-ink bg-paper px-3 py-2 text-xs font-bold uppercase tracking-widest outline-none focus:border-accent"
          >
            <option value="">Select dataset to analyze…</option>
            {(datasetsQuery.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
          <button
            onClick={() => analyzeMutation.mutate()}
            disabled={!analyzeDatasetId || analyzeMutation.isPending}
            className="border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
          >
            {analyzeMutation.isPending ? "Queued…" : "Run quality analysis"}
          </button>
          <select
            value={flagType}
            onChange={(e) => setFlagType(e.target.value as FlagType | "")}
            className="border-2 border-ink bg-paper px-3 py-2 text-xs font-bold uppercase tracking-widest outline-none focus:border-accent"
          >
            <option value="">All flags</option>
            {FLAG_TYPES.map((ft) => (
              <option key={ft} value={ft}>
                {ft.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>
      </div>

      <p className="tabular mb-6 text-xs uppercase tracking-widest text-ink/50">
        {queueQuery.data?.total ?? 0} images · sorted by difficulty, most suspicious first
      </p>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {items.map((item) => {
          const dataset = datasetById.get(item.dataset_id);
          const unresolved = item.flags.filter((f) => !f.resolution);
          return (
            <Link
              key={item.image_id}
              to={
                dataset
                  ? `/projects/${projectId}/datasets/${item.dataset_id}/images/${item.image_id}/annotate`
                  : "#"
              }
              className="group block border-4 border-plate bg-plate"
            >
              <div className="aspect-video overflow-hidden">
                <img
                  src={item.url}
                  alt=""
                  className="h-full w-full object-cover transition-transform duration-150 group-hover:scale-105"
                />
              </div>
              <div className="bg-paper px-2 py-1.5">
                <div className="flex items-center justify-between">
                  <span className="tabular text-[10px] font-bold uppercase tracking-widest text-ink/70">
                    {item.difficulty_score != null ? item.difficulty_score.toFixed(2) : "—"}
                  </span>
                  {unresolved.length > 0 && (
                    <span className="bg-accent px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest text-paper">
                      ⚠ {unresolved.length}
                    </span>
                  )}
                </div>
                {unresolved[0] && (
                  <p className="mt-1 truncate text-[9px] uppercase tracking-widest text-ink/40">
                    {unresolved[0].flag_type.replace(/_/g, " ")}
                  </p>
                )}
              </div>
            </Link>
          );
        })}
        {items.length === 0 && !queueQuery.isLoading && (
          <p className="col-span-full py-8 text-sm text-ink/50">
            Nothing in the queue yet — run quality analysis on a dataset first.
          </p>
        )}
      </div>
    </div>
  );
}
