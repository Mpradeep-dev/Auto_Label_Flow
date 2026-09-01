import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { EmptyState } from "@/components/layout/EmptyState";
import type { FlagType } from "@/types";

const PAGE_SIZE = 60;

const FLAG_TYPES: FlagType[] = [
  "SUSPICIOUS_CONE",
  "CONE_NEAR_PLAYER",
  "VERY_SMALL_CONE",
  "LOW_CONFIDENCE",
  "POSSIBLE_DUPLICATE",
  "ISOLATED_DETECTION",
  "TEMPORAL_ANOMALY",
];

// A project can hold several datasets (e.g. one imported, one an
// auto-annotation run produced) — pooling every dataset's pending/approved
// counts together made the totals here impossible to reconcile with what
// you'd actually reviewed in any one of them. Scoped to one dataset at a
// time instead, same as Auto Annotation/Export/Training Runs already are.
// `dataset_id` was always a supported filter on the queue endpoint; this
// page just never passed it.
function lastDatasetStorageKey(projectId: string): string {
  return `review-queue-last-dataset:${projectId}`;
}

export function ReviewQueuePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams] = useSearchParams();
  const [flagType, setFlagType] = useState<FlagType | "">("");
  // Arrives pre-filled from Auto Annotation's "Start reviewing →" CTA
  // (?datasetId=...); otherwise falls back to whatever was last picked here,
  // same persisted-selection pattern Export/Auto Annotation use.
  const [datasetId, setDatasetId] = useState(() => {
    const fromLink = searchParams.get("datasetId");
    if (fromLink) return fromLink;
    if (!projectId) return "";
    try {
      return localStorage.getItem(lastDatasetStorageKey(projectId)) ?? "";
    } catch {
      return "";
    }
  });
  // Split into two buckets by `review_status` rather than one undifferentiated
  // list — an image a human has already approved has nothing left to review,
  // so mixing it back in with what's still pending made "did approving this
  // actually do anything?" impossible to answer from this page alone.
  const [tab, setTab] = useState<"PENDING" | "APPROVED">("PENDING");
  const [page, setPage] = useState(0);
  const queryClient = useQueryClient();

  // Any of these changing means "different result set" — the page index
  // from the old one is meaningless (and can point past the end) against
  // the new one.
  useEffect(() => {
    setPage(0);
  }, [datasetId, flagType, tab]);

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });

  function selectDataset(id: string) {
    setDatasetId(id);
    if (!projectId) return;
    try {
      if (id) localStorage.setItem(lastDatasetStorageKey(projectId), id);
    } catch {
      /* private-browsing / storage disabled — just won't remember it next time */
    }
  }

  // Both fetched (not just the active tab) so switching tabs is instant and
  // each tab button can show its own live count. Only the active tab's
  // offset actually pages, though — `page` resets to 0 on every tab switch
  // (above), so the inactive one always just wants its first page anyway
  // (its items are never rendered, only its `total` for the tab label).
  const pendingQuery = useQuery({
    queryKey: ["review-queue", projectId, datasetId, flagType, "PENDING", tab === "PENDING" ? page : 0],
    queryFn: () =>
      api.getReviewQueue({
        project_id: projectId!,
        dataset_id: datasetId,
        flag_type: flagType || undefined,
        review_status: "PENDING",
        limit: PAGE_SIZE,
        offset: tab === "PENDING" ? page * PAGE_SIZE : 0,
      }),
    enabled: !!projectId && !!datasetId,
  });
  const approvedQuery = useQuery({
    queryKey: ["review-queue", projectId, datasetId, flagType, "APPROVED", tab === "APPROVED" ? page : 0],
    queryFn: () =>
      api.getReviewQueue({
        project_id: projectId!,
        dataset_id: datasetId,
        flag_type: flagType || undefined,
        review_status: "APPROVED",
        limit: PAGE_SIZE,
        offset: tab === "APPROVED" ? page * PAGE_SIZE : 0,
      }),
    enabled: !!projectId && !!datasetId,
  });
  const queueQuery = tab === "PENDING" ? pendingQuery : approvedQuery;

  const analyzeMutation = useMutation({
    mutationFn: () => api.analyzeDatasetQuality(datasetId),
    onSuccess: () => {
      // Background job — give it a moment, then refresh. Not exact, but
      // this is a manual trigger the reviewer can re-run if it's early.
      setTimeout(
        () => queryClient.invalidateQueries({ queryKey: ["review-queue", projectId, datasetId] }),
        3000,
      );
    },
  });

  const datasetById = new Map((datasetsQuery.data ?? []).map((d) => [d.id, d]));

  if (!projectId) return null;

  const items = queueQuery.data?.items ?? [];
  const total = queueQuery.data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Review queue</SectionLabel>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b-4 border-ink pb-8">
        <h1 className="text-5xl font-black uppercase tracking-tightest sm:text-7xl">Review Queue</h1>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={datasetId}
            onChange={(e) => selectDataset(e.target.value)}
            className="border-2 border-ink bg-paper px-3 py-2 text-xs font-bold uppercase tracking-widest outline-none focus:border-accent"
          >
            <option value="">Select a dataset…</option>
            {(datasetsQuery.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
          <button
            onClick={() => analyzeMutation.mutate()}
            disabled={!datasetId || analyzeMutation.isPending}
            className="border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange hover:text-ink disabled:opacity-40"
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

      {!datasetId ? (
        <EmptyState
          title="Pick a dataset"
          description="Select a dataset from the dropdown above to see its review queue — pending and approved counts are scoped to one dataset at a time, not pooled across the whole project."
        />
      ) : (
        <>
          <div className="mb-6 flex border-2 border-ink">
            {(
              [
                ["PENDING", "Pending review", pendingQuery.data?.total],
                ["APPROVED", "Human approved", approvedQuery.data?.total],
              ] as const
            ).map(([value, label, total]) => (
              <button
                key={value}
                onClick={() => setTab(value)}
                className={`flex-1 border-r-2 border-ink px-4 py-3 text-xs font-bold uppercase tracking-widest last:border-r-0 ${
                  tab === value ? "bg-ink text-paper" : "hover:bg-orange hover:text-ink"
                }`}
              >
                {label} <span className="tabular">{total ?? "…"}</span>
              </button>
            ))}
          </div>

          <p className="tabular mb-6 text-xs uppercase tracking-widest text-ink/60">
            {total} images ·{" "}
            {tab === "PENDING" ? "sorted by difficulty, most suspicious first" : "most recently approved first"}
          </p>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            {items.map((item) => {
              const dataset = datasetById.get(item.dataset_id);
              const unresolved = item.flags.filter((f) => !f.resolution);
              // Carried through so AnnotatePage can browse in this same
              // difficulty-ordered sequence instead of falling back to
              // plain dataset upload order once you open an image — see
              // `listAllReviewQueueImageIds`.
              const annotateQuery = new URLSearchParams({ from: "review-queue", reviewStatus: tab });
              if (flagType) annotateQuery.set("flagType", flagType);
              return (
                <Link
                  key={item.image_id}
                  to={
                    dataset
                      ? `/projects/${projectId}/datasets/${item.dataset_id}/images/${item.image_id}/annotate?${annotateQuery.toString()}`
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
                      <p className="mt-1 truncate text-[9px] uppercase tracking-widest text-ink/60">
                        {unresolved[0].flag_type.replace(/_/g, " ")}
                      </p>
                    )}
                  </div>
                </Link>
              );
            })}
            {items.length === 0 && !queueQuery.isLoading && (
              <p className="col-span-full py-8 text-sm text-ink/60">
                {tab === "PENDING"
                  ? "Nothing pending — run quality analysis on this dataset to populate this queue."
                  : "Nothing approved yet — approve an image from its Annotate page and it'll show up here."}
              </p>
            )}
          </div>

          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-4">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink disabled:opacity-30"
              >
                ← Prev
              </button>
              <span className="tabular text-xs font-bold uppercase tracking-widest text-ink/60">
                Page {page + 1} of {totalPages} · {total} images
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink disabled:opacity-30"
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
