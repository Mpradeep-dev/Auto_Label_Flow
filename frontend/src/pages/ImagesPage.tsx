import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { EmptyState } from "@/components/layout/EmptyState";
import { Skeleton } from "@/components/layout/Skeleton";
import type { AnnotationImage, ImageReviewStatus } from "@/types";

const STATUS_LABEL: Record<string, string> = {
  PENDING: "Pending",
  APPROVED: "Approved",
  REJECTED: "Rejected",
};

type StatusTab = "ALL" | ImageReviewStatus;

const PAGE_SIZE = 60;

function ImageCard({
  image,
  projectId,
  datasetId,
  annotateQuery,
  onDeleted,
}: {
  image: AnnotationImage;
  projectId: string;
  datasetId: string;
  annotateQuery: string;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteImage(image.id),
    onSuccess: onDeleted,
  });

  return (
    <div className="group relative block border-4 border-plate bg-plate">
      <Link
        to={`/projects/${projectId}/datasets/${datasetId}/images/${image.id}/annotate${annotateQuery}`}
        className="block"
      >
        <div className="aspect-video overflow-hidden">
          <img
            src={image.url}
            alt={image.original_filename}
            className="h-full w-full object-cover transition-transform duration-150 group-hover:scale-105"
          />
        </div>
        <div className="flex items-center justify-between bg-paper px-2 py-1.5">
          <span className="truncate text-[10px] font-bold uppercase tracking-widest text-ink/70">
            {STATUS_LABEL[image.review_status]}
          </span>
          <span className="tabular text-[10px] text-ink/40">
            {image.width}×{image.height}
          </span>
        </div>
      </Link>

      {confirming ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-ink/90 p-3 text-center">
          <p className="text-[10px] font-bold uppercase tracking-widest text-paper">Delete this image?</p>
          <div className="flex gap-2">
            <button
              onClick={(e) => {
                e.preventDefault();
                deleteMutation.mutate();
              }}
              disabled={deleteMutation.isPending}
              className="border border-accent bg-accent px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-paper hover:bg-paper hover:text-accent disabled:opacity-40"
            >
              {deleteMutation.isPending ? "…" : "Delete"}
            </button>
            <button
              onClick={(e) => {
                e.preventDefault();
                setConfirming(false);
              }}
              className="border border-paper/40 px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-paper hover:border-paper"
            >
              Cancel
            </button>
          </div>
          {deleteMutation.isError && (
            <p className="text-[9px] text-accent">{(deleteMutation.error as Error).message}</p>
          )}
        </div>
      ) : (
        <button
          onClick={(e) => {
            e.preventDefault();
            setConfirming(true);
          }}
          className="absolute right-1 top-1 border border-paper/60 bg-ink/70 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-paper opacity-0 hover:border-accent hover:text-accent group-hover:opacity-100"
        >
          ✕
        </button>
      )}
    </div>
  );
}

export function ImagesPage() {
  // `datasetId` is only present on the dataset-scoped route
  // (/datasets/:datasetId/images, reached by clicking a dataset). Reached
  // via the sidebar instead (/projects/:projectId/images, no datasetId) —
  // a project can hold several datasets, so there's no single "the"
  // images list at that level; a picker fills the gap instead of the dead
  // end this page used to be (a PlaceholderPage telling you to go find a
  // dataset first).
  const { projectId, datasetId: datasetIdParam } = useParams<{ projectId: string; datasetId?: string }>();
  const [pickedDatasetId, setPickedDatasetId] = useState("");
  const datasetId = datasetIdParam || pickedDatasetId;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(0);
  const [failedUploads, setFailedUploads] = useState<{ name: string; message: string }[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [page, setPage] = useState(0);
  // Split by review_status instead of one undifferentiated gallery — an
  // approved image has nothing left to do, so mixing it back in with what's
  // still pending made "did I actually finish this dataset?" impossible to
  // answer from this page (same reasoning ReviewQueuePage's Pending/Approved
  // split already uses; this just extends the same idea here, plus Rejected).
  const [tab, setTab] = useState<StatusTab>("ALL");
  const queryClient = useQueryClient();

  // A dataset or tab switch invalidates whatever page we were on — the old
  // page index is meaningless (and can point past the end) against a
  // different result set.
  useEffect(() => {
    setPage(0);
  }, [datasetId, tab]);

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId && !datasetIdParam,
  });

  const imagesQuery = useQuery({
    queryKey: ["images", datasetId, page, tab],
    queryFn: () => api.listImages(datasetId!, PAGE_SIZE, page * PAGE_SIZE, tab === "ALL" ? undefined : tab),
    enabled: !!datasetId,
  });

  // Every tab's total is fetched (not just the active one, `limit: 1` since
  // only `total` is needed) so the tab bar shows live counts and switching
  // feels instant instead of a blank "…" until you click it — same "fetch
  // every tab for its count" pattern ReviewQueuePage already uses.
  const pendingCountQuery = useQuery({
    queryKey: ["images", datasetId, "count", "PENDING"],
    queryFn: () => api.listImages(datasetId!, 1, 0, "PENDING"),
    enabled: !!datasetId,
  });
  const approvedCountQuery = useQuery({
    queryKey: ["images", datasetId, "count", "APPROVED"],
    queryFn: () => api.listImages(datasetId!, 1, 0, "APPROVED"),
    enabled: !!datasetId,
  });
  const rejectedCountQuery = useQuery({
    queryKey: ["images", datasetId, "count", "REJECTED"],
    queryFn: () => api.listImages(datasetId!, 1, 0, "REJECTED"),
    enabled: !!datasetId,
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadImage(datasetId!, file),
  });

  async function handleFiles(files: FileList | null) {
    if (!files || !datasetId) return;
    const list = Array.from(files);
    setUploading(list.length);
    setFailedUploads([]);
    // Uploaded concurrently (was a sequential `await` loop — N files meant
    // N round trips back to back) and each failure is now collected
    // instead of only `console.error`'d: a partial failure used to leave
    // the page showing the files that succeeded with zero indication
    // anything was missing (audit finding FE-04).
    const failures: { name: string; message: string }[] = [];
    await Promise.all(
      list.map(async (file) => {
        try {
          await uploadMutation.mutateAsync(file);
        } catch (err) {
          failures.push({ name: file.name, message: err instanceof ApiError ? err.message : "Upload failed" });
        } finally {
          setUploading((n) => n - 1);
        }
      }),
    );
    setFailedUploads(failures);
    setPage(0);
    queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
    queryClient.invalidateQueries({ queryKey: ["dataset-stats", datasetId] });
  }

  function onImageDeleted() {
    queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
    queryClient.invalidateQueries({ queryKey: ["dataset-stats", datasetId] });
  }

  if (!projectId) return null;

  const images = imagesQuery.data?.items ?? [];
  const total = imagesQuery.data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const pendingTotal = pendingCountQuery.data?.total;
  const approvedTotal = approvedCountQuery.data?.total;
  const rejectedTotal = rejectedCountQuery.data?.total;
  const allTotal =
    pendingTotal != null && approvedTotal != null && rejectedTotal != null
      ? pendingTotal + approvedTotal + rejectedTotal
      : undefined;
  // Carried into AnnotatePage so its Prev/Next/Filmstrip stay scoped to
  // this same tab instead of falling back to the full, unfiltered dataset —
  // same mechanism ReviewQueuePage's own tabs already use
  // (`listAllReviewQueueImageIds`/`fromReviewQueue`), just for a plain
  // status filter instead of a difficulty-ordered queue.
  const annotateQuery = tab === "ALL" ? "" : `?from=images&reviewStatus=${tab}`;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={3}>Images</SectionLabel>
      <div className="mb-12 flex flex-wrap items-end justify-between gap-4 border-b-4 border-ink pb-8">
        <h1 className="text-5xl font-black uppercase tracking-tightest sm:text-7xl">Images</h1>
        <div className="flex flex-wrap items-end gap-3">
          {!datasetIdParam && (
            <div>
              <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">
                Dataset
              </label>
              <select
                value={pickedDatasetId}
                onChange={(e) => setPickedDatasetId(e.target.value)}
                className="border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
              >
                <option value="">Select a dataset…</option>
                {(datasetsQuery.data ?? []).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          {uploading > 0 && (
            <span className="text-xs font-bold uppercase tracking-widest text-ink/50">
              Uploading {uploading}…
            </span>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/bmp"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!datasetId}
            className="border-2 border-ink bg-ink px-6 py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
          >
            Upload images
          </button>
        </div>
      </div>

      {failedUploads.length > 0 && (
        <div className="mb-8 border-2 border-accent bg-accent/5 px-4 py-3 text-xs">
          <div className="flex items-start justify-between gap-4">
            <p className="font-bold uppercase tracking-widest text-accent">
              {failedUploads.length} file{failedUploads.length === 1 ? "" : "s"} failed to upload
            </p>
            <button
              onClick={() => setFailedUploads([])}
              className="font-bold uppercase tracking-widest text-ink/40 hover:text-ink"
            >
              Dismiss
            </button>
          </div>
          <ul className="mt-2 space-y-1 text-ink/70">
            {failedUploads.map((f, i) => (
              <li key={i}>
                <span className="font-semibold">{f.name}</span> — {f.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!datasetId ? (
        <EmptyState
          title="Pick a dataset"
          description="Select a dataset from the dropdown above to see or upload its images."
        />
      ) : (
        <>
          <div className="mb-6 flex border-2 border-ink">
            {(
              [
                ["ALL", "All", allTotal],
                ["PENDING", "Pending", pendingTotal],
                ["APPROVED", "Approved", approvedTotal],
                ["REJECTED", "Rejected", rejectedTotal],
              ] as const
            ).map(([value, label, count]) => (
              <button
                key={value}
                onClick={() => setTab(value)}
                className={`flex-1 border-r-2 border-ink px-4 py-3 text-xs font-bold uppercase tracking-widest last:border-r-0 ${
                  tab === value ? "bg-ink text-paper" : "hover:bg-muted"
                }`}
              >
                {label} <span className="tabular">{count ?? "…"}</span>
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {imagesQuery.isLoading &&
              Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="aspect-video border-4 border-plate" />)}
            {images.map((image) => (
              <ImageCard
                key={image.id}
                image={image}
                projectId={projectId}
                datasetId={datasetId}
                annotateQuery={annotateQuery}
                onDeleted={onImageDeleted}
              />
            ))}
            {images.length === 0 && !imagesQuery.isLoading && (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragActive(false);
                  handleFiles(e.dataTransfer.files);
                }}
                className={`col-span-full ${dragActive ? "border-2 border-accent bg-muted" : ""}`}
              >
                <EmptyState
                  title="No images yet"
                  description="Drag and drop image files anywhere in this area, or use the upload button above to get started."
                >
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-ink bg-ink px-5 py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
                  >
                    Upload images
                  </button>
                </EmptyState>
              </div>
            )}
          </div>

          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-4">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-muted disabled:opacity-30"
              >
                ← Prev
              </button>
              <span className="tabular text-xs font-bold uppercase tracking-widest text-ink/50">
                Page {page + 1} of {totalPages} · {total} images
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-muted disabled:opacity-30"
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
