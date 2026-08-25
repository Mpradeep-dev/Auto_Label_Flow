import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { EmptyState } from "@/components/layout/EmptyState";
import { Skeleton } from "@/components/layout/Skeleton";
import type { AnnotationImage } from "@/types";

const STATUS_LABEL: Record<string, string> = {
  PENDING: "Pending",
  APPROVED: "Approved",
  REJECTED: "Rejected",
};

const PAGE_SIZE = 60;

function ImageCard({
  image,
  projectId,
  datasetId,
  onDeleted,
}: {
  image: AnnotationImage;
  projectId: string;
  datasetId: string;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteImage(image.id),
    onSuccess: onDeleted,
  });

  return (
    <div className="group relative block border-4 border-plate bg-plate">
      <Link to={`/projects/${projectId}/datasets/${datasetId}/images/${image.id}/annotate`} className="block">
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
  const [dragActive, setDragActive] = useState(false);
  const [page, setPage] = useState(0);
  const queryClient = useQueryClient();

  // A dataset switch (via the picker, or navigating to a different
  // dataset-scoped route) invalidates whatever page we were on.
  useEffect(() => {
    setPage(0);
  }, [datasetId]);

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId && !datasetIdParam,
  });

  const imagesQuery = useQuery({
    queryKey: ["images", datasetId, page],
    queryFn: () => api.listImages(datasetId!, PAGE_SIZE, page * PAGE_SIZE),
    enabled: !!datasetId,
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadImage(datasetId!, file),
  });

  async function handleFiles(files: FileList | null) {
    if (!files || !datasetId) return;
    const list = Array.from(files);
    setUploading(list.length);
    for (const file of list) {
      try {
        await uploadMutation.mutateAsync(file);
      } catch (err) {
        console.error("Upload failed", file.name, err);
      } finally {
        setUploading((n) => n - 1);
      }
    }
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

      {!datasetId ? (
        <EmptyState
          title="Pick a dataset"
          description="Select a dataset from the dropdown above to see or upload its images."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {imagesQuery.isLoading &&
              Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="aspect-video border-4 border-plate" />)}
            {images.map((image) => (
              <ImageCard
                key={image.id}
                image={image}
                projectId={projectId}
                datasetId={datasetId}
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
