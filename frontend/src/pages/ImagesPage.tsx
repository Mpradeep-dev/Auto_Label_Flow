import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";

const STATUS_LABEL: Record<string, string> = {
  PENDING: "Pending",
  APPROVED: "Approved",
  REJECTED: "Rejected",
};

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
  const queryClient = useQueryClient();

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId && !datasetIdParam,
  });

  const imagesQuery = useQuery({
    queryKey: ["images", datasetId],
    queryFn: () => api.listImages(datasetId!, 100, 0),
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
    queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
    queryClient.invalidateQueries({ queryKey: ["dataset-stats", datasetId] });
  }

  if (!projectId) return null;

  const images = imagesQuery.data?.items ?? [];

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
        <p className="text-sm text-ink/50">Select a dataset above to see or upload its images.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {images.map((image) => (
            <Link
              key={image.id}
              to={`/projects/${projectId}/datasets/${datasetId}/images/${image.id}/annotate`}
              className="group block border-4 border-plate bg-plate"
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
          ))}
          {images.length === 0 && !imagesQuery.isLoading && (
            <p className="col-span-full py-8 text-sm text-ink/50">
              No images yet — upload some to get started.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
