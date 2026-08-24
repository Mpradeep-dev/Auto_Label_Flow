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
  const { projectId, datasetId } = useParams<{ projectId: string; datasetId: string }>();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(0);
  const queryClient = useQueryClient();

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

  if (!projectId || !datasetId) return null;

  const images = imagesQuery.data?.items ?? [];

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={3}>Images</SectionLabel>
      <div className="mb-12 flex flex-wrap items-end justify-between gap-4 border-b-4 border-ink pb-8">
        <h1 className="text-5xl font-black uppercase tracking-tightest sm:text-7xl">Images</h1>
        <div className="flex items-center gap-3">
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
            className="border-2 border-ink bg-ink px-6 py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
          >
            Upload images
          </button>
        </div>
      </div>

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
    </div>
  );
}
