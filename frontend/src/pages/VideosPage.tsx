import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { VideoRecord } from "@/types";

// Same reasoning as AutoAnnotationPage's selectionStorageKey: the dataset
// dropdown here is plain form state with nothing else to restore it from —
// without persisting it, navigating away and back reset it to empty and the
// whole video list (including anything still extracting) appeared to vanish.
function selectionStorageKey(projectId: string): string {
  return `videos-selection:${projectId}`;
}

const STATUS_LABEL: Record<string, string> = {
  UPLOADED: "Uploaded",
  EXTRACTING: "Extracting…",
  EXTRACTED: "Extracted",
  FAILED: "Failed",
};

function VideoRow({ video, onExtract }: { video: VideoRecord; onExtract: (id: string) => void }) {
  const isExtracting = video.status === "EXTRACTING";

  const pollingQuery = useQuery({
    queryKey: ["video-poll", video.id],
    queryFn: () => api.listVideos(video.dataset_id).then((vs) => vs.find((v) => v.id === video.id) ?? video),
    enabled: isExtracting,
    refetchInterval: isExtracting ? 1500 : false,
  });
  const current = pollingQuery.data ?? video;

  return (
    <div className="flex items-center justify-between border-b-2 border-ink p-6">
      <div>
        <p className="text-lg font-bold uppercase tracking-tight">{current.original_filename}</p>
        <p className="tabular mt-1 text-xs uppercase tracking-widest text-ink/50">
          {current.width}×{current.height} · {current.fps?.toFixed(1)} fps ·{" "}
          {current.duration_s?.toFixed(1)}s · {current.total_frames} frames
        </p>
      </div>
      <div className="flex items-center gap-4">
        <span className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest bg-muted">
          {STATUS_LABEL[current.status]}
          {current.status === "EXTRACTED" && ` · ${current.extracted_frame_count}`}
        </span>
        {current.status === "UPLOADED" && (
          <button
            onClick={() => onExtract(current.id)}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
          >
            Extract frames
          </button>
        )}
      </div>
    </div>
  );
}

export function VideosPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [interval, setInterval_] = useState(5);
  const [uploading, setUploading] = useState(false);
  const [datasetId, setDatasetId] = useState(() => {
    if (!projectId) return "";
    try {
      return localStorage.getItem(selectionStorageKey(projectId)) ?? "";
    } catch {
      return "";
    }
  });
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!projectId) return;
    try {
      localStorage.setItem(selectionStorageKey(projectId), datasetId);
    } catch {
      /* private-browsing or storage-disabled — losing the remembered selection is harmless */
    }
  }, [projectId, datasetId]);

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });

  const videosQuery = useQuery({
    queryKey: ["videos", datasetId],
    queryFn: () => api.listVideos(datasetId),
    enabled: !!datasetId,
  });

  const extractMutation = useMutation({
    mutationFn: (videoId: string) => api.extractFrames(videoId, { interval }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["videos", datasetId] }),
  });

  async function handleFile(file: File | null) {
    if (!file || !datasetId) return;
    setUploading(true);
    try {
      await api.uploadVideo(datasetId, file);
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
    } finally {
      setUploading(false);
    }
  }

  if (!projectId) return null;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={4}>Videos</SectionLabel>
      <div className="mb-8 border-b-4 border-ink pb-8">
        <h1 className="mb-6 text-5xl font-black uppercase tracking-tightest sm:text-7xl">Videos</h1>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">
              Dataset
            </label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
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
          <label className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Sample every
            <input
              type="number"
              min={1}
              value={interval}
              onChange={(e) => setInterval_(Math.max(1, parseInt(e.target.value) || 1))}
              className="tabular mx-2 w-14 border border-ink/30 px-1.5 py-0.5 text-center text-xs"
            />
            frames
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,.mov,.avi,.mkv"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || !datasetId}
            className="border-2 border-ink bg-ink px-6 py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
          >
            {uploading ? "Uploading…" : "Upload video"}
          </button>
        </div>
      </div>

      {!datasetId && <p className="text-sm text-ink/50">Select a dataset above to see or upload its videos.</p>}

      <div className="max-w-4xl border-t-2 border-ink">
        {(videosQuery.data ?? []).map((video) => (
          <VideoRow key={video.id} video={video} onExtract={(id) => extractMutation.mutate(id)} />
        ))}
        {videosQuery.data?.length === 0 && (
          <p className="py-8 text-sm text-ink/50">No videos yet — upload one to get started.</p>
        )}
      </div>
    </div>
  );
}
