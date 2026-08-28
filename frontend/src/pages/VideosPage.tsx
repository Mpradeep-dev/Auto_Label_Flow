import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
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

function VideoRow({
  video,
  queued,
  onExtract,
  onDelete,
  deleting,
}: {
  video: VideoRecord;
  queued: boolean;
  onExtract: (id: string) => void;
  onDelete: (id: string) => void;
  deleting: boolean;
}) {
  const isExtracting = video.status === "EXTRACTING";
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  // The dev worker runs with concurrency 1 (see celery_app.py), so a video
  // only flips to EXTRACTING once it reaches the front of the queue — until
  // then it's still UPLOADED in the DB even though it's already queued.
  // `queued` (set locally the moment we ask for extraction) keeps polling
  // and hides the trigger button for that whole wait, so "queued but not
  // started yet" doesn't look identical to "never asked".
  const stillWaiting = queued && video.status === "UPLOADED";

  const pollingQuery = useQuery({
    queryKey: ["video-poll", video.id],
    queryFn: () => api.listVideos(video.dataset_id).then((vs) => vs.find((v) => v.id === video.id) ?? video),
    enabled: isExtracting || stillWaiting,
    refetchInterval: isExtracting || stillWaiting ? 1500 : false,
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
          {stillWaiting ? "Queued…" : STATUS_LABEL[current.status]}
          {current.status === "EXTRACTED" && ` · ${current.extracted_frame_count}`}
        </span>
        {current.status === "UPLOADED" && !stillWaiting && (
          <button
            onClick={() => onExtract(current.id)}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
          >
            Extract frames
          </button>
        )}
        {confirmingDelete ? (
          <div className="flex items-center gap-2">
            <button
              onClick={() => onDelete(current.id)}
              disabled={deleting}
              className="border-2 border-accent bg-accent px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:border-orange hover:bg-orange disabled:opacity-40"
            >
              {deleting ? "Removing…" : "Confirm"}
            </button>
            <button
              onClick={() => setConfirmingDelete(false)}
              disabled={deleting}
              className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmingDelete(true)}
            className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest text-ink/60 hover:border-orange hover:text-orange"
          >
            Remove
          </button>
        )}
      </div>
    </div>
  );
}

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"];

function isVideoFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}

export function VideosPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [interval, setInterval_] = useState(5);
  const [uploading, setUploading] = useState(false);
  const [folderProgress, setFolderProgress] = useState<{ done: number; total: number } | null>(null);
  // Upload/extraction failures used to only reach `finally` and reset
  // loading state with zero indication anything went wrong (audit finding
  // FE-04) — this surfaces what failed and why instead.
  const [uploadErrors, setUploadErrors] = useState<{ name: string; message: string }[]>([]);
  const [extractAllError, setExtractAllError] = useState<string | null>(null);
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

  // Videos we've already asked the backend to extract, kept locally because
  // the DB row stays "UPLOADED" until the (concurrency-1) worker actually
  // picks the task up — see VideoRow's `stillWaiting`. Without this, videos
  // still sitting in the queue looked untouched and re-clicking "Extract
  // all" re-enqueued them, so the same video got extracted twice.
  const [queuedIds, setQueuedIds] = useState<Set<string>>(new Set());
  function markQueued(ids: string[]) {
    setQueuedIds((prev) => new Set([...prev, ...ids]));
  }

  const uploadedCount = (videosQuery.data ?? []).filter(
    (v) => v.status === "UPLOADED" && !queuedIds.has(v.id),
  ).length;

  const extractMutation = useMutation({
    mutationFn: (videoId: string) => api.extractFrames(videoId, { interval }),
    onSuccess: (_data, videoId) => {
      markQueued([videoId]);
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
    },
  });

  // Kicks off extraction for every not-yet-queued video sequentially —
  // extract-frames just enqueues a worker task, so this returns as fast as
  // the requests round-trip and the rows then poll their own progress.
  // Continues past a single video's failure rather than aborting the rest
  // of the batch silently (audit finding FE-13) — one bad video used to
  // stop every video after it in the list from being queued at all, with
  // no indication that had happened.
  const extractAllMutation = useMutation({
    mutationFn: async () => {
      const targets = (videosQuery.data ?? []).filter((v) => v.status === "UPLOADED" && !queuedIds.has(v.id));
      const failed: string[] = [];
      for (const v of targets) {
        try {
          await api.extractFrames(v.id, { interval });
          markQueued([v.id]);
        } catch (err) {
          failed.push(`${v.original_filename}: ${err instanceof ApiError ? err.message : "failed to queue"}`);
        }
      }
      if (failed.length > 0) throw new Error(failed.join("; "));
    },
    onSuccess: () => {
      setExtractAllError(null);
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
    },
    onError: (err) => {
      setExtractAllError((err as Error).message);
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (videoId: string) => api.deleteVideo(videoId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["videos", datasetId] }),
  });

  async function handleFile(file: File | null) {
    if (!file || !datasetId) return;
    setUploading(true);
    setUploadErrors([]);
    try {
      await api.uploadVideo(datasetId, file);
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
    } catch (err) {
      setUploadErrors([{ name: file.name, message: err instanceof ApiError ? err.message : "Upload failed" }]);
    } finally {
      setUploading(false);
    }
  }

  // Browsers hand a folder pick over as a flat FileList (no real "upload a
  // directory" request exists), so we filter it down to video files and
  // upload them one at a time through the same single-file endpoint,
  // invalidating once at the end rather than after every file. Continues
  // past a single file's failure instead of aborting the rest of the
  // folder silently (audit finding FE-04) — one bad file used to stop
  // every file after it in the picked folder from uploading at all.
  async function handleFolder(fileList: FileList | null) {
    if (!fileList || !datasetId) return;
    const files = Array.from(fileList).filter(isVideoFile);
    if (files.length === 0) return;

    setFolderProgress({ done: 0, total: files.length });
    setUploadErrors([]);
    const failures: { name: string; message: string }[] = [];
    try {
      for (const file of files) {
        try {
          await api.uploadVideo(datasetId, file);
        } catch (err) {
          failures.push({ name: file.name, message: err instanceof ApiError ? err.message : "Upload failed" });
        }
        setFolderProgress((prev) => (prev ? { ...prev, done: prev.done + 1 } : prev));
      }
    } finally {
      queryClient.invalidateQueries({ queryKey: ["videos", datasetId] });
      setFolderProgress(null);
      setUploadErrors(failures);
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
            className="border-2 border-ink bg-ink px-6 py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
          >
            {uploading ? "Uploading…" : "Upload video"}
          </button>
          <input
            ref={folderInputRef}
            type="file"
            multiple
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            className="hidden"
            onChange={(e) => {
              const { files } = e.target;
              handleFolder(files);
              e.target.value = "";
            }}
          />
          <button
            onClick={() => folderInputRef.current?.click()}
            disabled={!!folderProgress || !datasetId}
            className="border-2 border-ink px-6 py-3 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper disabled:opacity-40"
          >
            {folderProgress ? `Uploading ${folderProgress.done}/${folderProgress.total}…` : "Upload folder"}
          </button>
          <button
            onClick={() => extractAllMutation.mutate()}
            disabled={extractAllMutation.isPending || !uploadedCount}
            className="border-2 border-ink px-6 py-3 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper disabled:opacity-40"
          >
            {extractAllMutation.isPending ? "Extracting…" : `Extract all${uploadedCount ? ` (${uploadedCount})` : ""}`}
          </button>
        </div>
      </div>

      {!datasetId && <p className="text-sm text-ink/50">Select a dataset above to see or upload its videos.</p>}

      {(uploadErrors.length > 0 || extractAllError) && (
        <div className="mb-8 max-w-4xl border-2 border-accent bg-accent/5 px-4 py-3 text-xs">
          <div className="flex items-start justify-between gap-4">
            <p className="font-bold uppercase tracking-widest text-accent">
              {uploadErrors.length > 0
                ? `${uploadErrors.length} file${uploadErrors.length === 1 ? "" : "s"} failed to upload`
                : "Some videos failed to queue for extraction"}
            </p>
            <button
              onClick={() => {
                setUploadErrors([]);
                setExtractAllError(null);
              }}
              className="font-bold uppercase tracking-widest text-ink/40 hover:text-ink"
            >
              Dismiss
            </button>
          </div>
          <ul className="mt-2 space-y-1 text-ink/70">
            {uploadErrors.map((f, i) => (
              <li key={i}>
                <span className="font-semibold">{f.name}</span> — {f.message}
              </li>
            ))}
            {extractAllError && <li>{extractAllError}</li>}
          </ul>
        </div>
      )}

      <div className="max-w-4xl border-t-2 border-ink">
        {(videosQuery.data ?? []).map((video) => (
          <VideoRow
            key={video.id}
            video={video}
            queued={queuedIds.has(video.id)}
            onExtract={(id) => extractMutation.mutate(id)}
            onDelete={(id) => deleteMutation.mutate(id)}
            deleting={deleteMutation.isPending && deleteMutation.variables === video.id}
          />
        ))}
        {videosQuery.data?.length === 0 && (
          <p className="py-8 text-sm text-ink/50">No videos yet — upload one to get started.</p>
        )}
      </div>
    </div>
  );
}
