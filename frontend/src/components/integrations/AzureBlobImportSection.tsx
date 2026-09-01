import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { ImportJobProgress } from "@/components/integrations/ImportJobProgress";
import type { BlobImportJob, BlobImportLabelFormat } from "@/types";

// Lives on the Datasets page, next to the file / Roboflow import cards,
// and only rendered when the backend's storage_backend is "azure". Each
// run registers images that already sit under `prefix` in the app's
// container as a new Dataset *by reference* — no download, no second copy.
export function AzureBlobImportSection({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const [prefix, setPrefix] = useState("");
  const [labelFormat, setLabelFormat] = useState<BlobImportLabelFormat>("auto");
  const [datasetName, setDatasetName] = useState("");
  const [job, setJob] = useState<BlobImportJob | null>(null);

  // Reattach to a job kicked off before a navigation away / reload.
  const latestJobQuery = useQuery({
    queryKey: ["blob-import-latest-job", projectId],
    queryFn: () => api.getLatestBlobImportJob(projectId),
  });
  useEffect(() => {
    const latest = latestJobQuery.data;
    if (latest && (latest.status === "RUNNING" || latest.status === "QUEUED")) {
      setJob((current) => current ?? latest);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestJobQuery.data]);

  const importMutation = useMutation({
    mutationFn: () =>
      api.importAzureBlobDataset(projectId, {
        prefix: prefix.trim(),
        label_format: labelFormat,
        dataset_name: datasetName.trim() || undefined,
      }),
    onSuccess: (created) => setJob(created),
  });

  const running = job != null && (job.status === "RUNNING" || job.status === "QUEUED");

  return (
    <div className="max-w-xl flex-1 border-2 border-ink p-6">
      <p className="mb-1 text-sm font-bold uppercase tracking-tight">Import from Azure Blob</p>
      <p className="mb-4 text-xs text-ink/60">
        Point at a folder (prefix) inside this app's blob container. Images are referenced in place —
        no download, no second copy — with their YOLO or COCO labels pulled in as approved ground
        truth.
      </p>
      <div className="space-y-3">
        <input
          value={prefix}
          onChange={(e) => setPrefix(e.target.value)}
          aria-label="BLOB PREFIX"
          placeholder="BLOB PREFIX  e.g. prod-batch-1/"
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
        />

        <div className="flex gap-6">
          {(["auto", "yolo", "coco"] as const).map((f) => (
            <label key={f} className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
              <input
                type="radio"
                checked={labelFormat === f}
                onChange={() => setLabelFormat(f)}
                className="accent-accent"
              />
              {f === "auto" ? "Auto-detect" : f.toUpperCase()}
            </label>
          ))}
        </div>

        <input
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          aria-label="DATASET NAME (optional)"
          placeholder="DATASET NAME (optional)"
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
        />

        <button
          onClick={() => importMutation.mutate()}
          disabled={!prefix.trim() || running}
          className="w-full border-2 border-ink bg-ink py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange hover:text-ink disabled:opacity-40"
        >
          {running ? `Importing… ${job?.processed_items ?? 0} / ${job?.total_items ?? 0}` : "Import"}
        </button>
        {importMutation.isError && (
          <p className="text-xs text-accent-ink">{(importMutation.error as Error).message}</p>
        )}
      </div>

      {job && (
        <ImportJobProgress
          key={job.id}
          jobId={job.id}
          initialStatus={job.status}
          initialProcessed={job.processed_items}
          initialTotal={job.total_items}
          initialError={job.error}
          streamUrl={`/api/v1/integrations/azure-blob/jobs/${job.id}/stream`}
          onCancel={() => api.cancelBlobImportJob(job.id)}
          onSettled={(status) => {
            if (status === "COMPLETED") {
              queryClient.invalidateQueries({ queryKey: ["datasets", projectId] });
            }
          }}
        />
      )}
    </div>
  );
}
