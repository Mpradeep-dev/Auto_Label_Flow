import { useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { DatasetVersion } from "@/types";

const STATUS_STYLE: Record<string, string> = {
  DRAFT: "bg-muted text-ink/70",
  EXPORTING: "bg-muted text-ink/70",
  EXPORTED: "bg-ink text-paper",
  FAILED: "bg-accent text-paper",
};

function VersionRow({ version, onExport }: { version: DatasetVersion; onExport: (id: string) => void }) {
  return (
    <div className="border-b-2 border-ink p-6">
      <div className="flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">Version {version.version_number}</p>
        <span className={`px-2 py-1 text-[10px] font-bold uppercase tracking-widest ${STATUS_STYLE[version.status]}`}>
          {version.status}
        </span>
      </div>
      <p className="tabular mt-1 text-xs uppercase tracking-widest text-ink/50">
        {version.total_images} images · {version.total_annotations} annotations · split{" "}
        {Math.round(version.train_ratio * 100)}/{Math.round(version.val_ratio * 100)}/
        {Math.round(version.test_ratio * 100)}
        {version.used_frame_level_fallback && " · frame-level fallback (too few source videos to group)"}
      </p>
      {version.error && <p className="mt-2 text-xs text-accent">{version.error}</p>}
      <div className="mt-3">
        {version.status === "DRAFT" && (
          <button
            onClick={() => onExport(version.id)}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
          >
            Export YOLO
          </button>
        )}
        {version.status === "EXPORTED" && version.download_url && (
          <a
            href={version.download_url}
            className="inline-block border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
          >
            Download .zip
          </a>
        )}
      </div>
    </div>
  );
}

export function ExportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [datasetId, setDatasetId] = useState("");
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const queryClient = useQueryClient();

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });
  const versionsQuery = useQuery({
    queryKey: ["versions", datasetId],
    queryFn: () => api.listDatasetVersions(datasetId),
    enabled: !!datasetId,
  });

  const createMutation = useMutation({
    mutationFn: () =>
      api.createDatasetVersion(datasetId, { train_ratio: trainRatio, val_ratio: valRatio, test_ratio: testRatio }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["versions", datasetId] }),
  });
  const exportMutation = useMutation({
    mutationFn: (versionId: string) => api.exportDatasetVersion(versionId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["versions", datasetId] }),
  });

  if (!projectId) return null;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Export</SectionLabel>
      <h1 className="mb-8 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Export
      </h1>

      <div className="mb-8 max-w-2xl">
        <label className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-ink/50">Dataset</label>
        <select
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
        >
          <option value="">Select a dataset…</option>
          {(datasetsQuery.data ?? []).map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {datasetId && (
        <>
          <div className="mb-8 flex max-w-2xl flex-wrap items-end gap-4 border-2 border-ink p-4">
            {(
              [
                ["Train", trainRatio, setTrainRatio],
                ["Val", valRatio, setValRatio],
                ["Test", testRatio, setTestRatio],
              ] as const
            ).map(([label, value, setter]) => (
              <label key={label} className="text-[10px] font-bold uppercase tracking-widest text-ink/50">
                {label}
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={value}
                  onChange={(e) => setter(parseFloat(e.target.value) || 0)}
                  className="tabular ml-2 w-16 border border-ink/30 px-1.5 py-0.5 text-center text-xs"
                />
              </label>
            ))}
            <button
              onClick={() => createMutation.mutate()}
              disabled={createMutation.isPending}
              className="ml-auto border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
            >
              {createMutation.isPending ? "Creating…" : "Create version"}
            </button>
          </div>
          {createMutation.isError && (
            <p className="mb-6 max-w-2xl text-xs text-accent">{(createMutation.error as Error).message}</p>
          )}

          <div className="max-w-2xl border-t-2 border-ink">
            {(versionsQuery.data ?? []).map((v) => (
              <VersionRow key={v.id} version={v} onExport={(id) => exportMutation.mutate(id)} />
            ))}
            {versionsQuery.data?.length === 0 && (
              <p className="py-8 text-sm text-ink/50">
                No versions yet — approve some images, then create one above.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
