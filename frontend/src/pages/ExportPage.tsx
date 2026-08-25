import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { RoboflowJobProgress } from "@/components/integrations/RoboflowJobProgress";
import { RoboflowProjectSelect } from "@/components/integrations/RoboflowProjectSelect";
import type { DatasetVersion, RoboflowJob } from "@/types";

const STATUS_STYLE: Record<string, string> = {
  DRAFT: "bg-muted text-ink/70",
  EXPORTING: "bg-muted text-ink/70",
  EXPORTED: "bg-ink text-paper",
  FAILED: "bg-accent text-paper",
};

function RoboflowExportControls({ versionId }: { versionId: string }) {
  const [open, setOpen] = useState(false);
  const [workspace, setWorkspace] = useState("");
  const [project, setProject] = useState("");
  const [job, setJob] = useState<RoboflowJob | null>(null);

  // Reattaches to a job this row kicked off before a navigation away or a
  // reload wiped `job` above — otherwise a still-running export just
  // disappears from the UI (collapsing back to the plain "Export to
  // Roboflow" button) even though it keeps going server-side.
  const latestJobQuery = useQuery({
    queryKey: ["roboflow-latest-job", "EXPORT", versionId],
    queryFn: () => api.getLatestRoboflowJob({ kind: "EXPORT", dataset_version_id: versionId }),
  });
  useEffect(() => {
    const latest = latestJobQuery.data;
    if (latest && (latest.status === "RUNNING" || latest.status === "QUEUED")) {
      setJob((current) => current ?? latest);
      setOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestJobQuery.data]);

  const exportMutation = useMutation({
    mutationFn: () => api.exportVersionToRoboflow(versionId, { workspace, project }),
    onSuccess: (created) => setJob(created),
  });

  if (!open && !job) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
      >
        Export to Roboflow
      </button>
    );
  }

  return (
    <div className="mt-3 max-w-sm border-2 border-ink p-4">
      <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/50">Export to Roboflow</p>
      {!job && (
        <div className="space-y-2">
          <RoboflowProjectSelect
            workspace={workspace}
            project={project}
            onChange={(ws, proj) => {
              setWorkspace(ws);
              setProject(proj);
            }}
          />
          <button
            onClick={() => exportMutation.mutate()}
            disabled={!workspace || !project || exportMutation.isPending}
            className="w-full border-2 border-ink bg-ink py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
          >
            {exportMutation.isPending ? "Starting…" : "Push"}
          </button>
          {exportMutation.isError && (
            <p className="text-xs text-accent">{(exportMutation.error as Error).message}</p>
          )}
        </div>
      )}
      {job && <RoboflowJobProgress key={job.id} job={job} />}
    </div>
  );
}

function FormatExportButton({
  label,
  downloadUrl,
  downloadLabel,
  onExport,
  pending,
}: {
  label: string;
  downloadUrl: string | null;
  downloadLabel: string;
  onExport: () => void;
  pending: boolean;
}) {
  if (downloadUrl) {
    return (
      <a
        href={downloadUrl}
        className="inline-block border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
      >
        {downloadLabel}
      </a>
    );
  }
  return (
    <button
      onClick={onExport}
      disabled={pending}
      className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper disabled:opacity-40"
    >
      {pending ? "Exporting…" : label}
    </button>
  );
}

function VersionRow({
  version,
  onExport,
  onExportCoco,
  onExportCvat,
  cocoPending,
  cvatPending,
}: {
  version: DatasetVersion;
  onExport: (id: string) => void;
  onExportCoco: (id: string) => void;
  onExportCvat: (id: string) => void;
  cocoPending: boolean;
  cvatPending: boolean;
}) {
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
      <div className="mt-3 flex flex-wrap gap-3">
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
        {/* COCO and CVAT-XML are the two formats CVAT's own UI imports
            directly — exporting either here is the bridge into CVAT
            without a live API integration (Task > Upload annotations,
            format "COCO 1.0" / "CVAT 1.1"). Available regardless of the
            YOLO export's own DRAFT/EXPORTED status — they're independent
            artifacts, not another state in that state machine. */}
        <FormatExportButton
          label="Export COCO"
          downloadLabel="Download COCO"
          downloadUrl={version.coco_download_url}
          onExport={() => onExportCoco(version.id)}
          pending={cocoPending}
        />
        <FormatExportButton
          label="Export CVAT XML"
          downloadLabel="Download CVAT XML"
          downloadUrl={version.cvat_download_url}
          onExport={() => onExportCvat(version.id)}
          pending={cvatPending}
        />
        <RoboflowExportControls versionId={version.id} />
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
  const exportCocoMutation = useMutation({
    mutationFn: (versionId: string) => api.exportVersionCoco(versionId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["versions", datasetId] }),
  });
  const exportCvatMutation = useMutation({
    mutationFn: (versionId: string) => api.exportVersionCvat(versionId),
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
              <VersionRow
                key={v.id}
                version={v}
                onExport={(id) => exportMutation.mutate(id)}
                onExportCoco={(id) => exportCocoMutation.mutate(id)}
                onExportCvat={(id) => exportCvatMutation.mutate(id)}
                cocoPending={exportCocoMutation.isPending && exportCocoMutation.variables === v.id}
                cvatPending={exportCvatMutation.isPending && exportCvatMutation.variables === v.id}
              />
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
