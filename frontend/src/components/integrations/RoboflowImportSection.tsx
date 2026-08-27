import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { RoboflowJobProgress } from "@/components/integrations/RoboflowJobProgress";
import { RoboflowProjectSelect } from "@/components/integrations/RoboflowProjectSelect";
import type { RoboflowJob } from "@/types";

// Lives on the Datasets page, alongside the local-file import — each run
// creates its own new Dataset underneath, same as any other import path.
export function RoboflowImportSection({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const [workspace, setWorkspace] = useState("");
  const [project, setProject] = useState("");
  const [version, setVersion] = useState<number | "">("");
  // Independent of `version` — pulls the project's raw uploaded images
  // (whatever's on them, labeled or not) instead of a generated Version.
  // Forced on below when the project has no version to pick from at all.
  const [importRaw, setImportRaw] = useState(false);
  // Narrows the raw pull further, to only images with no annotations yet
  // in Roboflow — only meaningful (and only shown) while `usingRaw`.
  const [unannotatedOnly, setUnannotatedOnly] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [job, setJob] = useState<RoboflowJob | null>(null);

  // Reattaches to a job this page kicked off before a navigation away or a
  // reload wiped `job` above — otherwise a still-running import just
  // disappears from the UI even though it keeps going server-side.
  const latestJobQuery = useQuery({
    queryKey: ["roboflow-latest-job", "IMPORT", projectId],
    queryFn: () => api.getLatestRoboflowJob({ kind: "IMPORT", project_id: projectId }),
  });
  useEffect(() => {
    const latest = latestJobQuery.data;
    if (latest && (latest.status === "RUNNING" || latest.status === "QUEUED")) {
      setJob((current) => current ?? latest);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestJobQuery.data]);

  const versionsQuery = useQuery({
    queryKey: ["roboflow-versions", workspace, project],
    queryFn: () => api.listRoboflowVersions(workspace, project),
    enabled: !!workspace && !!project,
  });

  const hasVersions = (versionsQuery.data ?? []).length > 0;
  const noVersions = !!workspace && !!project && !versionsQuery.isLoading && !hasVersions;
  const usingRaw = importRaw || noVersions;

  const importMutation = useMutation({
    mutationFn: () =>
      api.importRoboflowDataset(projectId, {
        workspace,
        project,
        version: usingRaw ? undefined : (version as number),
        dataset_name: datasetName.trim() || undefined,
        unannotated_only: usingRaw ? unannotatedOnly : undefined,
      }),
    onSuccess: (created) => setJob(created),
  });

  const running = job != null && (job.status === "RUNNING" || job.status === "QUEUED");

  return (
    <div className="max-w-xl flex-1 border-2 border-ink p-6">
      <p className="mb-4 text-sm font-bold uppercase tracking-tight">Import from Roboflow</p>
      <div className="space-y-3">
        <RoboflowProjectSelect
          workspace={workspace}
          project={project}
          onChange={(ws, proj) => {
            setWorkspace(ws);
            setProject(proj);
            setVersion("");
            setImportRaw(false);
            setUnannotatedOnly(false);
          }}
        />

        {workspace && project && (hasVersions || versionsQuery.isLoading) && (
          <select
            value={version}
            onChange={(e) => setVersion(e.target.value ? Number(e.target.value) : "")}
            disabled={usingRaw}
            className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent disabled:opacity-40"
          >
            <option value="">Select a version…</option>
            {(versionsQuery.data ?? []).map((v) => (
              <option key={v.version} value={v.version}>
                Version {v.version} ({v.image_count} images)
              </option>
            ))}
          </select>
        )}

        {workspace && project && !versionsQuery.isLoading && (
          <label className="flex items-start gap-2 border-2 border-ink p-3 text-xs">
            <input
              type="checkbox"
              checked={usingRaw}
              disabled={noVersions}
              onChange={(e) => setImportRaw(e.target.checked)}
              className="mt-0.5 accent-accent"
            />
            <span>
              {noVersions ? (
                <>
                  <strong className="font-bold uppercase">No generated version found.</strong> Import will pull
                  this project's raw uploaded images (and whatever's already labeled on them) directly instead.
                </>
              ) : (
                <>Import raw / unannotated images instead of a version.</>
              )}{" "}
              Pulled images land as pending review, not auto-approved.
            </span>
          </label>
        )}

        {usingRaw && (
          <label className="ml-6 flex items-start gap-2 text-xs text-ink/70">
            <input
              type="checkbox"
              checked={unannotatedOnly}
              onChange={(e) => setUnannotatedOnly(e.target.checked)}
              className="mt-0.5 accent-accent"
            />
            <span>
              Only pull images with no annotations yet in Roboflow — skip anything already labeled there.
            </span>
          </label>
        )}

        <input
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          placeholder="DATASET NAME (optional)"
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
        />

        <button
          onClick={() => importMutation.mutate()}
          disabled={!workspace || !project || (!usingRaw && version === "") || running}
          className="w-full border-2 border-ink bg-ink py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
        >
          {running ? `Importing… ${job?.processed_items ?? 0} / ${job?.total_items ?? 0}` : "Import"}
        </button>
        {importMutation.isError && (
          <p className="text-xs text-accent">{(importMutation.error as Error).message}</p>
        )}
      </div>

      {job && (
        <RoboflowJobProgress
          key={job.id}
          job={job}
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
