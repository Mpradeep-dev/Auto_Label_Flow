import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { Dataset } from "@/types";

function DatasetCard({ dataset }: { dataset: Dataset }) {
  const statsQuery = useQuery({
    queryKey: ["dataset-stats", dataset.id],
    queryFn: () => api.getDatasetStats(dataset.id),
  });
  const stats = statsQuery.data;

  return (
    <div className="group border-b-2 border-r-2 border-ink p-8 transition-colors duration-150 hover:bg-ink hover:text-paper">
      <Link to={`/projects/${dataset.project_id}/datasets/${dataset.id}/images`} className="block">
        <p className="text-2xl font-bold uppercase tracking-tight">{dataset.name}</p>
        <p className="mt-2 tabular text-xs uppercase tracking-widest text-ink/50 group-hover:text-paper/60">
          {stats ? `${stats.total_images} images · ${stats.total_videos} videos` : "…"}
        </p>
      </Link>
      <Link
        to={`/projects/${dataset.project_id}/datasets/${dataset.id}/statistics`}
        className="mt-3 inline-block text-[10px] font-bold uppercase tracking-widest underline decoration-1 underline-offset-2"
      >
        View statistics →
      </Link>
    </div>
  );
}

function FileImportSection({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [format, setFormat] = useState<"coco" | "cvat">("coco");
  const [datasetName, setDatasetName] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const importMutation = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("Choose a .zip file first");
      return format === "coco"
        ? api.importCocoDataset(projectId, file, datasetName.trim() || undefined)
        : api.importCvatDataset(projectId, file, datasetName.trim() || undefined);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", projectId] });
      setFile(null);
      setDatasetName("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
  });

  return (
    <div className="mb-12 max-w-xl border-2 border-ink p-6">
      <p className="mb-1 text-sm font-bold uppercase tracking-tight">Import from a file</p>
      <p className="mb-4 text-xs text-ink/50">
        COCO or CVAT-XML — the two formats CVAT's own UI exports. Round-trips with CVAT: export a task
        there as "COCO 1.0" or "CVAT 1.1", import that zip here; export a version from this app in
        either format (on the Export page) to bring it back into CVAT.
      </p>
      <div className="space-y-3">
        <div className="flex gap-6">
          {(["coco", "cvat"] as const).map((f) => (
            <label key={f} className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
              <input type="radio" checked={format === f} onChange={() => setFormat(f)} className="accent-accent" />
              {f === "coco" ? "COCO" : "CVAT XML"}
            </label>
          ))}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".zip"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-xs outline-none focus:border-accent"
        />
        <input
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          placeholder="DATASET NAME (optional)"
          className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button
          onClick={() => importMutation.mutate()}
          disabled={!file || importMutation.isPending}
          className="w-full border-2 border-ink bg-ink py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {importMutation.isPending ? "Importing…" : "Import"}
        </button>
        {importMutation.isError && (
          <p className="text-xs text-accent">{(importMutation.error as Error).message}</p>
        )}
      </div>
    </div>
  );
}

export function DatasetsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [name, setName] = useState("");
  const queryClient = useQueryClient();

  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: !!projectId,
  });

  const createMutation = useMutation({
    mutationFn: () => api.createDataset(projectId!, { name: name.trim() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", projectId] });
      setName("");
    },
  });

  if (!projectId) return null;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={2}>Datasets</SectionLabel>
      <h1 className="mb-4 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Datasets
      </h1>
      <p className="mb-12 max-w-2xl text-sm text-ink/60">
        Looking to pull data in from Roboflow?{" "}
        <Link
          to={`/projects/${projectId}/settings`}
          className="font-semibold underline decoration-1 underline-offset-2 hover:text-accent"
        >
          That's on Project Settings →
        </Link>{" "}
        — a Roboflow project is a source for this whole project, not any one dataset.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) createMutation.mutate();
        }}
        className="mb-12 flex max-w-xl border-2 border-ink"
      >
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="NEW DATASET NAME"
          className="flex-1 bg-paper px-4 py-3 text-sm font-semibold uppercase tracking-wide outline-none placeholder:text-ink/30 focus:bg-muted"
        />
        <button
          type="submit"
          disabled={!name.trim() || createMutation.isPending}
          className="border-l-2 border-ink bg-ink px-6 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {createMutation.isPending ? "Creating…" : "Create"}
        </button>
      </form>

      <FileImportSection projectId={projectId} />

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {(datasetsQuery.data ?? []).map((dataset) => (
          <DatasetCard key={dataset.id} dataset={dataset} />
        ))}
        {datasetsQuery.data?.length === 0 && (
          <p className="col-span-2 py-8 text-sm text-ink/50">No datasets yet — create one above.</p>
        )}
      </div>
    </div>
  );
}
