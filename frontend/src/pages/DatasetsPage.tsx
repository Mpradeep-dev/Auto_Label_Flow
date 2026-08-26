import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { FieldError } from "@/components/layout/FieldError";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { EmptyState } from "@/components/layout/EmptyState";
import { Skeleton } from "@/components/layout/Skeleton";
import { RoboflowImportSection } from "@/components/integrations/RoboflowImportSection";
import type { Dataset } from "@/types";

function DeleteDatasetPanel({ dataset, onCancel }: { dataset: Dataset; onCancel: () => void }) {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteDataset(dataset.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", dataset.project_id] });
    },
  });

  return (
    <div className="p-8">
      <p className="text-xs font-bold uppercase tracking-widest text-accent">Delete this dataset?</p>
      <p className="mt-2 text-xs text-ink/60">
        Permanently deletes <span className="font-bold text-ink">{dataset.name}</span> and every image,
        video, and annotation inside it.
      </p>
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => deleteMutation.mutate()}
          disabled={deleteMutation.isPending}
          className="border-2 border-accent bg-accent px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-ink hover:border-ink disabled:opacity-40"
        >
          {deleteMutation.isPending ? "Deleting…" : "Delete permanently"}
        </button>
        <button
          onClick={onCancel}
          className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-muted"
        >
          Cancel
        </button>
      </div>
      {deleteMutation.isError && (
        <p className="mt-2 text-xs text-accent">{(deleteMutation.error as Error).message}</p>
      )}
    </div>
  );
}

function DatasetCard({ dataset }: { dataset: Dataset }) {
  const [confirming, setConfirming] = useState(false);
  const statsQuery = useQuery({
    queryKey: ["dataset-stats", dataset.id],
    queryFn: () => api.getDatasetStats(dataset.id),
  });
  const stats = statsQuery.data;

  if (confirming) {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <DeleteDatasetPanel dataset={dataset} onCancel={() => setConfirming(false)} />
      </div>
    );
  }

  return (
    <div className="group relative border-b-2 border-r-2 border-ink p-8 transition-colors duration-150 hover:bg-ink hover:text-paper">
      <Link to={`/projects/${dataset.project_id}/datasets/${dataset.id}/images`} className="block pr-16">
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
      <button
        onClick={() => setConfirming(true)}
        className="absolute right-4 top-4 border border-ink/30 px-2 py-1 text-[10px] font-bold uppercase tracking-widest opacity-0 hover:border-accent hover:text-accent group-hover:opacity-100 group-hover:border-paper/40 group-hover:text-paper group-hover:hover:border-accent group-hover:hover:text-accent"
      >
        Delete
      </button>
    </div>
  );
}

function FileImportSection({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [format, setFormat] = useState<"coco" | "cvat">("coco");
  const [datasetName, setDatasetName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

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

  function pickFile(picked: File | null) {
    if (picked && !picked.name.toLowerCase().endsWith(".zip")) return;
    setFile(picked);
  }

  return (
    <div className="max-w-xl flex-1 border-2 border-ink p-6">
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
          onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            pickFile(e.dataTransfer.files?.[0] ?? null);
          }}
          className={`flex w-full flex-col items-center justify-center gap-1 border-2 border-dashed px-4 py-8 text-center transition-colors duration-150 ${
            dragActive ? "border-accent bg-muted" : "border-ink/40 hover:border-ink hover:bg-muted"
          }`}
        >
          {file ? (
            <>
              <span className="text-sm font-bold">{file.name}</span>
              <span className="text-[10px] uppercase tracking-widest text-ink/50">
                Click or drop to replace
              </span>
            </>
          ) : (
            <>
              <span className="text-sm font-bold uppercase tracking-wide">Drop a .zip file here</span>
              <span className="text-[10px] uppercase tracking-widest text-ink/50">or click to browse</span>
            </>
          )}
        </button>

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
  const nameInputRef = useRef<HTMLInputElement>(null);
  const roboflowRef = useRef<HTMLDivElement>(null);

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
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) createMutation.mutate();
        }}
        className="mb-12 flex max-w-xl border-2 border-ink"
      >
        <input
          ref={nameInputRef}
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
      {createMutation.isError && <FieldError error={createMutation.error} />}

      <div className="mb-12 flex flex-wrap gap-8">
        <FileImportSection projectId={projectId} />
        <div ref={roboflowRef}>
          <RoboflowImportSection projectId={projectId} />
        </div>
      </div>

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {datasetsQuery.isLoading &&
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="border-b-2 border-r-2 border-ink p-8">
              <Skeleton className="h-7 w-2/3" />
              <Skeleton className="mt-3 h-3 w-1/3" />
            </div>
          ))}
        {(datasetsQuery.data ?? []).map((dataset) => (
          <DatasetCard key={dataset.id} dataset={dataset} />
        ))}
        {datasetsQuery.data?.length === 0 && (
          <EmptyState
            title="No datasets yet"
            description="Create your first dataset from scratch, or import one from Roboflow — either way, you'll land here ready to upload images."
          >
            <button
              onClick={() => nameInputRef.current?.focus()}
              className="border-2 border-ink bg-ink px-5 py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
            >
              Create a dataset
            </button>
            <button
              onClick={() => roboflowRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })}
              className="border-2 border-ink px-5 py-2.5 text-xs font-bold uppercase tracking-widest hover:bg-muted"
            >
              Import from Roboflow
            </button>
          </EmptyState>
        )}
      </div>
    </div>
  );
}
