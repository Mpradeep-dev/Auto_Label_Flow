import { useState } from "react";
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
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
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
