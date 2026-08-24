import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { ModelKind } from "@/types";

export function ModelsPage() {
  const [name, setName] = useState("");
  const [weightsPath, setWeightsPath] = useState("");
  const [kind, setKind] = useState<ModelKind>("DETECTOR");
  const queryClient = useQueryClient();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.listModels });

  const registerMutation = useMutation({
    mutationFn: () => api.registerModel({ name: name.trim(), weights_path: weightsPath.trim(), kind }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setName("");
      setWeightsPath("");
    },
  });

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Models</SectionLabel>
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Models
      </h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim() && weightsPath.trim()) registerMutation.mutate();
        }}
        className="mb-12 max-w-2xl border-2 border-ink"
      >
        <div className="grid grid-cols-[1fr_2fr_auto] divide-x-2 divide-ink border-b-2 border-ink">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="NAME (E.G. detect_v1)"
            className="bg-paper px-3 py-3 text-xs font-semibold uppercase tracking-wide outline-none placeholder:text-ink/30 focus:bg-muted"
          />
          <input
            value={weightsPath}
            onChange={(e) => setWeightsPath(e.target.value)}
            placeholder="WEIGHTS PATH (e.g. .../artifacts/models/pt/detect_v1.pt)"
            className="bg-paper px-3 py-3 text-xs font-semibold tracking-wide outline-none placeholder:text-ink/30 focus:bg-muted"
          />
          <select
            value={kind}
            onChange={(e) => setKind(e.target.value as ModelKind)}
            className="bg-paper px-3 py-3 text-xs font-bold uppercase tracking-widest outline-none"
          >
            <option value="DETECTOR">Detector</option>
            <option value="POSE">Pose (auxiliary)</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={!name.trim() || !weightsPath.trim() || registerMutation.isPending}
          className="w-full bg-ink py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {registerMutation.isPending ? "Registering…" : "Register model"}
        </button>
        {registerMutation.isError && (
          <p className="border-t-2 border-ink bg-muted px-3 py-2 text-xs text-accent">
            {(registerMutation.error as Error).message}
          </p>
        )}
      </form>

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {(modelsQuery.data ?? []).map((model) => (
          <div key={model.id} className="border-b-2 border-r-2 border-ink p-8">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xl font-bold uppercase tracking-tight">{model.name}</p>
              <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-muted">
                {model.kind}
              </span>
            </div>
            <p className="tabular text-xs uppercase tracking-widest text-ink/50">
              {model.class_config.map((c) => c.name).join(" · ") || "no classes"}
            </p>
            {model.base_model_id && (
              <p className="mt-1 text-[10px] uppercase tracking-widest text-ink/30">Fine-tuned from a base model</p>
            )}
            {Object.keys(model.metrics).length > 0 && (
              <div className="mt-4 grid grid-cols-4 gap-2 border-t border-ink/10 pt-3">
                {Object.entries(model.metrics)
                  .filter(([, v]) => typeof v === "number")
                  .slice(0, 8)
                  .map(([key, value]) => (
                    <div key={key}>
                      <p className="text-[8px] font-bold uppercase tracking-widest text-ink/40">
                        {key.replace(/_/g, " ")}
                      </p>
                      <p className="tabular text-sm font-bold">{(value as number).toFixed(3)}</p>
                    </div>
                  ))}
              </div>
            )}
          </div>
        ))}
        {modelsQuery.data?.length === 0 && (
          <p className="col-span-2 py-8 text-sm text-ink/50">
            No models registered yet — point at a weights file above (e.g. the copied{" "}
            <code className="tabular">artifacts/models/pt/detect_v1.pt</code>).
          </p>
        )}
      </div>
    </div>
  );
}
