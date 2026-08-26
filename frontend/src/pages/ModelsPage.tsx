import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { EmptyState } from "@/components/layout/EmptyState";
import { Skeleton } from "@/components/layout/Skeleton";
import type { MLModel, ModelKind } from "@/types";

type Framework = "ultralytics" | "yolo-world";

const DEFAULT_FRAMEWORK_BY_KIND: Record<ModelKind, Framework> = {
  DETECTOR: "ultralytics",
  POSE: "ultralytics",
};

function stripExtension(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, "");
}

function RegisterModelForm() {
  const [source, setSource] = useState<"upload" | "url">("upload");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [kind, setKind] = useState<ModelKind>("DETECTOR");
  const [framework, setFramework] = useState<Framework>("ultralytics");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const registerMutation = useMutation({
    mutationFn: () =>
      source === "url"
        ? api.downloadModel({ name: name.trim(), url: url.trim(), kind, framework })
        : api.uploadModel(file as File, { name: name.trim(), kind, framework }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setName("");
      setUrl("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
  });

  function pickFile(picked: File | null) {
    if (picked && !picked.name.toLowerCase().endsWith(".pt")) return;
    setFile(picked);
    if (picked && !name.trim()) setName(stripExtension(picked.name));
  }

  const hasValue = source === "url" ? url.trim() : !!file;
  const canSubmit = name.trim() && hasValue && !registerMutation.isPending;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) registerMutation.mutate();
      }}
      className="mb-12 max-w-2xl border-2 border-ink"
    >
      <div className="flex divide-x-2 divide-ink border-b-2 border-ink text-[10px] font-bold uppercase tracking-widest">
        {(["upload", "url"] as const).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSource(s)}
            className={`flex-1 py-2 ${source === s ? "bg-ink text-paper" : "bg-paper hover:bg-muted"}`}
          >
            {s === "upload" ? "Upload from this computer" : "Download from link"}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-[1fr_auto_auto] divide-x-2 divide-ink border-b-2 border-ink">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="NAME (E.G. detect_v1)"
          className="bg-paper px-3 py-3 text-xs font-semibold uppercase tracking-wide outline-none placeholder:text-ink/30 focus:bg-muted"
        />
        <select
          value={kind}
          onChange={(e) => {
            const nextKind = e.target.value as ModelKind;
            setKind(nextKind);
            setFramework(DEFAULT_FRAMEWORK_BY_KIND[nextKind]);
          }}
          className="bg-paper px-3 py-3 text-xs font-bold uppercase tracking-widest outline-none"
        >
          <option value="DETECTOR">Detector</option>
          <option value="POSE">Pose (auxiliary)</option>
        </select>
        <select
          value={framework}
          onChange={(e) => setFramework(e.target.value as Framework)}
          disabled={kind !== "DETECTOR"}
          title="YOLO-World is open-vocabulary: it detects whatever classes the current project is configured with, instead of a fixed set baked into the weights."
          className="bg-paper px-3 py-3 text-xs font-bold uppercase tracking-widest outline-none disabled:opacity-40"
        >
          <option value="ultralytics">Ultralytics YOLO</option>
          <option value="yolo-world">YOLO-World (open-vocab)</option>
        </select>
      </div>

      {source === "upload" ? (
        <div className="border-b-2 border-ink p-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pt"
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
                <span className="text-[10px] uppercase tracking-widest text-ink/50">Click or drop to replace</span>
              </>
            ) : (
              <>
                <span className="text-sm font-bold uppercase tracking-wide">Drop a .pt weights file here</span>
                <span className="text-[10px] uppercase tracking-widest text-ink/50">
                  or click to browse your computer
                </span>
              </>
            )}
          </button>
        </div>
      ) : (
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="DIRECT LINK TO .PT FILE (e.g. https://.../detect_v1.pt)"
          className="w-full border-b-2 border-ink bg-paper px-3 py-3 text-xs font-semibold tracking-wide outline-none placeholder:text-ink/30 focus:bg-muted"
        />
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full bg-ink py-3 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
      >
        {registerMutation.isPending
          ? source === "url"
            ? "Downloading…"
            : "Uploading…"
          : "Register model"}
      </button>
      {registerMutation.isError && (
        <p className="border-t-2 border-ink bg-muted px-3 py-2 text-xs text-accent">
          {(registerMutation.error as Error).message}
        </p>
      )}
    </form>
  );
}

function RenameModelPanel({ model, onCancel }: { model: MLModel; onCancel: () => void }) {
  const [name, setName] = useState(model.name);
  const queryClient = useQueryClient();

  const renameMutation = useMutation({
    mutationFn: () => api.renameModel(model.id, name.trim()),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      onCancel();
    },
  });

  return (
    <div className="p-8">
      <p className="text-xs font-bold uppercase tracking-widest text-ink/50">Rename model</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        autoFocus
        className="mt-3 w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase tracking-wide outline-none focus:border-accent"
      />
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => renameMutation.mutate()}
          disabled={!name.trim() || renameMutation.isPending}
          className="border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent disabled:opacity-40"
        >
          {renameMutation.isPending ? "Saving…" : "Save"}
        </button>
        <button
          onClick={onCancel}
          className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-muted"
        >
          Cancel
        </button>
      </div>
      {renameMutation.isError && (
        <p className="mt-2 text-xs text-accent">
          {renameMutation.error instanceof ApiError ? renameMutation.error.message : "Save failed"}
        </p>
      )}
    </div>
  );
}

function DeleteModelPanel({ model, onCancel }: { model: MLModel; onCancel: () => void }) {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteModel(model.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  return (
    <div className="p-8">
      <p className="text-xs font-bold uppercase tracking-widest text-accent">Delete this model?</p>
      <p className="mt-2 text-xs text-ink/60">
        Removes <span className="font-bold text-ink">{model.name}</span> from the registry and deletes its
        weights file. Any project using it as its primary/pose model falls back to none.
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
        <p className="mt-2 text-xs text-accent">
          {deleteMutation.error instanceof ApiError ? deleteMutation.error.message : "Delete failed"}
        </p>
      )}
    </div>
  );
}

type ModelCardMode = "view" | "menu" | "rename" | "delete";

function ModelCard({ model }: { model: MLModel }) {
  const [mode, setMode] = useState<ModelCardMode>("view");

  if (mode === "rename") {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <RenameModelPanel model={model} onCancel={() => setMode("view")} />
      </div>
    );
  }

  if (mode === "delete") {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <DeleteModelPanel model={model} onCancel={() => setMode("view")} />
      </div>
    );
  }

  return (
    <div className="group relative border-b-2 border-r-2 border-ink p-8">
      <div className="mb-2 flex items-center justify-between pr-16">
        <p className="text-xl font-bold uppercase tracking-tight">{model.name}</p>
        <div className="flex gap-1">
          {model.is_promptable && (
            <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-accent text-paper">
              Open-vocab
            </span>
          )}
          <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-muted">{model.kind}</span>
        </div>
      </div>
      <p className="tabular text-xs uppercase tracking-widest text-ink/50">
        {model.is_promptable
          ? "Detects whatever classes the project it runs against is configured with"
          : model.class_config.map((c) => c.name).join(" · ") || "no classes"}
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
      {mode === "menu" ? (
        <div className="absolute right-4 top-4 flex gap-1 border border-ink/30 bg-paper p-1">
          <button
            onClick={() => setMode("rename")}
            className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
          >
            Rename
          </button>
          <button
            onClick={() => setMode("delete")}
            className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-accent hover:bg-accent hover:text-paper"
          >
            Delete
          </button>
          <button
            onClick={() => setMode("view")}
            className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-ink/50 hover:text-ink"
          >
            Close
          </button>
        </div>
      ) : (
        <button
          onClick={() => setMode("menu")}
          className="absolute right-4 top-4 border border-ink/30 px-2 py-1 text-[10px] font-bold uppercase tracking-widest opacity-0 hover:border-accent hover:text-accent group-hover:opacity-100"
        >
          Edit
        </button>
      )}
    </div>
  );
}

export function ModelsPage() {
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const registerFormRef = useRef<HTMLDivElement>(null);

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Models</SectionLabel>
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Models
      </h1>

      <div ref={registerFormRef}>
        <RegisterModelForm />
      </div>

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {modelsQuery.isLoading &&
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="border-b-2 border-r-2 border-ink p-8">
              <Skeleton className="h-7 w-2/3" />
              <Skeleton className="mt-3 h-3 w-1/2" />
            </div>
          ))}
        {(modelsQuery.data ?? []).map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
        {modelsQuery.data?.length === 0 && (
          <EmptyState
            title="Register a model to enable auto-annotation"
            description="Upload a .pt weights file from your computer, or paste a direct download link. A DETECTOR model unlocks the Auto-annotate stage on the pipeline."
          >
            <button
              onClick={() => registerFormRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })}
              className="border-2 border-ink bg-ink px-5 py-2.5 text-xs font-bold uppercase tracking-widest text-paper hover:bg-accent"
            >
              Register a model
            </button>
            <a
              href="https://docs.ultralytics.com/models/"
              target="_blank"
              rel="noreferrer"
              className="border-2 border-ink px-5 py-2.5 text-xs font-bold uppercase tracking-widest hover:bg-muted"
            >
              YOLO model reference →
            </a>
          </EmptyState>
        )}
      </div>
    </div>
  );
}
