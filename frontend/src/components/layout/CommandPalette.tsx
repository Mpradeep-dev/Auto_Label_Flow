import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";

interface Result {
  id: string;
  group: "Projects" | "Datasets" | "Models";
  label: string;
  detail?: string;
  to: string;
}

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { projectId } = useParams<{ projectId?: string }>();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // ⌘K / Ctrl+K opens the palette from anywhere in the app; Escape closes
  // it. Lives here rather than in Sidebar so it works regardless of which
  // page (or nothing) currently has focus.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenChange(true);
      } else if (e.key === "Escape" && open) {
        onOpenChange(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onOpenChange]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      // Let the overlay mount before focusing.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects, enabled: open });
  const datasetsQuery = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.listDatasets(projectId!),
    enabled: open && !!projectId,
  });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.listModels, enabled: open && !!projectId });

  const results = useMemo<Result[]>(() => {
    const q = query.trim().toLowerCase();
    const matches = (s: string) => !q || s.toLowerCase().includes(q);

    const projects: Result[] = (projectsQuery.data ?? [])
      .filter((p) => matches(p.name))
      .map((p) => ({ id: `project-${p.id}`, group: "Projects", label: p.name, to: `/projects/${p.id}` }));

    const datasets: Result[] = (datasetsQuery.data ?? [])
      .filter((d) => matches(d.name))
      .map((d) => ({
        id: `dataset-${d.id}`,
        group: "Datasets",
        label: d.name,
        detail: "this project",
        to: `/projects/${d.project_id}/datasets/${d.id}/images`,
      }));

    const models: Result[] = projectId
      ? (modelsQuery.data ?? [])
          .filter((m) => matches(m.name))
          .map((m) => ({
            id: `model-${m.id}`,
            group: "Models",
            label: m.name,
            detail: m.kind,
            to: `/projects/${projectId}/models`,
          }))
      : [];

    return [...projects, ...datasets, ...models];
  }, [query, projectsQuery.data, datasetsQuery.data, modelsQuery.data, projectId]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  function select(result: Result) {
    onOpenChange(false);
    navigate(result.to);
  }

  if (!open) return null;

  let groupCursor = "";

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-ink/50 pt-[15vh]"
      onClick={() => onOpenChange(false)}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg border-4 border-ink bg-paper shadow-[8px_8px_0_0_rgba(0,0,0,1)]"
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActiveIndex((i) => Math.min(results.length - 1, i + 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActiveIndex((i) => Math.max(0, i - 1));
            } else if (e.key === "Enter" && results[activeIndex]) {
              select(results[activeIndex]);
            }
          }}
          placeholder="SEARCH PROJECTS, DATASETS, MODELS…"
          className="w-full border-b-2 border-ink bg-paper px-4 py-4 text-sm font-semibold uppercase tracking-wide outline-none placeholder:text-ink/30"
        />
        <div className="max-h-96 overflow-y-auto">
          {results.length === 0 && (
            <p className="px-4 py-6 text-center text-xs font-bold uppercase tracking-widest text-ink/40">
              {query ? "No matches" : "Type to search"}
            </p>
          )}
          {results.map((result, i) => {
            const showGroup = result.group !== groupCursor;
            groupCursor = result.group;
            return (
              <div key={result.id}>
                {showGroup && (
                  <p className="border-b border-ink/10 bg-muted px-4 py-1 text-[9px] font-bold uppercase tracking-widest text-ink/40">
                    {result.group}
                  </p>
                )}
                <button
                  onClick={() => select(result)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`flex w-full items-center justify-between border-b border-ink/10 px-4 py-2.5 text-left text-sm font-semibold ${
                    i === activeIndex ? "bg-ink text-paper" : "hover:bg-orange hover:text-paper"
                  }`}
                >
                  <span className="truncate uppercase tracking-wide">{result.label}</span>
                  {result.detail && (
                    <span
                      className={`shrink-0 pl-3 text-[10px] font-bold uppercase tracking-widest ${
                        i === activeIndex ? "text-paper/50" : "text-ink/40"
                      }`}
                    >
                      {result.detail}
                    </span>
                  )}
                </button>
              </div>
            );
          })}
        </div>
        <div className="flex items-center justify-between border-t-2 border-ink px-4 py-2 text-[9px] font-bold uppercase tracking-widest text-ink/40">
          <span>↑↓ navigate · ↵ open</span>
          <span>Esc to close</span>
        </div>
      </div>
    </div>
  );
}
