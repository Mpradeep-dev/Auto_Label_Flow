import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";

export function ProjectsPage() {
  const [name, setName] = useState("");
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });

  const createMutation = useMutation({
    mutationFn: () => api.createProject({ name: name.trim() }),
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      setName("");
      navigate(`/projects/${project.id}`);
    },
  });

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Projects</SectionLabel>
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Projects
      </h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) createMutation.mutate();
        }}
        className="mb-12 flex max-w-xl gap-0 border-2 border-ink"
      >
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="NEW PROJECT NAME"
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

      {projectsQuery.isLoading && <p className="text-sm text-ink/50">Loading…</p>}

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {(projectsQuery.data ?? []).map((project) => (
          <Link
            key={project.id}
            to={`/projects/${project.id}`}
            className="group border-b-2 border-r-2 border-ink p-8 transition-colors duration-150 hover:bg-ink hover:text-paper"
          >
            <p className="text-2xl font-bold uppercase tracking-tight">{project.name}</p>
            <p className="mt-2 text-xs uppercase tracking-widest text-ink/50 group-hover:text-paper/60">
              {project.class_config.length > 0
                ? project.class_config.map((c) => c.name).join(" · ")
                : "no model registered yet"}
            </p>
          </Link>
        ))}
        {projectsQuery.data?.length === 0 && (
          <p className="col-span-2 py-8 text-sm text-ink/50">No projects yet — create one above.</p>
        )}
      </div>
    </div>
  );
}
