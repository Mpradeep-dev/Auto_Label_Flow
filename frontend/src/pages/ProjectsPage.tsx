import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { FieldError } from "@/components/layout/FieldError";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { Project } from "@/types";

function RenameProjectPanel({ project, onCancel }: { project: Project; onCancel: () => void }) {
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description ?? "");
  const queryClient = useQueryClient();

  const renameMutation = useMutation({
    mutationFn: () => api.updateProject(project.id, { name: name.trim(), description: description.trim() || null }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      onCancel();
    },
  });

  return (
    <div className="p-8">
      <p className="text-xs font-bold uppercase tracking-widest text-ink/50">Rename project</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        autoFocus
        className="mt-3 w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase tracking-wide outline-none focus:border-accent"
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Description (optional)"
        rows={2}
        className="mt-2 w-full border border-ink/30 bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
      />
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => renameMutation.mutate()}
          disabled={!name.trim() || renameMutation.isPending}
          className="border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
        >
          {renameMutation.isPending ? "Saving…" : "Save"}
        </button>
        <button
          onClick={onCancel}
          className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
        >
          Cancel
        </button>
      </div>
      {renameMutation.isError && <FieldError error={renameMutation.error} />}
    </div>
  );
}

function DeleteProjectPanel({ project, onCancel }: { project: Project; onCancel: () => void }) {
  const [confirmText, setConfirmText] = useState("");
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteProject(project.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const matches = confirmText === project.name;

  return (
    <div className="p-8">
      <p className="text-xs font-bold uppercase tracking-widest text-accent">Delete this project?</p>
      <p className="mt-2 text-xs text-ink/60">
        Permanently deletes every dataset, image, annotation, and version inside it. Type{" "}
        <span className="font-bold text-ink">{project.name}</span> to confirm.
      </p>
      <input
        value={confirmText}
        onChange={(e) => setConfirmText(e.target.value)}
        placeholder={project.name}
        autoFocus
        className="mt-3 w-full border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
      />
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => deleteMutation.mutate()}
          disabled={!matches || deleteMutation.isPending}
          className="border-2 border-accent bg-accent px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:border-orange hover:bg-orange disabled:opacity-40"
        >
          {deleteMutation.isPending ? "Deleting…" : "Delete permanently"}
        </button>
        <button
          onClick={onCancel}
          className="border-2 border-ink/30 px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
        >
          Cancel
        </button>
      </div>
      {deleteMutation.isError && <FieldError error={deleteMutation.error} />}
    </div>
  );
}

type ProjectCardMode = "view" | "menu" | "rename" | "delete";

function ProjectEditMenu({
  onRename,
  onDelete,
  onClose,
}: {
  onRename: () => void;
  onDelete: () => void;
  onClose: () => void;
}) {
  return (
    <div className="p-8">
      <p className="mb-4 text-xs font-bold uppercase tracking-widest text-ink/50">Edit project</p>
      <div className="flex flex-col gap-2">
        <button
          onClick={onRename}
          className="border-2 border-ink px-4 py-2 text-left text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
        >
          Rename / edit description
        </button>
        <button
          onClick={onDelete}
          className="border-2 border-ink/30 px-4 py-2 text-left text-xs font-bold uppercase tracking-widest text-accent hover:border-orange"
        >
          Delete project
        </button>
      </div>
      <button
        onClick={onClose}
        className="mt-3 text-[10px] font-bold uppercase tracking-widest text-ink/50 hover:text-ink"
      >
        Close
      </button>
    </div>
  );
}

function ProjectCard({ project }: { project: Project }) {
  const [mode, setMode] = useState<ProjectCardMode>("view");

  if (mode === "menu") {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <ProjectEditMenu
          onRename={() => setMode("rename")}
          onDelete={() => setMode("delete")}
          onClose={() => setMode("view")}
        />
      </div>
    );
  }

  if (mode === "rename") {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <RenameProjectPanel project={project} onCancel={() => setMode("view")} />
      </div>
    );
  }

  if (mode === "delete") {
    return (
      <div className="border-b-2 border-r-2 border-ink">
        <DeleteProjectPanel project={project} onCancel={() => setMode("view")} />
      </div>
    );
  }

  return (
    <div className="group relative border-b-2 border-r-2 border-ink transition-colors duration-150 hover:bg-orange hover:text-paper">
      <Link to={`/projects/${project.id}`} className="block p-8">
        <p className="text-2xl font-bold uppercase tracking-tight">{project.name}</p>
        <p className="mt-2 text-xs uppercase tracking-widest text-ink/50 group-hover:text-paper/60">
          {project.class_config.length > 0
            ? project.class_config.map((c) => c.name).join(" · ")
            : "no model registered yet"}
        </p>
      </Link>
      <button
        onClick={() => setMode("menu")}
        className="absolute right-4 top-4 border border-ink/30 px-2 py-1 text-[10px] font-bold uppercase tracking-widest opacity-0 hover:border-ink hover:bg-paper hover:text-ink group-hover:opacity-100 group-hover:border-paper/40 group-hover:text-paper group-hover:hover:border-ink group-hover:hover:bg-paper group-hover:hover:text-ink"
      >
        Edit
      </button>
    </div>
  );
}

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
          className="border-l-2 border-ink bg-ink px-6 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
        >
          {createMutation.isPending ? "Creating…" : "Create"}
        </button>
      </form>
      {createMutation.isError && <FieldError error={createMutation.error} />}

      {projectsQuery.isLoading && <p className="text-sm text-ink/50">Loading…</p>}

      <div className="grid max-w-5xl grid-cols-1 gap-0 border-t-2 border-ink sm:grid-cols-2">
        {(projectsQuery.data ?? []).map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
        {projectsQuery.data?.length === 0 && (
          <p className="col-span-2 py-8 text-sm text-ink/50">No projects yet — create one above.</p>
        )}
      </div>
    </div>
  );
}
