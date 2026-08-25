import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import { RoboflowImportSection } from "@/components/integrations/RoboflowImportSection";
import { classColor } from "@/config/classColors";

function FieldError({ error }: { error: unknown }) {
  if (!error) return null;
  const message = error instanceof ApiError ? error.message : (error as Error).message;
  return <p className="mt-2 text-xs text-accent">{message}</p>;
}

// --- Project identity + class taxonomy + quality packs -------------------

function ProjectSection({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId),
  });
  const project = projectQuery.data;

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [editing, setEditing] = useState(false);

  const saveMutation = useMutation({
    mutationFn: () => api.updateProject(projectId, { name: name.trim(), description: description.trim() || null }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      setEditing(false);
    },
  });

  const anatomicalPack = (project?.quality_rule_config?.anatomical_proximity ?? {}) as { enabled?: boolean };
  const hasConeLikeClass = (project?.class_config ?? []).some((c) => c.name.toLowerCase().includes("cone"));
  const packEffectivelyOn = anatomicalPack.enabled ?? hasConeLikeClass;

  const togglePackMutation = useMutation({
    mutationFn: (enabled: boolean) =>
      api.updateProject(projectId, {
        quality_rule_config: {
          ...(project?.quality_rule_config ?? {}),
          anatomical_proximity: { ...anatomicalPack, enabled },
        },
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["project", projectId] }),
  });

  if (!project) return null;

  return (
    <section>
      <SectionLabel index={1}>Project</SectionLabel>
      <div className="max-w-5xl border-2 border-ink">
        <div className="flex flex-wrap items-start justify-between gap-6 border-b-2 border-ink p-6">
          {!editing ? (
            <div>
              <p className="text-2xl font-bold uppercase tracking-tight">{project.name}</p>
              <p className="tabular mt-1 text-xs uppercase tracking-widest text-ink/50">/{project.slug}</p>
              {project.description && <p className="mt-3 max-w-xl text-sm text-ink/70">{project.description}</p>}
            </div>
          ) : (
            <div className="flex-1 space-y-3">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="PROJECT NAME"
                className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase tracking-wide outline-none focus:border-accent"
              />
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Description"
                rows={2}
                className="w-full border border-ink/30 bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
              />
              <FieldError error={saveMutation.error} />
            </div>
          )}
          <div className="flex shrink-0 gap-2">
            {editing ? (
              <>
                <button
                  onClick={() => saveMutation.mutate()}
                  disabled={!name.trim() || saveMutation.isPending}
                  className="border-2 border-ink bg-ink px-4 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
                >
                  {saveMutation.isPending ? "Saving…" : "Save"}
                </button>
                <button
                  onClick={() => setEditing(false)}
                  className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-muted"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                onClick={() => {
                  setName(project.name);
                  setDescription(project.description ?? "");
                  setEditing(true);
                }}
                className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
              >
                Edit
              </button>
            )}
          </div>
        </div>

        <div className="border-b-2 border-ink p-6">
          <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-ink/50">
            Class taxonomy — read from the primary detector's own weights, not typed in
          </p>
          <div className="flex flex-wrap gap-2">
            {project.class_config.length === 0 && (
              <p className="text-sm text-ink/50">No classes yet — register a detector model to seed this.</p>
            )}
            {project.class_config.map((c) => (
              <span
                key={c.id}
                className="flex items-center gap-2 border border-ink/30 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide"
              >
                <span className="h-2.5 w-2.5" style={{ backgroundColor: classColor(c.id) }} />
                {c.name}
                <span className="tabular text-ink/40">#{c.id}</span>
              </span>
            ))}
          </div>
        </div>

        <div className="p-6">
          <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-ink/50">Quality rule packs</p>
          <label className="flex max-w-xl items-start gap-3 text-sm">
            <input
              type="checkbox"
              checked={packEffectivelyOn}
              onChange={(e) => togglePackMutation.mutate(e.target.checked)}
              className="mt-1 h-4 w-4 accent-ink"
            />
            <span>
              <span className="font-bold uppercase tracking-wide">Anatomical proximity</span>{" "}
              <span className="tabular text-[10px] uppercase tracking-widest text-ink/40">
                CONE_NEAR_PLAYER · SUSPICIOUS_CONE
              </span>
              <br />
              <span className="text-ink/60">
                Flags a detection sitting suspiciously close to a player's foot instead of on the ground — needs a
                pose model configured on this project.{" "}
                {!anatomicalPack.enabled && hasConeLikeClass && "Currently on by default (a cone-like class exists)."}
              </span>
            </span>
          </label>
        </div>
      </div>
    </section>
  );
}

export function SettingsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  if (!projectId) return null;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <h1 className="mb-4 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Project settings
      </h1>
      <p className="mb-12 max-w-2xl text-sm text-ink/60">
        Looking for Kaggle or Roboflow?{" "}
        <Link to="/settings" className="font-semibold underline decoration-1 underline-offset-2 hover:text-accent">
          Those are account-wide, under Settings →
        </Link>
      </p>
      <ProjectSection projectId={projectId} />

      <section className="mt-12">
        <SectionLabel index={2}>Import</SectionLabel>
        <RoboflowImportSection projectId={projectId} />
      </section>
    </div>
  );
}
