import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";

// Shared by the Datasets page (import) and Export page (export) — both
// need "pick a connected Roboflow workspace/project", the one piece those
// two flows actually have in common. Reads live off `listRoboflowProjects`
// rather than asking for free-typed slugs, since a typo there fails deep
// inside a background job instead of at input time.
export function RoboflowProjectSelect({
  workspace,
  project,
  onChange,
}: {
  workspace: string;
  project: string;
  onChange: (workspace: string, project: string) => void;
}) {
  const integrationsQuery = useQuery({ queryKey: ["integrations"], queryFn: () => api.listIntegrations() });
  const connected = (integrationsQuery.data ?? []).some((s) => s.provider === "ROBOFLOW" && s.connected);

  const projectsQuery = useQuery({
    queryKey: ["roboflow-projects"],
    queryFn: () => api.listRoboflowProjects(),
    enabled: connected,
  });

  if (integrationsQuery.isLoading) return null;

  if (!connected) {
    return (
      <p className="text-xs text-ink/50">
        Roboflow isn't connected yet.{" "}
        <Link to="/settings" className="underline">
          Connect it in Settings
        </Link>
        .
      </p>
    );
  }

  const projects = projectsQuery.data ?? [];
  const selected = `${workspace}/${project}`;

  return (
    <div>
      <select
        value={workspace && project ? selected : ""}
        onChange={(e) => {
          const [ws, proj] = e.target.value.split("/");
          onChange(ws ?? "", proj ?? "");
        }}
        className="w-full border-2 border-ink bg-paper px-3 py-2 text-sm font-semibold uppercase outline-none focus:border-accent"
      >
        <option value="">Select a Roboflow project…</option>
        {projects.map((p) => (
          <option key={`${p.workspace}/${p.project}`} value={`${p.workspace}/${p.project}`}>
            {p.workspace}/{p.project} ({p.name})
          </option>
        ))}
      </select>
      {projectsQuery.isLoading && <p className="mt-1 text-[10px] text-ink/40">Loading projects…</p>}
      {projects.length === 0 && !projectsQuery.isLoading && (
        <p className="mt-1 text-[10px] text-ink/40">No projects found for this Roboflow account.</p>
      )}
    </div>
  );
}
