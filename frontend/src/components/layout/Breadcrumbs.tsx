import { Link, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";

const PAGE_LABELS: Record<string, string> = {
  images: "Images",
  videos: "Videos",
  "auto-annotation": "Auto Annotation",
  review: "Review Queue",
  models: "Models",
  training: "Training Runs",
  export: "Export",
  settings: "Project Settings",
};

interface Crumb {
  label: string;
  to?: string;
}

// Deep routes like /projects/:projectId/datasets/:datasetId/images/:imageId/annotate
// leave users disoriented about where they are and how to get back out.
// This walks the current path against the route shapes in App.tsx and
// renders it as a trail — no per-page wiring required.
export function Breadcrumbs() {
  const { projectId, datasetId } = useParams<{ projectId?: string; datasetId?: string }>();
  const location = useLocation();

  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId!),
    enabled: !!projectId,
    staleTime: 60_000,
  });
  const datasetQuery = useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => api.getDataset(datasetId!),
    enabled: !!datasetId,
    staleTime: 60_000,
  });

  // Top-level pages (project list, account settings) have nothing to trail
  // back to — their own <h1> already says where you are.
  if (!projectId) return null;

  const afterProject = location.pathname.split(`/projects/${projectId}`)[1] ?? "";
  const segments = afterProject.split("/").filter(Boolean);

  const crumbs: Crumb[] = [
    { label: "Projects", to: "/projects" },
    { label: projectQuery.data?.name ?? "…", to: `/projects/${projectId}` },
  ];

  if (segments[0] === "datasets") {
    crumbs.push({ label: "Datasets", to: `/projects/${projectId}/datasets` });
    if (datasetId) {
      const toImages = `/projects/${projectId}/datasets/${datasetId}/images`;
      crumbs.push({ label: datasetQuery.data?.name ?? "…", to: toImages });
      if (segments.includes("statistics")) {
        crumbs.push({ label: "Statistics" });
      } else if (segments.includes("annotate")) {
        crumbs.push({ label: "Images", to: toImages });
        crumbs.push({ label: "Annotate" });
      }
    }
  } else if (segments[0]) {
    crumbs.push({ label: PAGE_LABELS[segments[0]] ?? segments[0] });
  }

  return (
    <nav
      aria-label="Breadcrumb"
      className="flex h-9 shrink-0 items-center gap-1.5 overflow-x-auto border-b-2 border-ink/10 bg-paper px-6 text-[11px] font-bold uppercase tracking-widest text-ink/60"
    >
      {crumbs.map((crumb, i) => {
        const isLast = i === crumbs.length - 1;
        return (
          <span key={i} className="flex shrink-0 items-center gap-1.5">
            {i > 0 && <span className="text-ink/20">/</span>}
            {crumb.to && !isLast ? (
              <Link to={crumb.to} className="hover:text-ink hover:underline">
                {crumb.label}
              </Link>
            ) : (
              <span className={isLast ? "text-ink" : ""}>{crumb.label}</span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
