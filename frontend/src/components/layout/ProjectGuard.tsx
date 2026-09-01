import { Link, Navigate, Outlet, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";

/** Every `/projects/:projectId/*` route sits under this one guard instead
 * of each page independently fetching the project and silently ignoring a
 * failure (audit finding FE-03) — a mistyped URL, a stale bookmark, or a
 * link to a since-deleted project used to render a fully-functional-looking
 * empty project (live sidebar nav, "no images yet") with nothing anywhere
 * indicating the project doesn't actually exist. This fetches once, at the
 * route boundary, and shows an honest not-found/error state instead of
 * letting every child page fail open into a fake empty state. */
export function ProjectGuard() {
  const { projectId } = useParams<{ projectId: string }>();

  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId!),
    enabled: !!projectId,
    retry: (failureCount, error) => (error instanceof ApiError && error.status === 404 ? false : failureCount < 1),
  });

  if (!projectId) return <Navigate to="/projects" replace />;

  if (projectQuery.isLoading) {
    return <div className="flex h-full items-center justify-center text-sm text-ink/60">Loading…</div>;
  }

  if (projectQuery.isError) {
    const notFound = projectQuery.error instanceof ApiError && projectQuery.error.status === 404;
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 px-8 text-center">
        <p className="text-2xl font-black uppercase tracking-tight">
          {notFound ? "Project not found" : "Couldn't load this project"}
        </p>
        <p className="max-w-md text-sm text-ink/60">
          {notFound
            ? "It may have been deleted, or the link is out of date."
            : "Something went wrong reaching the server — check your connection and try again."}
        </p>
        <Link
          to="/projects"
          className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink"
        >
          Back to projects
        </Link>
      </div>
    );
  }

  return <Outlet />;
}
