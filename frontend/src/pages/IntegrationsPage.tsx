import { IntegrationsSection } from "@/components/settings/IntegrationsPanel";

// Top-level route (/settings) — deliberately NOT nested under
// /projects/:projectId. Kaggle/Roboflow are account-wide connections, so
// gating them behind "pick a project first" was a real flow bug: the
// project-scoped SettingsPage used to also render this panel, which meant
// you couldn't connect Roboflow without already having a project to view
// it from. This page is the fix — always reachable from the sidebar.
export function IntegrationsPage() {
  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <h1 className="mb-4 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Settings
      </h1>
      <p className="mb-12 max-w-2xl text-sm text-ink/60">
        Account-wide connections, shared across every project. Per-project settings — name, class taxonomy, quality
        rule packs — live inside each project's own Settings page instead.
      </p>
      <IntegrationsSection sectionIndex={1} />
    </div>
  );
}
