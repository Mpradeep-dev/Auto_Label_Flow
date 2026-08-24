import { SectionLabel } from "@/components/layout/SectionLabel";

/** Sidebar entries not yet built land here instead of a dead link, so the
 * shell is fully navigable from Phase 1 on even though most pages arrive
 * in later phases (see PLAN build order). */
export function PlaceholderPage({ title, phase }: { title: string; phase: string }) {
  return (
    <div className="flex min-h-full flex-col items-start justify-center px-8 py-20 sm:px-16">
      <SectionLabel index={0}>Not built yet</SectionLabel>
      <h1 className="mb-4 text-5xl font-black uppercase tracking-tightest sm:text-7xl">{title}</h1>
      <p className="max-w-lg text-sm text-ink/60">{phase}</p>
    </div>
  );
}
