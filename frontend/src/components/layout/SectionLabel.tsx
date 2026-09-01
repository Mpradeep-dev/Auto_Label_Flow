/** Editorial-register numbered section label — "01. REVIEW" in Swiss red.
 * The one recurring motif across every editorial page (PipelineHome,
 * dashboards, training runs). Never used in the instrument-register canvas. */
export function SectionLabel({ index, children }: { index: number; children: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-baseline gap-3">
      <span className="font-mono text-sm font-bold text-accent-ink">{String(index).padStart(2, "0")}.</span>
      <h2 className="text-sm font-bold uppercase tracking-widest text-ink">{children}</h2>
    </div>
  );
}
