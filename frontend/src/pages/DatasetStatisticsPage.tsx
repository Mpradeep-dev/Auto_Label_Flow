import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";

function BigStat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border-2 border-ink p-6">
      <p className="text-[10px] font-bold uppercase tracking-widest text-ink/50">{label}</p>
      <p className="tabular text-4xl font-black">{value}</p>
      {sub && <p className="tabular mt-1 text-xs text-ink/40">{sub}</p>}
    </div>
  );
}

function Bar({ label, count, total }: { label: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="mb-3">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-bold uppercase tracking-widest">{label}</span>
        <span className="tabular text-ink/50">{count}</span>
      </div>
      <div className="h-2 w-full bg-muted">
        <div className="h-full bg-ink" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function DatasetStatisticsPage() {
  const { projectId, datasetId } = useParams<{ projectId: string; datasetId: string }>();

  const statsQuery = useQuery({
    queryKey: ["dataset-statistics", datasetId],
    queryFn: () => api.getDatasetStatistics(datasetId!),
    enabled: !!datasetId,
  });
  const errorsQuery = useQuery({
    queryKey: ["error-analysis", datasetId],
    queryFn: () => api.getErrorAnalysis(datasetId!),
    enabled: !!datasetId,
  });

  if (!projectId || !datasetId) return null;
  const stats = statsQuery.data;

  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <SectionLabel index={1}>Dataset statistics</SectionLabel>
      <h1 className="mb-12 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Statistics
      </h1>

      {stats && (
        <>
          <div className="mb-12 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <BigStat label="Total images" value={String(stats.total_images)} />
            <BigStat
              label="Completion"
              value={`${stats.completion_pct}%`}
              sub={`${stats.reviewed_images} reviewed · ${stats.pending_images} pending`}
            />
            <BigStat label="Total annotations" value={String(stats.total_annotations)} />
            <BigStat
              label="Avg confidence"
              value={stats.average_confidence != null ? stats.average_confidence.toFixed(2) : "—"}
            />
          </div>

          <div className="mb-12 grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div>
              <SectionLabel index={2}>By class</SectionLabel>
              {Object.entries(stats.annotations_by_class).map(([name, count]) => (
                <Bar key={name} label={name} count={count} total={stats.total_annotations} />
              ))}
              {Object.keys(stats.annotations_by_class).length === 0 && (
                <p className="text-sm text-ink/40">No annotations yet.</p>
              )}
            </div>
            <div>
              <SectionLabel index={3}>By source</SectionLabel>
              {(["AUTO", "HUMAN", "CORRECTED"] as const).map((source) => (
                <Bar
                  key={source}
                  label={source}
                  count={stats.annotations_by_source[source] ?? 0}
                  total={stats.total_annotations}
                />
              ))}
            </div>
          </div>

          <SectionLabel index={4}>Quality signals</SectionLabel>
          <div className="mb-12 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <BigStat label="Low confidence" value={String(stats.low_confidence_predictions)} />
            <BigStat label="Suspicious cones" value={String(stats.suspicious_cones)} />
            <BigStat
              label="Acceptance rate"
              value={
                stats.auto_label_acceptance.acceptance_rate != null
                  ? `${stats.auto_label_acceptance.acceptance_rate}%`
                  : "—"
              }
              sub={
                stats.auto_label_acceptance.total_auto_predictions > 0
                  ? `${stats.auto_label_acceptance.accepted} accepted · ${stats.auto_label_acceptance.corrected} corrected · ${stats.auto_label_acceptance.rejected} rejected`
                  : "no auto predictions yet"
              }
            />
          </div>

          {errorsQuery.data && errorsQuery.data.total_categorized_deletions > 0 && (
            <>
              <SectionLabel index={5}>Error analysis</SectionLabel>
              <p className="tabular mb-4 text-xs uppercase tracking-widest text-ink/50">
                {errorsQuery.data.total_categorized_deletions} categorized deletions
              </p>
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
                <div>
                  <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/40">By category</p>
                  {Object.entries(errorsQuery.data.by_category).map(([cat, count]) => (
                    <Bar key={cat} label={cat} count={count} total={errorsQuery.data!.total_categorized_deletions} />
                  ))}
                </div>
                <div>
                  <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/40">By reason</p>
                  {Object.entries(errorsQuery.data.by_reason).map(([reason, count]) => (
                    <Bar
                      key={reason}
                      label={reason}
                      count={count}
                      total={errorsQuery.data!.total_categorized_deletions}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
