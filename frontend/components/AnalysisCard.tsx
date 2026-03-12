"use client";

import { clsx } from "clsx";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Analysis = any;

function Section({ title, items, color }: { title: string; items: string[]; color: string }) {
  if (!items?.length) return null;
  return (
    <div>
      <h5 className={clsx("text-[10px] font-bold uppercase tracking-wider mb-2", color)}>{title}</h5>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-xs text-zinc-300">
            <span className={clsx("mt-0.5 flex-shrink-0 w-1.5 h-1.5 rounded-full mt-1.5", color.replace("text-", "bg-"))} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ScoreBadge({ score, label }: { score: number; label: string }) {
  const color =
    score >= 8 ? "text-emerald-400 border-emerald-500/40 bg-emerald-500/10" :
    score >= 6 ? "text-yellow-400 border-yellow-500/40 bg-yellow-500/10" :
    score >= 4 ? "text-orange-400 border-orange-500/40 bg-orange-500/10" :
                 "text-red-400 border-red-500/40 bg-red-500/10";
  return (
    <div className={clsx("flex flex-col items-center justify-center w-16 h-16 rounded-xl border flex-shrink-0", color)}>
      <span className="text-2xl font-bold leading-none">{score}</span>
      <span className="text-[9px] font-semibold uppercase mt-0.5">{label}</span>
    </div>
  );
}

export default function AnalysisCard({ analysis, isLoading }: { analysis: Analysis; isLoading?: boolean }) {
  if (isLoading) {
    return (
      <div className="bg-[var(--bg-card)] border border-violet-500/30 rounded-xl p-5 animate-pulse space-y-3">
        <div className="flex items-center gap-2">
          <svg className="animate-spin h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          <span className="text-xs text-violet-400 font-medium">Gemini is analysing results...</span>
        </div>
        <div className="h-3 bg-zinc-800 rounded w-3/4" />
        <div className="h-3 bg-zinc-800 rounded w-1/2" />
      </div>
    );
  }

  if (!analysis) return null;

  // Comparison mode
  if (analysis.winner !== undefined) {
    const winnerColor = analysis.winner === "A" ? "text-violet-400" : analysis.winner === "B" ? "text-emerald-400" : "text-zinc-400";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const configDiff: Array<{ param: string; job_a: any; job_b: any; changed: boolean }> = analysis.config_diff || [];
    const changedRows = configDiff.filter((r) => r.changed);
    const unchangedRows = configDiff.filter((r) => !r.changed);

    return (
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 space-y-4 animate-fade-in">
        {/* Header + winner */}
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">AI Comparison</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 font-medium">Gemini</span>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">{analysis.summary}</p>
          </div>
          <div className={clsx("flex flex-col items-center flex-shrink-0 px-3 py-2 rounded-lg border",
            analysis.winner === "A" ? "border-violet-500/30 bg-violet-500/10" :
            analysis.winner === "B" ? "border-emerald-500/30 bg-emerald-500/10" : "border-zinc-700 bg-zinc-800")}>
            <span className="text-[9px] text-zinc-500 uppercase">Winner</span>
            <span className={clsx("text-xl font-bold", winnerColor)}>
              {analysis.winner === "tie" ? "Tie" : `Job ${analysis.winner}`}
            </span>
          </div>
        </div>

        {/* Config diff table */}
        {configDiff.length > 0 && (
          <div className="pt-2 border-t border-[var(--border)]">
            <h5 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">
              Configuration Comparison
              {changedRows.length > 0 && (
                <span className="ml-2 px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-400">
                  {changedRows.length} changed
                </span>
              )}
            </h5>
            <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
              <table className="w-full text-[11px] font-mono">
                <thead>
                  <tr className="bg-zinc-900/60">
                    <th className="text-left px-3 py-1.5 text-zinc-500 font-semibold uppercase tracking-wider">Parameter</th>
                    <th className="text-center px-3 py-1.5 text-violet-400 font-semibold">Job A ({analysis.job_id1})</th>
                    <th className="text-center px-3 py-1.5 text-emerald-400 font-semibold">Job B ({analysis.job_id2})</th>
                    <th className="text-center px-3 py-1.5 text-zinc-500 font-semibold uppercase tracking-wider">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  {changedRows.map((row) => (
                    <tr key={row.param} className="border-t border-yellow-500/20 bg-yellow-500/5">
                      <td className="px-3 py-1.5 text-yellow-300 font-semibold">{row.param}</td>
                      <td className="px-3 py-1.5 text-center text-violet-300">{String(row.job_a ?? "—")}</td>
                      <td className="px-3 py-1.5 text-center text-emerald-300">{String(row.job_b ?? "—")}</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-yellow-500/20 text-yellow-400 uppercase">changed</span>
                      </td>
                    </tr>
                  ))}
                  {unchangedRows.map((row) => (
                    <tr key={row.param} className="border-t border-[var(--border)] opacity-50">
                      <td className="px-3 py-1.5 text-zinc-500">{row.param}</td>
                      <td className="px-3 py-1.5 text-center text-zinc-400">{String(row.job_a ?? "—")}</td>
                      <td className="px-3 py-1.5 text-center text-zinc-400">{String(row.job_b ?? "—")}</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className="text-zinc-700 text-[9px]">same</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Metrics comparison */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-[var(--border)]">
          <Section title="Improvements" items={analysis.improvements} color="text-emerald-400" />
          <Section title="Regressions" items={analysis.regressions} color="text-red-400" />
        </div>
        {analysis.config_impact?.length > 0 && (
          <div className="pt-2 border-t border-[var(--border)]">
            <Section title="Config Impact Analysis" items={analysis.config_impact} color="text-yellow-400" />
          </div>
        )}
        {analysis.recommendation && (
          <div className="pt-2 border-t border-[var(--border)] flex gap-2">
            <svg className="w-4 h-4 text-violet-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <p className="text-xs text-zinc-300">{analysis.recommendation}</p>
          </div>
        )}
      </div>
    );
  }

  // Single job analysis
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 space-y-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-start gap-3">
        {analysis.score && (
          <ScoreBadge score={analysis.score} label={analysis.score_label || ""} />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">AI Analysis</span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 font-medium">Gemini</span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">{analysis.summary}</p>
        </div>
      </div>

      {/* Grid sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-[var(--border)]">
        <Section title="Observations" items={analysis.observations} color="text-zinc-400" />
        <Section title="Issues" items={analysis.issues} color="text-orange-400" />
        <Section title="Class Insights" items={analysis.class_insights} color="text-yellow-400" />
        <Section title="Recommendations" items={analysis.recommendations} color="text-violet-400" />
      </div>

      {/* Next steps */}
      {analysis.next_steps?.length > 0 && (
        <div className="pt-3 border-t border-[var(--border)]">
          <h5 className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 mb-2">Next Steps</h5>
          <ol className="space-y-1.5">
            {analysis.next_steps.map((step: string, i: number) => (
              <li key={i} className="flex gap-2 text-xs text-zinc-300">
                <span className="flex-shrink-0 w-4 h-4 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-[9px] flex items-center justify-center font-bold">
                  {i + 1}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
