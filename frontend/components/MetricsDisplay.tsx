"use client";

interface MetricCardProps {
  label: string;
  value: string;
  highlight?: boolean;
}

function MetricCard({ label, value, highlight }: MetricCardProps) {
  return (
    <div className={`border rounded-lg p-4 text-center animate-fade-in ${highlight ? "bg-emerald-950/30 border-emerald-700/40" : "bg-[var(--bg-secondary)] border-[var(--border)]"}`}>
      <div className={`text-2xl font-bold font-mono ${highlight ? "text-emerald-400" : "text-violet-400"}`}>{value}</div>
      <div className="text-[11px] text-zinc-500 mt-1 uppercase tracking-wider font-medium">
        {label}
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-widest text-zinc-600 pt-1 pb-0.5 border-b border-zinc-800/60">
      {children}
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function MetricsDisplay({ metrics }: { metrics: any }) {
  if (!metrics) return null;

  // Semantic accuracy cards (Stage 1 or Stage 2 DRN)
  const semanticCards: { label: string; value: string }[] = [];
  if (metrics.miou !== undefined) {
    semanticCards.push({ label: "mIoU",      value: `${(metrics.miou * 100).toFixed(2)}%` });
    semanticCards.push({ label: "Pixel Acc", value: `${(metrics.overall_accuracy * 100).toFixed(2)}%` });
    semanticCards.push({ label: "Mean Acc",  value: `${(metrics.mean_accuracy * 100).toFixed(2)}%` });
    semanticCards.push({ label: "FW-IoU",    value: `${(metrics.fwiou * 100).toFixed(2)}%` });
  }
  if (metrics.drn) {
    semanticCards.push({ label: "DRN mIoU",      value: `${(metrics.drn.miou * 100).toFixed(2)}%` });
    semanticCards.push({ label: "DRN Pixel Acc", value: `${(metrics.drn.overall_accuracy * 100).toFixed(2)}%` });
    semanticCards.push({ label: "DRN Mean Acc",  value: `${(metrics.drn.mean_accuracy * 100).toFixed(2)}%` });
    semanticCards.push({ label: "DRN FW-IoU",    value: `${(metrics.drn.fwiou * 100).toFixed(2)}%` });
  }

  // Image quality cards
  const imageCards: { label: string; value: string }[] = [];
  if (metrics.image_quality) {
    const iq = metrics.image_quality;
    if (iq.ssim  !== undefined) imageCards.push({ label: "SSIM ↑",   value: iq.ssim.toFixed(4) });
    if (iq.psnr  !== undefined) imageCards.push({ label: "PSNR ↑",   value: `${iq.psnr.toFixed(2)} dB` });
    if (iq.lpips !== undefined) imageCards.push({ label: "LPIPS ↓",  value: iq.lpips.toFixed(4) });
  }

  // Video quality cards (FID / FVD)
  const videoCards: { label: string; value: string }[] = [];
  if (metrics.fid     != null) videoCards.push({ label: "FID ↓",     value: metrics.fid.toFixed(2) });
  if (metrics.fvd_i3d != null) videoCards.push({ label: "FVD-I3D ↓", value: metrics.fvd_i3d.toFixed(2) });
  if (metrics.fvd_videomae != null) videoCards.push({ label: "FVD-VideoMAE ↓", value: metrics.fvd_videomae.toFixed(2) });

  const perClass = metrics.per_class || (metrics.drn && metrics.drn.per_class);

  return (
    <div className="space-y-4">
      {/* Semantic accuracy */}
      {semanticCards.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Semantic Accuracy</SectionLabel>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {semanticCards.map((c) => <MetricCard key={c.label} {...c} />)}
          </div>
        </div>
      )}

      {/* Image quality */}
      {imageCards.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Image Quality (per-frame)</SectionLabel>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {imageCards.map((c) => <MetricCard key={c.label} {...c} />)}
          </div>
        </div>
      )}

      {/* Video quality — FID / FVD */}
      {videoCards.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Video Quality — FID / FVD (lower is better)</SectionLabel>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {videoCards.map((c) => <MetricCard key={c.label} label={c.label} value={c.value} highlight />)}
          </div>
          <p className="text-[10px] text-zinc-600 leading-relaxed">
            FID uses Inception-v3 features (frame-level). FVD-I3D uses cdfvd/I3D (i3d_pretrained_400.pt, Kinetics400).
            FVD-VideoMAE uses VideoMAE-Base (MCG-NJU/videomae-base, ViT-B/16). Both FVD scores are computed on a single video —
            interpret as quality indicators rather than statistically robust estimates.
          </p>
        </div>
      )}

      {/* Per-class table */}
      {perClass && (
        <div className="space-y-2">
          <SectionLabel>Per-Class IoU</SectionLabel>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-zinc-500 font-medium uppercase text-[10px] tracking-wider">Class</th>
                  <th className="text-right px-3 py-2 text-zinc-500 font-medium uppercase text-[10px] tracking-wider">IoU</th>
                  <th className="text-right px-3 py-2 text-zinc-500 font-medium uppercase text-[10px] tracking-wider">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perClass).map(([name, vals]) => {
                  const v = vals as { iou: number | null; accuracy: number | null };
                  return (
                    <tr key={name} className="border-b border-zinc-800/50 hover:bg-[var(--bg-card-hover)]">
                      <td className="px-3 py-1.5 text-zinc-300">{name}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-zinc-400">
                        {v.iou !== null ? `${(v.iou * 100).toFixed(1)}%` : "N/A"}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-zinc-400">
                        {v.accuracy !== null ? `${(v.accuracy * 100).toFixed(1)}%` : "N/A"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
