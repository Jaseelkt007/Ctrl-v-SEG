"use client";

interface MetricCardProps {
  label: string;
  value: string;
}

function MetricCard({ label, value }: MetricCardProps) {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-4 text-center animate-fade-in">
      <div className="text-2xl font-bold text-violet-400 font-mono">{value}</div>
      <div className="text-[11px] text-zinc-500 mt-1 uppercase tracking-wider font-medium">
        {label}
      </div>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function MetricsDisplay({ metrics }: { metrics: any }) {
  if (!metrics) return null;

  const cards: { label: string; value: string }[] = [];

  // Stage 1 direct metrics
  if (metrics.miou !== undefined) {
    cards.push({ label: "mIoU", value: `${(metrics.miou * 100).toFixed(2)}%` });
    cards.push({ label: "Pixel Acc", value: `${(metrics.overall_accuracy * 100).toFixed(2)}%` });
    cards.push({ label: "Mean Acc", value: `${(metrics.mean_accuracy * 100).toFixed(2)}%` });
    cards.push({ label: "FW-IoU", value: `${(metrics.fwiou * 100).toFixed(2)}%` });
  }

  // Stage 2 DRN metrics
  if (metrics.drn) {
    cards.push({ label: "DRN mIoU", value: `${(metrics.drn.miou * 100).toFixed(2)}%` });
    cards.push({ label: "DRN Pixel Acc", value: `${(metrics.drn.overall_accuracy * 100).toFixed(2)}%` });
    cards.push({ label: "DRN Mean Acc", value: `${(metrics.drn.mean_accuracy * 100).toFixed(2)}%` });
    cards.push({ label: "DRN FW-IoU", value: `${(metrics.drn.fwiou * 100).toFixed(2)}%` });
  }

  // Image quality
  if (metrics.image_quality) {
    const iq = metrics.image_quality;
    if (iq.ssim !== undefined) cards.push({ label: "SSIM", value: iq.ssim.toFixed(4) });
    if (iq.psnr !== undefined) cards.push({ label: "PSNR", value: `${iq.psnr.toFixed(2)} dB` });
    if (iq.lpips !== undefined) cards.push({ label: "LPIPS", value: iq.lpips.toFixed(4) });
  }

  const perClass = metrics.per_class || (metrics.drn && metrics.drn.per_class);

  return (
    <div className="space-y-4">
      {cards.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {cards.map((c) => (
            <MetricCard key={c.label} {...c} />
          ))}
        </div>
      )}

      {perClass && (
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
      )}
    </div>
  );
}
