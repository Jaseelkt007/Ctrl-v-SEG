"use client";

import { useEffect, useRef } from "react";
import { clsx } from "clsx";

interface LogEntry {
  message: string;
  type?: "info" | "error" | "success";
}

export default function LogViewer({
  logs,
  status,
}: {
  logs: LogEntry[];
  status: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll only the log container, never the whole page
  useEffect(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const statusColor: Record<string, string> = {
    running: "text-yellow-400",
    completed: "text-emerald-400",
    evaluated: "text-emerald-400",
    error: "text-red-400",
    queued: "text-zinc-500",
  };

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[#0c0c10] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--bg-secondary)] border-b border-[var(--border)]">
        <span className="text-xs font-medium text-zinc-400 tracking-wide uppercase">
          Pipeline Log
        </span>
        {status && (
          <span
            className={clsx(
              "text-[10px] font-semibold uppercase tracking-wider",
              statusColor[status] || "text-zinc-500"
            )}
          >
            {status === "running" && (
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-yellow-400 mr-1.5 animate-pulse" />
            )}
            {status}
          </span>
        )}
      </div>
      <div
        ref={containerRef}
        className="p-3 h-[220px] overflow-y-auto font-mono text-[11px] leading-relaxed space-y-0.5"
      >
        {logs.length === 0 ? (
          <p className="text-zinc-600 italic">Waiting to start...</p>
        ) : (
          logs.map((log, i) => (
            <p
              key={i}
              className={clsx(
                "whitespace-pre-wrap break-all",
                log.type === "error" && "text-red-400",
                log.type === "success" && "text-emerald-400",
                !log.type && "text-zinc-400"
              )}
            >
              {log.message}
            </p>
          ))
        )}
      </div>
    </div>
  );
}
