"use client";

import { RefObject } from "react";
import { getFrameUrl } from "@/lib/api";

export default function FrameGallery({
  jobId,
  subdir,
  files,
  label,
  containerRef,
  onScroll,
}: {
  jobId: string;
  subdir: string;
  files: string[];
  label?: string;
  containerRef?: RefObject<HTMLDivElement>;
  onScroll?: (e: React.UIEvent<HTMLDivElement>) => void;
}) {
  if (!files || files.length === 0) return null;

  return (
    <div className="animate-fade-in">
      {label && (
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
          {label}
        </h4>
      )}
      <div ref={containerRef} onScroll={onScroll} className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin">
        {files.map((file) => (
          <img
            key={file}
            src={getFrameUrl(jobId, subdir, file)}
            alt={file}
            className="flex-shrink-0 h-[80px] rounded border border-[var(--border)] hover:border-violet-500 transition-colors cursor-pointer"
            loading="lazy"
            onClick={() => window.open(getFrameUrl(jobId, subdir, file), "_blank")}
          />
        ))}
      </div>
    </div>
  );
}
