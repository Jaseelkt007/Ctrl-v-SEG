"use client";

import { useState, useEffect, useCallback } from "react";
import { clsx } from "clsx";
import { healthCheck, getInfo, listJobs, compareJobs, deleteJob } from "@/lib/api";
import StagePanel from "@/components/StagePanel";
import AnalysisCard from "@/components/AnalysisCard";

type Tab = "stage1" | "stage2" | "history";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type JobEntry = any;

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("stage1");
  const [connected, setConnected] = useState(false);
  const [gpuName, setGpuName] = useState("");
  const [checkpointInfo, setCheckpointInfo] = useState<{ s1Step?: number; s2Step?: number }>({});
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const [connecting, setConnecting] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [preloadS1, setPreloadS1] = useState<Record<string, unknown> | null>(null);
  const [preloadS2, setPreloadS2] = useState<Record<string, unknown> | null>(null);
  const [compareJob1, setCompareJob1] = useState("");
  const [compareJob2, setCompareJob2] = useState("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [compareResult, setCompareResult] = useState<any>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const copyJobId = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(id).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    });
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    tryConnect();
  }, []);

  const tryConnect = async () => {
    setConnecting(true);
    try {
      const health = await healthCheck();
      setConnected(true);
      setGpuName(health.gpu_name || "");
      const info = await getInfo();
      setCheckpointInfo({
        s1Step: info.stage1?.step,
        s2Step: info.stage2?.step,
      });
    } catch {
      setConnected(false);
    }
    setConnecting(false);
  };

  const loadHistory = async () => {
    try {
      const data = await listJobs();
      setJobs(data.jobs || []);
    } catch {
      /* ignore */
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDeleting(true);
    try {
      await deleteJob(id);
      setJobs((prev) => prev.filter((j) => j.job_id !== id));
      if (pendingDelete === id) setPendingDelete(null);
    } catch {
      /* ignore */
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCompare = async () => {
    if (!compareJob1 || !compareJob2) return;
    setIsComparing(true);
    setCompareResult(null);
    try {
      const data = await compareJobs(compareJob1, compareJob2);
      setCompareResult(data);
    } catch (e) {
      console.error("Compare error:", e);
    } finally {
      setIsComparing(false);
    }
  };

  useEffect(() => {
    if (activeTab === "history") loadHistory();
  }, [activeTab]);

  const tabs: { key: Tab; label: string; desc: string; num: string }[] = [
    { key: "stage1", label: "Stage 1: Semantic Predictor", desc: "RGB → Semantic Maps", num: "1" },
    { key: "stage2", label: "Stage 2: Video Generator", desc: "Semantic Maps → RGB Video", num: "2" },
    { key: "history", label: "History", desc: "Past Jobs", num: "•" },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* ---- Connection Bar ---- */}
      <div className="sticky top-0 z-50 bg-[var(--bg-secondary)]/80 backdrop-blur-md border-b border-[var(--border)]">
        <div className="max-w-[1440px] mx-auto px-6 py-2 flex items-center gap-3 text-xs">
          <span className="text-zinc-500 font-medium">Backend:</span>
          <div
            className={clsx(
              "px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
              connected
                ? "bg-emerald-500/15 text-emerald-400"
                : "bg-red-500/15 text-red-400"
            )}
          >
            {connecting ? "Connecting..." : connected ? "Connected" : "Disconnected"}
          </div>
          {!connected && !connecting && (
            <button
              onClick={tryConnect}
              className="ml-2 px-3 py-1 text-[11px] font-medium bg-violet-600 text-white rounded hover:bg-violet-500 transition-colors"
            >
              Retry
            </button>
          )}
          {gpuName && (
            <span className="ml-auto text-zinc-600 font-mono text-[10px]">
              GPU: {gpuName}
              {checkpointInfo.s1Step && ` · S1: step ${checkpointInfo.s1Step}`}
              {checkpointInfo.s2Step && ` · S2: step ${checkpointInfo.s2Step}`}
            </span>
          )}
        </div>
      </div>

      {/* ---- Header ---- */}
      <header className="border-b border-[var(--border)] bg-gradient-to-b from-[var(--bg-secondary)] to-[var(--bg-primary)]">
        <div className="max-w-[1440px] mx-auto px-6 pt-8 pb-6">
          <h1 className="text-3xl font-bold tracking-tight">
            Ctrl-V<span className="text-violet-400">-Seg</span>
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Semantic-Conditioned Autonomous Driving Video Generation · KITTI-360
          </p>
        </div>
      </header>

      {/* ---- Tabs ---- */}
      <nav className="border-b border-[var(--border)]">
        <div className="max-w-[1440px] mx-auto px-6 flex">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={clsx(
                "flex items-center gap-2.5 px-5 py-3.5 text-sm font-medium border-b-2 transition-colors",
                activeTab === tab.key
                  ? "text-zinc-100 border-violet-500"
                  : "text-zinc-500 border-transparent hover:text-zinc-300 hover:bg-white/[0.02]"
              )}
            >
              <span
                className={clsx(
                  "inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold",
                  activeTab === tab.key
                    ? "bg-violet-500 text-white"
                    : "bg-[var(--bg-card)] border border-[var(--border)] text-zinc-500"
                )}
              >
                {tab.num}
              </span>
              <span>{tab.label}</span>
              <span className="text-[10px] text-zinc-600">{tab.desc}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* ---- Content ---- */}
      <main className="flex-1 max-w-[1440px] mx-auto w-full px-6 py-6">
        {activeTab === "stage1" && <StagePanel stage={1} initialConfig={preloadS1} />}
        {activeTab === "stage2" && <StagePanel stage={2} initialConfig={preloadS2} />}
        {activeTab === "history" && (
          <div className="max-w-3xl mx-auto animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Job History</h2>
              <button
                onClick={loadHistory}
                className="px-4 py-1.5 text-xs font-medium bg-[var(--bg-card)] border border-[var(--border)] rounded-lg text-zinc-400 hover:text-zinc-200 transition-colors"
              >
                Refresh
              </button>
            </div>
            {jobs.length === 0 ? (
              <p className="text-zinc-600 text-sm italic py-8 text-center">
                No jobs yet. Generate something first.
              </p>
            ) : (
              <div className="space-y-2">
                {jobs.map((job) => (
                  <div
                    key={job.job_id}
                    onClick={() => {
                      if (pendingDelete === job.job_id) return;
                      const cfg = {
                        sample_index: job.sample_index,
                        num_frames: job.num_frames,
                        num_inference_steps: job.num_inference_steps,
                        min_guidance_scale: job.min_guidance_scale,
                        max_guidance_scale: job.max_guidance_scale,
                        noise_aug_strength: job.noise_aug_strength,
                        seed: job.seed,
                        num_cond_bbox_frames: job.num_cond_bbox_frames,
                        conditioning_scale: job.conditioning_scale,
                      };
                      if (job.stage === 1) setPreloadS1(cfg);
                      else setPreloadS2(cfg);
                      setActiveTab(job.stage === 1 ? "stage1" : "stage2");
                    }}
                    className={clsx(
                      "flex items-center gap-3 px-4 py-3 bg-[var(--bg-card)] border rounded-lg transition-all",
                      pendingDelete === job.job_id
                        ? "border-red-500/50 bg-red-500/5 cursor-default"
                        : "border-[var(--border)] cursor-pointer hover:bg-[var(--bg-card-hover)] hover:border-zinc-600"
                    )}
                  >
                    <code className="text-sm font-mono font-bold text-violet-400">
                      {job.job_id}
                    </code>
                    {/* Copy button */}
                    <button
                      onClick={(e) => copyJobId(job.job_id, e)}
                      title="Copy job ID"
                      className="p-1 rounded text-zinc-600 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors flex-shrink-0"
                    >
                      {copiedId === job.job_id ? (
                        <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      )}
                    </button>
                    <span
                      className={clsx(
                        "px-2 py-0.5 text-[10px] font-semibold rounded uppercase",
                        job.stage === 1
                          ? "bg-violet-500/15 text-violet-300"
                          : "bg-emerald-500/15 text-emerald-300"
                      )}
                    >
                      Stage {job.stage}
                    </span>
                    {job.created_at && (
                      <span className="text-[10px] text-zinc-700 font-mono">
                        {new Date(job.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                      </span>
                    )}
                    {job.sample_index !== null && job.sample_index !== undefined && (
                      <span className="text-[10px] text-zinc-600">idx {job.sample_index}</span>
                    )}
                    <span className="ml-auto text-xs text-zinc-600">{job.status}</span>

                    {/* Delete controls */}
                    {pendingDelete === job.job_id ? (
                      <div className="flex items-center gap-1 ml-1" onClick={(e) => e.stopPropagation()}>
                        <span className="text-[10px] text-red-400 font-medium mr-1">Delete?</span>
                        <button
                          onClick={(e) => handleDelete(job.job_id, e)}
                          disabled={isDeleting}
                          title="Confirm delete"
                          className="p-1 rounded text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); setPendingDelete(null); }}
                          title="Cancel"
                          className="p-1 rounded text-zinc-500 hover:bg-zinc-700/50 transition-colors"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={(e) => { e.stopPropagation(); setPendingDelete(job.job_id); }}
                        title="Delete job"
                        className="p-1 rounded text-zinc-700 hover:text-red-400 hover:bg-red-500/10 transition-colors flex-shrink-0"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Compare section */}
            <div className="mt-8">
              <h3 className="text-sm font-semibold mb-3">Compare Two Jobs</h3>
              <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Job A</label>
                    <input
                      type="text"
                      value={compareJob1}
                      onChange={(e) => setCompareJob1(e.target.value)}
                      placeholder="job ID..."
                      className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-xs font-mono text-zinc-200 focus:outline-none focus:border-violet-500"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Job B</label>
                    <input
                      type="text"
                      value={compareJob2}
                      onChange={(e) => setCompareJob2(e.target.value)}
                      placeholder="job ID..."
                      className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-xs font-mono text-zinc-200 focus:outline-none focus:border-violet-500"
                    />
                  </div>
                </div>
                <button
                  onClick={handleCompare}
                  disabled={isComparing || !compareJob1 || !compareJob2}
                  className="w-full py-2 text-xs font-semibold bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isComparing ? (
                    <>
                      <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      Comparing with AI...
                    </>
                  ) : "Compare with Gemini AI"}
                </button>
                {compareResult && (
                  <AnalysisCard analysis={compareResult} />
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ---- Footer ---- */}
      <footer className="border-t border-[var(--border)] py-4 text-center text-[11px] text-zinc-600">
        Ctrl-V-Seg — KITTI-360 Semantic Video Generation — University of Stuttgart
      </footer>
    </div>
  );
}
