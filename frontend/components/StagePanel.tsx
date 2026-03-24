"use client";

import { useState, useCallback, useRef, useEffect, RefObject } from "react";
import { clsx } from "clsx";
import {
  startStage1,
  startStage2,
  getJobStatus,
  evaluateJob,
  uploadGT,
  getGifUrl,
  getDownloadUrl,
  getDatasetSamples,
  getDatasetSampleByIndex,
  getThumbnailUrl,
  pollLogs,
  analyseJob,
} from "@/lib/api";
import PipelineFlow, { PipelineStep, StepState } from "./PipelineFlow";
import LogViewer from "./LogViewer";
import FrameGallery from "./FrameGallery";
import MetricsDisplay from "./MetricsDisplay";
import ParamInput from "./ParamInput";
import AnalysisCard from "./AnalysisCard";

interface LogEntry {
  message: string;
  type?: "info" | "error" | "success";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type JobResult = any;

// Map log lines to pipeline step indices
// Stage 1 steps: 0=Load Input, 1=Load Model, 2=Diffusion, 3=Decode, 4=Save
// Stage 2 steps: 0=Load Input, 1=Load Model, 2=Diffusion, 3=Decode RGB, 4=Save
function detectStepFromLog(line: string): number | null {
  const l = line.toLowerCase();
  if (l.includes("loading dataset") || l.includes("using uploaded") || l.includes("using control frames") || l.includes("loading dataset sample")) return 0;
  if (l.includes("loading stage") || l.includes("loading unet") || l.includes("loading pipeline") || l.includes("checkpoint weights")) return 1;
  if (l.includes("starting inference") || l.includes("inference params") || l.includes("input: image_init")) return 2;
  if (l.includes("decoding semantic") || l.includes("decoding latents") || l.includes("generated semantic frames")) return 3;
  if (l.includes("saving generated") || l.includes("saved gif") || l.includes("complete") || l.includes("save")) return 4;
  return null;
}

// Parse diffusion step from tqdm-like log: looks for patterns like "25/30" or "step 25"
function parseDiffusionStep(line: string): number | null {
  const m = line.match(/(\d+)\s*\/\s*(\d+)/);
  if (m) return parseInt(m[1]);
  const m2 = line.match(/step[:\s]+(\d+)/i);
  if (m2) return parseInt(m2[1]);
  return null;
}

export default function StagePanel({ stage, initialConfig }: { stage: 1 | 2; initialConfig?: Record<string, unknown> | null }) {
  const [inputMode, setInputMode] = useState<"dataset" | "upload" | "upload_frames" | "stage1">("dataset");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [stage1JobId, setStage1JobId] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string>("");
  const [firstFrame, setFirstFrame] = useState<File | null>(null);
  const [firstFramePreview, setFirstFramePreview] = useState<string>("");
  const [lastFrame, setLastFrame] = useState<File | null>(null);
  const [lastFramePreview, setLastFramePreview] = useState<string>("");
  const [samplePreview, setSamplePreview] = useState<{ rgb?: string; sem?: string } | null>(null);

  // Params
  const [numFrames, setNumFrames] = useState(25);
  const [steps, setSteps] = useState(30);
  const [minGuidance, setMinGuidance] = useState(stage === 1 ? 3.0 : 1.0);
  const [maxGuidance, setMaxGuidance] = useState(stage === 1 ? 7.0 : 3.0);
  const [condScale, setCondScale] = useState(1.0);
  const [noiseAug, setNoiseAug] = useState(0.01);
  const [seed, setSeed] = useState(1234);
  const [condFrames, setCondFrames] = useState(1);
  const [prompt, setPrompt] = useState(
    stage === 1
      ? "Autonomous driving scene, semantic segmentation"
      : "Photorealistic autonomous driving video, KITTI-360"
  );
  const [showInfo, setShowInfo] = useState(false);

  // State
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [result, setResult] = useState<JobResult>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalStep, setEvalStep] = useState<string>("");
  const [activeStep, setActiveStep] = useState<number>(-1);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [diffusionStep, setDiffusionStep] = useState(0);
  const [analysis, setAnalysis] = useState<unknown>(null);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const evalPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logOffsetRef = useRef(0);
  const gtInputRef = useRef<HTMLInputElement>(null);
  const predScrollRef = useRef<HTMLDivElement>(null);
  const gtScrollRef = useRef<HTMLDivElement>(null);

  const handlePredScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (gtScrollRef.current) gtScrollRef.current.scrollLeft = (e.target as HTMLDivElement).scrollLeft;
  }, []);

  const handleGtScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (predScrollRef.current) predScrollRef.current.scrollLeft = (e.target as HTMLDivElement).scrollLeft;
  }, []);

  const addLog = useCallback((msg: string, type?: "error" | "success") => {
    setLogs((prev) => [...prev, { message: msg, type }]);
    // Detect which pipeline step this log belongs to
    const step = detectStepFromLog(msg);
    if (step !== null) {
      setActiveStep(step);
      setCompletedSteps((prev) => {
        const next = new Set(prev);
        for (let i = 0; i < step; i++) next.add(i);
        return next;
      });
    }
    // Detect diffusion step progress
    if (msg.toLowerCase().includes("step") || msg.includes("/")) {
      const ds = parseDiffusionStep(msg);
      if (ds !== null) setDiffusionStep(ds);
    }
    // If complete, mark all steps done
    if (msg.toLowerCase().includes("generation complete") || msg.toLowerCase().includes("stage 1 generation complete") || msg.toLowerCase().includes("stage 2 generation complete")) {
      setActiveStep(-1);
      setCompletedSteps(new Set([0, 1, 2, 3, 4]));
    }
  }, []);

  // Start polling — tries log endpoint first, falls back to status endpoint
  const startLogPolling = useCallback(
    (jid: string) => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      logOffsetRef.current = 0;
      let useLogEndpoint = true;

      const poll = async () => {
        // Try the rich logs endpoint; fall back to plain status if it's not available (404)
        if (useLogEndpoint) {
          try {
            const data = await pollLogs(jid, logOffsetRef.current);
            if (data.lines.length > 0) {
              for (const line of data.lines) {
                const isErr = line.toLowerCase().includes("error");
                const isOk = line.toLowerCase().includes("complete") || line.toLowerCase().includes("success");
                addLog(line, isErr ? "error" : isOk ? "success" : undefined);
              }
              logOffsetRef.current = data.offset;
            }
            const done = data.status === "completed" || data.status === "evaluated";
            setStatus(data.status);
            if (done) {
              loadJobResult(jid);
              setIsGenerating(false);
              if (pollingRef.current) clearInterval(pollingRef.current);
            } else if (data.status === "error") {
              setIsGenerating(false);
              if (pollingRef.current) clearInterval(pollingRef.current);
            }
            return;
          } catch {
            // logs endpoint not yet available — switch to status polling
            useLogEndpoint = false;
          }
        }

        // Fallback: poll /status only
        try {
          const data = await getJobStatus(jid);
          const s = data.status ?? "running";
          setStatus(s);
          if (s === "completed" || s === "evaluated") {
            setResult(data);
            setIsGenerating(false);
            addLog("Generation complete!", "success");
            if (pollingRef.current) clearInterval(pollingRef.current);
          } else if (s === "error") {
            addLog(`Error: ${data.error ?? "unknown"}`, "error");
            setIsGenerating(false);
            if (pollingRef.current) clearInterval(pollingRef.current);
          }
        } catch {
          // keep trying
        }
      };

      poll();
      pollingRef.current = setInterval(poll, 2000);
    },
    [addLog]
  );

  const loadJobResult = async (jid: string) => {
    try {
      const data = await getJobStatus(jid);
      setResult(data);
      setStatus(data.status ?? "");
      // Restore all parameters from the job's saved config
      const c = data.config;
      if (c) {
        if (c.sample_index !== undefined && c.sample_index !== null) setSampleIndex(Number(c.sample_index));
        if (c.num_frames !== undefined && c.num_frames !== null) setNumFrames(Number(c.num_frames));
        if (c.num_inference_steps !== undefined && c.num_inference_steps !== null) setSteps(Number(c.num_inference_steps));
        if (c.min_guidance_scale !== undefined && c.min_guidance_scale !== null) setMinGuidance(Number(c.min_guidance_scale));
        if (c.max_guidance_scale !== undefined && c.max_guidance_scale !== null) setMaxGuidance(Number(c.max_guidance_scale));
        if (c.noise_aug_strength !== undefined && c.noise_aug_strength !== null) setNoiseAug(Number(c.noise_aug_strength));
        if (c.seed !== undefined && c.seed !== null) setSeed(Number(c.seed));
        if (c.num_cond_bbox_frames !== undefined && c.num_cond_bbox_frames !== null) setCondFrames(Number(c.num_cond_bbox_frames));
        if (c.conditioning_scale !== undefined && c.conditioning_scale !== null) setCondScale(Number(c.conditioning_scale));
        if (c.control_job_id) { setStage1JobId(c.control_job_id); setInputMode("stage1"); }
        else if (c.sample_index !== undefined && c.sample_index !== null) setInputMode("dataset");
      }
    } catch {
      /* ignore */
    }
  };

  // Preview sample
  const previewSample = async () => {
    setIsPreviewing(true);
    setSamplePreview(null);
    try {
      const s = await getDatasetSampleByIndex(sampleIndex);
      setSamplePreview({
        rgb: s.rgb_thumb ? getThumbnailUrl(s.rgb_thumb) : undefined,
        sem: s.sem_thumb ? getThumbnailUrl(s.sem_thumb) : undefined,
      });
    } catch (e) {
      addLog(`Preview failed: ${e}`, "error");
    } finally {
      setIsPreviewing(false);
    }
  };

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (ev) => setUploadPreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  // Handle first frame upload
  const handleFirstFrameUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFirstFrame(file);
      const reader = new FileReader();
      reader.onload = (ev) => setFirstFramePreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  // Handle last frame upload
  const handleLastFrameUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setLastFrame(file);
      const reader = new FileReader();
      reader.onload = (ev) => setLastFramePreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  // Generate
  const handleGenerate = async () => {
    setIsGenerating(true);
    setLogs([]);
    setResult(null);
    setStatus("queued");
    setActiveStep(-1);
    setCompletedSteps(new Set());
    setDiffusionStep(0);
    setAnalysis(null);
    addLog(`Starting Stage ${stage} generation...`);

    try {
      let data;
      if (stage === 1) {
        data = await startStage1({
          sample_index: inputMode === "dataset" ? sampleIndex : undefined,
          image: inputMode === "upload" ? (uploadedFile ?? undefined) : undefined,
          first_frame: inputMode === "upload_frames" ? (firstFrame ?? undefined) : undefined,
          last_frame: inputMode === "upload_frames" ? (lastFrame ?? undefined) : undefined,
          num_frames: numFrames,
          num_inference_steps: steps,
          min_guidance_scale: minGuidance,
          max_guidance_scale: maxGuidance,
          noise_aug_strength: noiseAug,
          seed,
          num_cond_bbox_frames: condFrames,
        });
      } else {
        data = await startStage2({
          sample_index: inputMode === "dataset" ? sampleIndex : undefined,
          control_job_id: inputMode === "stage1" ? stage1JobId : undefined,
          image: inputMode === "upload" ? (uploadedFile ?? undefined) : undefined,
          first_frame: inputMode === "upload_frames" ? (firstFrame ?? undefined) : undefined,
          last_frame: inputMode === "upload_frames" ? (lastFrame ?? undefined) : undefined,
          num_frames: numFrames,
          num_inference_steps: steps,
          min_guidance_scale: minGuidance,
          max_guidance_scale: maxGuidance,
          conditioning_scale: condScale,
          noise_aug_strength: noiseAug,
          seed,
        });
      }
      setJobId(data.job_id);
      setStatus("running");
      addLog(`Job created: ${data.job_id}`);
      startLogPolling(data.job_id);
    } catch (e) {
      addLog(`Error: ${e}`, "error");
      setIsGenerating(false);
    }
  };

  // Evaluate
  const handleEvaluate = async () => {
    if (!jobId) return;
    const jid = jobId; // capture at call time
    if (evalPollRef.current) clearInterval(evalPollRef.current);
    setIsEvaluating(true);
    setEvalStep(stage === 1 ? "Comparing predicted vs GT semantic IDs..." : "Running DRN + FID + FVD-I3D + FVD-VideoMAE...");
    addLog("Starting evaluation...");
    try {
      await evaluateJob(jid);
      addLog("Evaluation submitted, computing metrics...");
      setEvalStep(stage === 1 ? "Computing mIoU, pixel accuracy..." : "Computing DRN mIoU + SSIM/PSNR/LPIPS + FID + FVD-I3D + FVD-VideoMAE...");
      evalPollRef.current = setInterval(async () => {
        try {
          const data = await getJobStatus(jid);
          // detect completion by status OR by eval_results being present
          if (data.status === "evaluated" || data.eval_results) {
            setResult(data);
            setStatus("evaluated");
            setIsEvaluating(false);
            setEvalStep("");
            addLog("Evaluation complete!", "success");
            if (evalPollRef.current) { clearInterval(evalPollRef.current); evalPollRef.current = null; }
          } else if (data.status === "error") {
            setIsEvaluating(false);
            setEvalStep("");
            if (evalPollRef.current) { clearInterval(evalPollRef.current); evalPollRef.current = null; }
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch (e) {
      addLog(`Evaluation error: ${e}`, "error");
      setIsEvaluating(false);
      setEvalStep("");
    }
  };

  // Analyse with Gemini
  const handleAnalyse = async () => {
    if (!jobId) return;
    setIsAnalysing(true);
    setAnalysis(null);
    try {
      const data = await analyseJob(jobId);
      setAnalysis(data);
    } catch (e) {
      addLog(`Analysis error: ${e}`, "error");
    } finally {
      setIsAnalysing(false);
    }
  };

  // GT upload
  const handleGTUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!jobId || !e.target.files) return;
    addLog(`Uploading ${e.target.files.length} GT frames...`);
    try {
      const data = await uploadGT(jobId, e.target.files);
      addLog(`Uploaded ${data.num_frames} GT frames`, "success");
    } catch (err) {
      addLog(`Upload failed: ${err}`, "error");
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (evalPollRef.current) clearInterval(evalPollRef.current);
    };
  }, []);

  // Apply preloaded config from history
  useEffect(() => {
    if (!initialConfig) return;
    if (initialConfig.sample_index !== undefined && initialConfig.sample_index !== null)
      setSampleIndex(Number(initialConfig.sample_index));
    if (initialConfig.num_frames !== undefined && initialConfig.num_frames !== null) setNumFrames(Number(initialConfig.num_frames));
    if (initialConfig.num_inference_steps !== undefined && initialConfig.num_inference_steps !== null) setSteps(Number(initialConfig.num_inference_steps));
    if (initialConfig.min_guidance_scale !== undefined && initialConfig.min_guidance_scale !== null) setMinGuidance(Number(initialConfig.min_guidance_scale));
    if (initialConfig.max_guidance_scale !== undefined && initialConfig.max_guidance_scale !== null) setMaxGuidance(Number(initialConfig.max_guidance_scale));
    if (initialConfig.noise_aug_strength !== undefined && initialConfig.noise_aug_strength !== null) setNoiseAug(Number(initialConfig.noise_aug_strength));
    if (initialConfig.seed !== undefined && initialConfig.seed !== null) setSeed(Number(initialConfig.seed));
    if (initialConfig.num_cond_bbox_frames !== undefined && initialConfig.num_cond_bbox_frames !== null) setCondFrames(Number(initialConfig.num_cond_bbox_frames));
    if (initialConfig.conditioning_scale !== undefined && initialConfig.conditioning_scale !== null) setCondScale(Number(initialConfig.conditioning_scale));
  }, [initialConfig]);

  const stepDefs =
    stage === 1
      ? [
          { label: "Load Input",     sublabel: "dataset / upload" },
          { label: "Load Model",     sublabel: "UNet + VAE" },
          { label: "Diffusion",      sublabel: `${steps} steps` },
          { label: "Decode Semantic",sublabel: "VAE → IDs" },
          { label: "Save & Export",  sublabel: "GIF + frames" },
        ]
      : [
          { label: "Load Input",     sublabel: "semantic controls" },
          { label: "Load Model",     sublabel: "ControlNet + UNet" },
          { label: "Diffusion",      sublabel: `${steps} steps` },
          { label: "Decode RGB",     sublabel: "latents → frames" },
          { label: "Save & Export",  sublabel: "GIF + frames" },
        ];

  const hasError = status === "error";
  const pipelineSteps: PipelineStep[] = stepDefs.map((s, i) => {
    let state: StepState = "pending";
    if (completedSteps.has(i)) state = "done";
    if (activeStep === i) state = hasError ? "error" : "active";
    if (status === "completed" || status === "evaluated") state = "done";
    return { ...s, state };
  });

  const genDir = stage === 1 ? "pred_color" : "gen_rgb";
  const gtDir = stage === 1 ? "gt_color" : "gt_rgb";

  // Info panel content per stage
  const INFO = stage === 1 ? {
    title: "Stage 1 — Semantic Predictor (RGB → Semantic Maps)",
    tagline: "Given an initial RGB frame, predict future semantic segmentation sequences.",
    inputs: [
      { label: "First RGB Frame", detail: "192×704×3 — used as CLIP image conditioning" },
      { label: "GT Semantic IDs (25 frames)", detail: "25×192×704 int64 trainIDs (0–18) — VAE conditioning from dataset" },
    ],
    outputs: [
      { label: "Predicted Semantic Maps", detail: "25×192×704 — colorized + grayscale KITTI-360 semantic IDs" },
      { label: "output.gif", detail: "Animated visualization at 7 fps" },
    ],
    components: [
      { label: "CLIP Encoder", detail: "Encodes first RGB frame → 768d image embedding for UNet conditioning" },
      { label: "Semantic VAE Encoder", detail: "GT semantic IDs → one-hot (19 ch) → 4-ch latents (24×88)" },
      { label: "SVD UNet", detail: "Stable Video Diffusion UNet — iterative denoising over N steps" },
      { label: "Semantic VAE Decoder", detail: "Pred latents → 19-class logits → argmax → trainIDs (0–18)" },
    ],
    controls: [
      { label: "Steps", detail: "Diffusion denoising steps (more = higher quality, slower)" },
      { label: "Min/Max Guidance", detail: "Classifier-free guidance range — higher = stronger conditioning" },
      { label: "Noise Aug", detail: "Augmentation on conditioning image (0.01 recommended)" },
      { label: "Cond Frames", detail: "Number of conditioning frames passed to the UNet (1 = first frame only)" },
    ],
    checkpoint: "kitti360_semantic_predict_vae",
  } : {
    title: "Stage 2 — Sem2Video ControlNet (Semantic → RGB Video)",
    tagline: "Given semantic map sequences, generate photorealistic RGB driving video.",
    inputs: [
      { label: "Semantic Maps (25 frames)", detail: "25×192×704 — either from Stage 1 output or dataset GT" },
      { label: "First RGB Frame", detail: "192×704×3 — CLIP conditioning for scene appearance" },
    ],
    outputs: [
      { label: "Generated RGB Video", detail: "25×192×704×3 uint8 — photorealistic driving frames" },
      { label: "output.gif", detail: "Animated preview at 7 fps" },
    ],
    components: [
      { label: "Semantic VAE Encoder", detail: "Sem maps → one-hot (19 ch) → 4-ch control latents (24×88)" },
      { label: "CLIP Encoder", detail: "First RGB frame → 768d image embedding" },
      { label: "ControlNet", detail: "Encodes semantic control latents → residual features injected into UNet" },
      { label: "SVD UNet", detail: "Denoises conditioned on ControlNet + CLIP — outputs RGB latents" },
      { label: "RGB VAE Decoder", detail: "AutoencoderKLTemporalDecoder — latents → RGB frames (3×192×704)" },
    ],
    controls: [
      { label: "Steps", detail: "Diffusion denoising steps" },
      { label: "Ctrl Scale", detail: "ControlNet conditioning strength (1.0 = full strength)" },
      { label: "Min/Max Guidance", detail: "CFG range — lower values (1–3) work best for Stage 2" },
      { label: "Noise Aug", detail: "Conditioning image noise augmentation" },
    ],
    checkpoint: "kitti360_semantic2video_vae",
  };

  return (
    <div className="animate-fade-in">
      {/* Info toggle button */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-600">
            {stage === 1 ? "RGB → Semantic Maps" : "Semantic Maps → RGB Video"}
          </span>
        </div>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md border transition-colors bg-[var(--bg-card)] border-[var(--border)] text-zinc-400 hover:text-zinc-200 hover:border-zinc-500"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {showInfo ? "Hide Info" : "Stage Info"}
        </button>
      </div>

      {/* Expandable info panel */}
      {showInfo && (
        <div className="mb-5 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 animate-fade-in space-y-4">
          <div>
            <h3 className="text-sm font-bold text-zinc-100">{INFO.title}</h3>
            <p className="text-xs text-zinc-500 mt-1">{INFO.tagline}</p>
            <code className="text-[10px] text-violet-400 font-mono">checkpoint: {INFO.checkpoint}</code>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Inputs */}
            <div>
              <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Inputs</h4>
              <div className="space-y-2">
                {INFO.inputs.map((item) => (
                  <div key={item.label} className="bg-[var(--bg-secondary)] rounded-md px-2 py-1.5">
                    <p className="text-[10px] font-semibold text-violet-300">{item.label}</p>
                    <p className="text-[9px] text-zinc-600 mt-0.5">{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>
            {/* Outputs */}
            <div>
              <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Outputs</h4>
              <div className="space-y-2">
                {INFO.outputs.map((item) => (
                  <div key={item.label} className="bg-[var(--bg-secondary)] rounded-md px-2 py-1.5">
                    <p className="text-[10px] font-semibold text-emerald-400">{item.label}</p>
                    <p className="text-[9px] text-zinc-600 mt-0.5">{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>
            {/* Architecture components */}
            <div>
              <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Components</h4>
              <div className="space-y-2">
                {INFO.components.map((item) => (
                  <div key={item.label} className="bg-[var(--bg-secondary)] rounded-md px-2 py-1.5">
                    <p className="text-[10px] font-semibold text-zinc-300">{item.label}</p>
                    <p className="text-[9px] text-zinc-600 mt-0.5">{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>
            {/* Controls */}
            <div>
              <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Controls</h4>
              <div className="space-y-2">
                {INFO.controls.map((item) => (
                  <div key={item.label} className="bg-[var(--bg-secondary)] rounded-md px-2 py-1.5">
                    <p className="text-[10px] font-semibold text-zinc-300">{item.label}</p>
                    <p className="text-[9px] text-zinc-600 mt-0.5">{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <PipelineFlow
        stage={stage}
        steps={pipelineSteps}
        diffusionStep={activeStep === 2 ? diffusionStep : undefined}
        diffusionTotal={activeStep === 2 ? steps : undefined}
      />

      <div className="grid grid-cols-1 lg:grid-cols-[400px_1fr] gap-5">
        {/* ---- Left: Input + Params ---- */}
        <div className="space-y-5">
          {/* Input Card */}
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h3 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-4">
              Input
            </h3>

            {/* Mode Toggle */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <button
                onClick={() => setInputMode("dataset")}
                className={clsx(
                  "px-3 py-2 text-xs font-medium rounded-md transition-colors",
                  inputMode === "dataset"
                    ? "bg-violet-600 text-white"
                    : "bg-[var(--bg-input)] text-zinc-400 hover:text-zinc-200 border border-[var(--border)]"
                )}
              >
                Dataset Sample
              </button>
              <button
                onClick={() => setInputMode("upload")}
                className={clsx(
                  "px-3 py-2 text-xs font-medium rounded-md transition-colors",
                  inputMode === "upload"
                    ? "bg-violet-600 text-white"
                    : "bg-[var(--bg-input)] text-zinc-400 hover:text-zinc-200 border border-[var(--border)]"
                )}
              >
                Upload Frame
              </button>
              <button
                onClick={() => setInputMode("upload_frames")}
                className={clsx(
                  "px-3 py-2 text-xs font-medium rounded-md transition-colors",
                  inputMode === "upload_frames"
                    ? "bg-violet-600 text-white"
                    : "bg-[var(--bg-input)] text-zinc-400 hover:text-zinc-200 border border-[var(--border)]"
                )}
              >
                First & Last
              </button>
              {stage === 2 && (
                <button
                  onClick={() => setInputMode("stage1")}
                  className={clsx(
                    "px-3 py-2 text-xs font-medium rounded-md transition-colors",
                    inputMode === "stage1"
                      ? "bg-violet-600 text-white"
                      : "bg-[var(--bg-input)] text-zinc-400 hover:text-zinc-200 border border-[var(--border)]"
                  )}
                >
                  From Stage 1
                </button>
              )}
            </div>

            {/* Dataset mode */}
            {inputMode === "dataset" && (
              <div className="space-y-3">
                <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  Sample Index
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={sampleIndex}
                    onChange={(e) => setSampleIndex(parseInt(e.target.value) || 0)}
                    min={0}
                    className="flex-1 bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-sm font-mono text-zinc-200 focus:outline-none focus:border-violet-500"
                  />
                  <button
                    onClick={previewSample}
                    disabled={isPreviewing}
                    className="px-4 py-2 text-xs font-medium bg-[var(--bg-input)] border border-[var(--border)] rounded-md text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors disabled:opacity-50 flex items-center gap-1.5"
                  >
                    {isPreviewing ? (
                      <>
                        <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                        </svg>
                        Loading...
                      </>
                    ) : "Preview"}
                  </button>
                </div>
                {samplePreview && (
                  <div className="space-y-2 mt-2">
                    {samplePreview.rgb && (
                      <img
                        src={samplePreview.rgb}
                        alt="RGB"
                        className="w-full rounded-md border border-[var(--border)]"
                      />
                    )}
                    {samplePreview.sem && (
                      <img
                        src={samplePreview.sem}
                        alt="Semantic"
                        className="w-full rounded-md border border-[var(--border)]"
                      />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Upload mode */}
            {inputMode === "upload" && (
              <div className="space-y-3">
                <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  RGB Frame
                </label>
                <label className="flex flex-col items-center justify-center border-2 border-dashed border-[var(--border)] rounded-xl p-6 cursor-pointer hover:border-violet-500 hover:bg-violet-500/5 transition-all">
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                  <svg className="w-8 h-8 text-zinc-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-xs text-zinc-500">Click to upload</span>
                  <span className="text-[10px] text-zinc-600 mt-1">192 x 704 recommended</span>
                </label>
                {uploadPreview && (
                  <img
                    src={uploadPreview}
                    alt="Preview"
                    className="w-full rounded-md border border-[var(--border)]"
                  />
                )}
              </div>
            )}

            {/* Upload First & Last Frames mode */}
            {inputMode === "upload_frames" && (
              <div className="space-y-4">
                <div>
                  <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-2">
                    First Frame
                  </label>
                  <label className="flex flex-col items-center justify-center border-2 border-dashed border-[var(--border)] rounded-xl p-4 cursor-pointer hover:border-violet-500 hover:bg-violet-500/5 transition-all">
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleFirstFrameUpload}
                    />
                    <svg className="w-6 h-6 text-zinc-600 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-xs text-zinc-500">{firstFrame ? firstFrame.name : "Click to upload"}</span>
                  </label>
                  {firstFramePreview && (
                    <img
                      src={firstFramePreview}
                      alt="First Frame"
                      className="w-full rounded-md border border-[var(--border)] mt-2"
                    />
                  )}
                </div>

                <div>
                  <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-2">
                    Last Frame
                  </label>
                  <label className="flex flex-col items-center justify-center border-2 border-dashed border-[var(--border)] rounded-xl p-4 cursor-pointer hover:border-violet-500 hover:bg-violet-500/5 transition-all">
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleLastFrameUpload}
                    />
                    <svg className="w-6 h-6 text-zinc-600 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-xs text-zinc-500">{lastFrame ? lastFrame.name : "Click to upload"}</span>
                  </label>
                  {lastFramePreview && (
                    <img
                      src={lastFramePreview}
                      alt="Last Frame"
                      className="w-full rounded-md border border-[var(--border)] mt-2"
                    />
                  )}
                </div>
              </div>
            )}

            {/* Stage 1 mode (for stage 2) */}
            {inputMode === "stage1" && stage === 2 && (
              <div className="space-y-3">
                <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  Stage 1 Job ID
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={stage1JobId}
                    onChange={(e) => setStage1JobId(e.target.value)}
                    placeholder="e.g. a1b2c3d4"
                    className="flex-1 bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-sm font-mono text-zinc-200 focus:outline-none focus:border-violet-500"
                  />
                </div>
                {stage1JobId && (
                  <div className="mt-2">
                    <img
                      src={getGifUrl(stage1JobId)}
                      alt="Stage 1 output"
                      className="w-full rounded-md border border-[var(--border)]"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Parameters Card */}
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h3 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-4">
              Parameters
            </h3>
            <div className="grid grid-cols-3 gap-3">
              <ParamInput label="Frames" value={numFrames} onChange={(v) => setNumFrames(parseInt(v) || 25)} min={1} max={25} />
              <ParamInput label="Steps" value={steps} onChange={(v) => setSteps(parseInt(v) || 30)} min={1} max={100} />
              <ParamInput label="Seed" value={seed} onChange={(v) => setSeed(parseInt(v) || 0)} />
              <ParamInput label="Min Guidance" value={minGuidance} onChange={(v) => setMinGuidance(parseFloat(v) || 0)} step={0.5} />
              <ParamInput label="Max Guidance" value={maxGuidance} onChange={(v) => setMaxGuidance(parseFloat(v) || 0)} step={0.5} />
              <ParamInput label="Noise Aug" value={noiseAug} onChange={(v) => setNoiseAug(parseFloat(v) || 0)} step={0.01} />
              {stage === 1 && (
                <ParamInput label="Cond Frames" value={condFrames} onChange={(v) => setCondFrames(parseInt(v) || 1)} min={1} max={5} />
              )}
              {stage === 2 && (
                <ParamInput label="Ctrl Scale" value={condScale} onChange={(v) => setCondScale(parseFloat(v) || 0)} step={0.1} />
              )}
            </div>

            <div className="mt-4">
              <label className="block text-[11px] font-medium text-zinc-500 mb-1.5 uppercase tracking-wider">
                Prompt
              </label>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-sm text-zinc-200 font-mono focus:outline-none focus:border-violet-500 transition-colors"
              />
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className={clsx(
                "w-full mt-5 py-3 rounded-lg text-sm font-semibold transition-all",
                isGenerating
                  ? "bg-violet-600/50 text-violet-200 cursor-not-allowed"
                  : "bg-violet-600 text-white hover:bg-violet-500 hover:shadow-lg hover:shadow-violet-500/20 active:scale-[0.98]"
              )}
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Generating...
                </span>
              ) : stage === 1 ? (
                "Generate Semantic Controls"
              ) : (
                "Generate RGB Video"
              )}
            </button>

            {/* Load existing job result */}
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                placeholder="Load existing job ID (e.g. 4c9acb96)"
                className="flex-1 bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-xs font-mono text-zinc-400 focus:outline-none focus:border-violet-500"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    const id = (e.target as HTMLInputElement).value.trim();
                    if (id) {
                      setJobId(id);
                      loadJobResult(id);
                      addLog(`Loading job ${id}...`);
                    }
                  }
                }}
              />
              <button
                onClick={(e) => {
                  const input = (e.currentTarget.previousElementSibling as HTMLInputElement);
                  const id = input?.value.trim();
                  if (id) {
                    setJobId(id);
                    loadJobResult(id);
                    addLog(`Loading job ${id}...`);
                  }
                }}
                className="px-3 py-2 text-xs font-medium bg-[var(--bg-input)] border border-[var(--border)] rounded-md text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
              >
                Load
              </button>
            </div>
          </div>
        </div>

        {/* ---- Right: Output ---- */}
        <div className="space-y-5">
          {/* Log Viewer */}
          <LogViewer logs={logs} status={status} />

          {/* Results */}
          {result && result.status !== "error" && (
            <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 animate-fade-in space-y-5">
              {/* Header */}
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-200">
                  {stage === 1 ? "Generated Semantic Frames" : "Generated RGB Video"}
                </h3>
                <div className="flex gap-2">
                  {jobId && (
                    <>
                      <span className="px-2 py-1 text-[10px] font-mono text-zinc-500 bg-[var(--bg-secondary)] rounded">
                        {jobId}
                      </span>
                      <a
                        href={getGifUrl(jobId)}
                        target="_blank"
                        className="px-3 py-1.5 text-[11px] font-semibold bg-[var(--bg-input)] border border-[var(--border)] rounded-md text-zinc-400 hover:text-white hover:border-violet-500 transition-colors"
                      >
                        GIF
                      </a>
                      <a
                        href={getDownloadUrl(jobId)}
                        target="_blank"
                        className="px-3 py-1.5 text-[11px] font-semibold bg-[var(--bg-input)] border border-[var(--border)] rounded-md text-zinc-400 hover:text-white hover:border-violet-500 transition-colors"
                      >
                        ZIP
                      </a>
                    </>
                  )}
                </div>
              </div>

              {/* GIF */}
              {jobId && result.has_gif && (
                <div className="flex justify-center">
                  <img
                    src={getGifUrl(jobId)}
                    alt="Generated output"
                    className="max-w-full rounded-lg border border-[var(--border)]"
                  />
                </div>
              )}

              {/* Generated Frames */}
              {jobId && result.frame_dirs?.[genDir] && (
                <FrameGallery
                  jobId={jobId}
                  subdir={genDir}
                  files={result.frame_dirs[genDir]}
                  label={stage === 1 ? "Predicted Semantic" : "Generated RGB"}
                  containerRef={predScrollRef as RefObject<HTMLDivElement>}
                  onScroll={handlePredScroll}
                />
              )}

              {/* GT Frames */}
              {jobId && result.frame_dirs?.[gtDir] && (
                <div className="border-t border-[var(--border)] pt-4">
                  <FrameGallery
                    jobId={jobId}
                    subdir={gtDir}
                    files={result.frame_dirs[gtDir]}
                    label="Ground Truth"
                    containerRef={gtScrollRef as RefObject<HTMLDivElement>}
                    onScroll={handleGtScroll}
                  />
                </div>
              )}

              {/* Evaluation Section */}
              <div className="border-t border-[var(--border)] pt-4">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                    Evaluation
                  </h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => gtInputRef.current?.click()}
                      className="px-3 py-1.5 text-[11px] font-medium bg-[var(--bg-input)] border border-[var(--border)] rounded-md text-zinc-400 hover:text-zinc-200 transition-colors"
                    >
                      Upload GT
                    </button>
                    <input
                      ref={gtInputRef}
                      type="file"
                      multiple
                      accept="image/*"
                      className="hidden"
                      onChange={handleGTUpload}
                    />
                    <button
                      onClick={handleEvaluate}
                      disabled={isEvaluating}
                      className="px-4 py-1.5 text-[11px] font-semibold bg-violet-600 text-white rounded-md hover:bg-violet-500 transition-colors disabled:opacity-50 flex items-center gap-1.5"
                    >
                      {isEvaluating ? (
                        <>
                          <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                          </svg>
                          Evaluating...
                        </>
                      ) : "Run Evaluation"}
                    </button>
                  </div>
                </div>

                {/* Evaluation progress */}
                {isEvaluating && evalStep && (
                  <div className="flex items-center gap-3 px-3 py-2.5 bg-violet-500/5 border border-violet-500/20 rounded-lg animate-fade-in">
                    <svg className="animate-spin h-4 w-4 text-violet-400 flex-shrink-0" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                    </svg>
                    <div>
                      <p className="text-xs font-medium text-violet-300">{evalStep}</p>
                      <p className="text-[10px] text-zinc-500 mt-0.5">
                        {stage === 1
                          ? "Comparing predicted semantic IDs against ground truth · computing per-class IoU"
                          : "DRN semantic segmentation on generated RGB · computing SSIM / PSNR / LPIPS"}
                      </p>
                    </div>
                  </div>
                )}

                <MetricsDisplay metrics={result.eval_results} />

                {/* AI Analysis */}
                {(result.eval_results || analysis || isAnalysing) && (
                  <div className="pt-3 border-t border-[var(--border)]">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">AI Analysis</h4>
                      <button
                        onClick={handleAnalyse}
                        disabled={isAnalysing || !result.eval_results}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold bg-violet-600 text-white rounded-md hover:bg-violet-500 transition-colors disabled:opacity-50"
                      >
                        {isAnalysing ? (
                          <>
                            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                            </svg>
                            Analysing...
                          </>
                        ) : (
                          <>
                            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.346a3.75 3.75 0 01-5.794 0l-.347-.346z" />
                            </svg>
                            {analysis ? "Re-analyse" : "Analyse with AI"}
                          </>
                        )}
                      </button>
                    </div>
                    <AnalysisCard analysis={analysis} isLoading={isAnalysing} />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
