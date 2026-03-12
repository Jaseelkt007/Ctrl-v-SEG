"use client";

import { clsx } from "clsx";

export type StepState = "pending" | "active" | "done" | "error";

export interface PipelineStep {
  label: string;
  sublabel?: string;
  state: StepState;
}

// ─── Architecture phase definitions ──────────────────────────────────────────

interface Phase {
  label: string;
  components: string[];    // boxes shown inside the phase
  dataIn: string;          // data entering this phase
  dataOut: string;         // data leaving this phase
}

const STAGE1_PHASES: Phase[] = [
  {
    label: "Load Input",
    components: ["RGB Frame", "GT Sem IDs"],
    dataIn: "KITTI-360 clip",
    dataOut: "192×704×3\n+ 25×192×704",
  },
  {
    label: "Dual Encoding",
    components: ["CLIP Encoder", "Sem VAE Enc"],
    dataIn: "RGB + Sem IDs",
    dataOut: "768d embed\n+ 25×4×24×88",
  },
  {
    label: "SVD UNet",
    components: ["Noise Scheduler", "UNet Denoiser"],
    dataIn: "Conditioned latents",
    dataOut: "Pred latents\n25×4×24×88",
  },
  {
    label: "Sem VAE Decode",
    components: ["Semantic VAE Dec", "Argmax (19 cls)"],
    dataIn: "25×4×24×88",
    dataOut: "Sem IDs\n25×192×704",
  },
  {
    label: "Colorize & Save",
    components: ["Label → Color", "GIF encoder"],
    dataIn: "Sem class IDs",
    dataOut: "25 sem frames\n+ output.gif",
  },
];

const STAGE2_PHASES: Phase[] = [
  {
    label: "Load Controls",
    components: ["Sem Maps", "RGB Frame"],
    dataIn: "Stage 1 output",
    dataOut: "25×192×704\n+ 192×704×3",
  },
  {
    label: "Dual Encoding",
    components: ["Sem VAE Enc", "CLIP Encoder"],
    dataIn: "Sem maps + RGB",
    dataOut: "Ctrl 25×4×24×88\n+ 768d embed",
  },
  {
    label: "ControlNet + UNet",
    components: ["ControlNet", "SVD UNet"],
    dataIn: "Semantic ctrl latents",
    dataOut: "RGB latents\n25×4×24×88",
  },
  {
    label: "RGB VAE Decode",
    components: ["AutoencoderKL", "Temporal Dec"],
    dataIn: "25×4×24×88",
    dataOut: "RGB frames\n25×3×192×704",
  },
  {
    label: "Render & Save",
    components: ["Frame writer", "GIF encoder"],
    dataIn: "RGB frame tensors",
    dataOut: "25 RGB frames\n+ output.gif",
  },
];

// ─── Phase card ──────────────────────────────────────────────────────────────

function PhaseCard({
  phase,
  index,
  state,
  diffusionStep,
  diffusionTotal,
}: {
  phase: Phase;
  index: number;
  state: StepState;
  diffusionStep?: number;
  diffusionTotal?: number;
}) {
  const isDiffusion = index === 2;

  return (
    <div
      className={clsx(
        "relative flex-1 min-w-0 rounded-lg border p-3 transition-all duration-500",
        state === "active" && "border-violet-500/70 bg-violet-500/5 shadow-lg shadow-violet-500/10",
        state === "done"   && "border-emerald-500/40 bg-emerald-500/5",
        state === "error"  && "border-red-500/40 bg-red-500/5",
        state === "pending"&& "border-[var(--border)] bg-[var(--bg-card)] opacity-50",
      )}
    >
      {/* Step number + label */}
      <div className="flex items-center gap-1.5 mb-2">
        <span
          className={clsx(
            "inline-flex items-center justify-center w-4 h-4 rounded-full text-[9px] font-bold flex-shrink-0",
            state === "active"  && "bg-violet-500 text-white",
            state === "done"    && "bg-emerald-500 text-white",
            state === "error"   && "bg-red-500 text-white",
            state === "pending" && "bg-zinc-700 text-zinc-400",
          )}
        >
          {state === "done" ? (
            <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          ) : state === "active" ? (
            <svg className="w-2.5 h-2.5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-30" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            index + 1
          )}
        </span>
        <span
          className={clsx(
            "text-[10px] font-semibold truncate",
            state === "active"  && "text-violet-300",
            state === "done"    && "text-emerald-400",
            state === "error"   && "text-red-400",
            state === "pending" && "text-zinc-600",
          )}
        >
          {phase.label}
        </span>
      </div>

      {/* Component chips */}
      <div className="flex flex-col gap-1 mb-2">
        {phase.components.map((c) => (
          <div
            key={c}
            className={clsx(
              "px-1.5 py-0.5 rounded text-[9px] font-mono border text-center truncate",
              state === "active"  && "bg-violet-500/15 border-violet-500/30 text-violet-300",
              state === "done"    && "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
              state === "error"   && "bg-red-500/10 border-red-500/20 text-red-400",
              state === "pending" && "bg-zinc-800 border-zinc-700 text-zinc-600",
            )}
          >
            {c}
          </div>
        ))}
      </div>

      {/* Data-out shape */}
      <div
        className={clsx(
          "px-1.5 py-1 rounded text-[9px] font-mono whitespace-pre-line border-l-2 pl-2",
          state === "active"  && "border-violet-500 text-zinc-400",
          state === "done"    && "border-emerald-500/50 text-zinc-500",
          state === "pending" && "border-zinc-700 text-zinc-700",
          state === "error"   && "border-red-500 text-zinc-500",
        )}
      >
        {phase.dataOut}
      </div>

      {/* Diffusion progress bar inside UNet card */}
      {isDiffusion && state === "active" && diffusionTotal && diffusionTotal > 0 && (
        <div className="mt-2 pt-2 border-t border-violet-500/20">
          <div className="flex justify-between text-[9px] mb-1">
            <span className="text-zinc-500">denoising</span>
            <span className="text-violet-400 font-mono">{diffusionStep ?? 0}/{diffusionTotal}</span>
          </div>
          <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-600 to-violet-400 rounded-full transition-all duration-500"
              style={{ width: `${Math.round(((diffusionStep ?? 0) / diffusionTotal) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Pulse ring when active */}
      {state === "active" && (
        <span className="absolute inset-0 rounded-lg border border-violet-500/30 animate-ping pointer-events-none" />
      )}
    </div>
  );
}

// ─── Connector arrow ─────────────────────────────────────────────────────────

function Arrow({ lit }: { lit: boolean }) {
  return (
    <div className="flex-shrink-0 flex items-center justify-center w-5 mt-[-8px]">
      <svg
        className={clsx("w-4 h-4 transition-colors duration-500", lit ? "text-violet-500" : "text-zinc-700")}
        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export default function PipelineFlow({
  stage,
  steps,
  diffusionStep,
  diffusionTotal,
}: {
  stage: 1 | 2;
  steps: PipelineStep[];      // still used for state; one per phase
  diffusionStep?: number;
  diffusionTotal?: number;
}) {
  const phases = stage === 1 ? STAGE1_PHASES : STAGE2_PHASES;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 mb-6">
      {/* Stage label */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-600">
          {stage === 1 ? "Stage 1 · RGB → Semantic Pipeline" : "Stage 2 · Semantic → RGB Pipeline"}
        </span>
        {steps.some((s) => s.state === "active") && (
          <span className="inline-flex items-center gap-1 text-[10px] text-violet-400 font-medium">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
            running
          </span>
        )}
        {steps.every((s) => s.state === "done") && steps.length > 0 && (
          <span className="text-[10px] text-emerald-400 font-medium">✓ complete</span>
        )}
      </div>

      {/* Phase cards + arrows */}
      <div className="flex items-start gap-0">
        {phases.map((phase, i) => (
          <div key={i} className="flex items-start flex-1 min-w-0">
            <PhaseCard
              phase={phase}
              index={i}
              state={steps[i]?.state ?? "pending"}
              diffusionStep={diffusionStep}
              diffusionTotal={diffusionTotal}
            />
            {i < phases.length - 1 && (
              <Arrow lit={
                (steps[i]?.state === "done") ||
                (steps[i + 1]?.state === "active" || steps[i + 1]?.state === "done")
              } />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
