"use client";

interface ParamInputProps {
  label: string;
  value: number | string;
  onChange: (val: string) => void;
  type?: "number" | "text";
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  className?: string;
}

export default function ParamInput({
  label,
  value,
  onChange,
  type = "number",
  step,
  min,
  max,
  disabled,
  className = "",
}: ParamInputProps) {
  return (
    <div className={className}>
      <label className="block text-[11px] font-medium text-zinc-500 mb-1.5 uppercase tracking-wider">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-md px-3 py-2 text-sm font-mono text-zinc-200 focus:outline-none focus:border-violet-500 transition-colors disabled:opacity-50"
      />
    </div>
  );
}
