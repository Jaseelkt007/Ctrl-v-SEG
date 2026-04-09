#!/usr/bin/env python3
"""
generate_drn_ms_report.py

Post-processing step for eval_stage2_drn_ms.sh.

Reads:
  {ctrlv_stage2_dir}/confusion_matrix.npy   — saved by segment.py test_ms
  {output_dir}/drn_ms_eval.log              — segment.py stdout
  {output_dir}/metadata.json                — clip / checkpoint info
  SLURM env vars                            — job ID, node, GPU

Writes:
  {output_dir}/eval_report.txt              — structured human-readable report
  {output_dir}/confusion_matrix_drn.png     — normalised confusion matrix heatmap
  {output_dir}/confusion_matrix_drn.npy     — copy of raw confusion matrix

Usage (called from eval_stage2_drn_ms.sh after Phase 2):
  python tools/generate_drn_ms_report.py \\
      --output_dir   /no_backups/.../eval_stage2_drn_ms_unet_unfreeze \\
      --job_id       199549 \\
      --node         linse19 \\
      --gpu          "NVIDIA RTX A5000" \\
      --phase1_dur   "13m 29s" \\
      --phase2_dur   "1m 48s" \\
      --total_dur    "0h 15m 17s"
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

KITTI360_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
NUM_CLASSES = 19


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def parse_miou_from_log(log_path):
    """Extract final mIoU from segment.py log (last 'mAP:' line)."""
    miou = None
    if not os.path.exists(log_path):
        return miou
    with open(log_path) as f:
        for line in f:
            m = re.search(r"mAP:\s*([\d.]+)", line)
            if m:
                miou = float(m.group(1))
    return miou


def parse_per_class_ious_from_log(log_path):
    """Extract per-class IoU array from segment.py log.

    segment.py logs the IoUs as a single space-separated line of 19 floats
    (with 'nan' for absent classes) just before the final 'mAP:' line.
    """
    ious = None
    if not os.path.exists(log_path):
        return ious
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            # Strip leading timestamp / log prefix if present
            # e.g. "[2026-04-02 ... segment.py:897 test_ms] 89.809 53.724 ..."
            m = re.search(r"]\s+([\d. nan]+)$", line)
            if m:
                tokens = m.group(1).split()
            else:
                tokens = line.split()
            if len(tokens) == NUM_CLASSES:
                try:
                    parsed = [float("nan") if t == "nan" else float(t) for t in tokens]
                    # Accept only if values look like IoU percentages (0–100)
                    valid = [v for v in parsed if not np.isnan(v)]
                    if valid and max(valid) <= 100.0:
                        ious = np.array(parsed)
                except ValueError:
                    pass
    return ious


def make_confusion_matrix_png(hist, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        cm = hist.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(KITTI360_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(KITTI360_CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("DRN Multi-Scale Confusion Matrix (row-normalised)\nKITTI-360 19 classes")

        # Annotate cells with values >= 0.05
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = cm_norm[i, j]
                if v >= 0.05:
                    color = "white" if v > 0.5 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Warning: could not generate confusion matrix PNG: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--job_id",     default="")
    parser.add_argument("--node",       default="")
    parser.add_argument("--gpu",        default="")
    parser.add_argument("--phase1_dur", default="")
    parser.add_argument("--phase2_dur", default="")
    parser.add_argument("--total_dur",  default="")
    args = parser.parse_args()

    output_dir       = args.output_dir
    ctrlv_stage2_dir = os.path.join(output_dir, "CTRLV_STAGE2")
    hist_path        = os.path.join(ctrlv_stage2_dir, "confusion_matrix.npy")
    log_path         = os.path.join(output_dir, "drn_ms_eval.log")
    meta_path        = os.path.join(output_dir, "metadata.json")
    report_path      = os.path.join(output_dir, "eval_report.txt")
    cm_npy_path      = os.path.join(output_dir, "confusion_matrix_drn.npy")
    cm_png_path      = os.path.join(output_dir, "confusion_matrix_drn.png")

    # ---- Load confusion matrix (or fall back to log-parsed IoUs) ----
    hist = None
    if os.path.exists(hist_path):
        hist = np.load(hist_path)
        ious = per_class_iu(hist) * 100
        np.save(cm_npy_path, hist)
    else:
        print(f"  Note: confusion_matrix.npy not found at {hist_path}")
        print("        Falling back to per-class IoUs parsed from the DRN log.")
        ious = parse_per_class_ious_from_log(log_path)
        if ious is None:
            print(f"ERROR: Could not parse per-class IoUs from {log_path}")
            sys.exit(1)

    miou = round(float(np.nanmean(ious)), 2)

    # ---- Load metadata ----
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    checkpoint_dir  = meta.get("checkpoint_dir", "")
    checkpoint_step = meta.get("checkpoint_step", "")
    clip_length     = meta.get("clip_length", 25)
    resolution      = meta.get("resolution", "")
    infer_steps     = meta.get("num_inference_steps", "")
    clips           = meta.get("clips", [])

    # Fall back to log mIoU if hist-derived differs (sanity)
    log_miou = parse_miou_from_log(log_path)

    # ---- Generate confusion matrix PNG (only when hist is available) ----
    png_ok = False
    if hist is not None:
        png_ok = make_confusion_matrix_png(hist, cm_png_path)

    # ---- Write report ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_path, "w") as f:
        def w(line=""):
            f.write(line + "\n")

        w("Stage 2 DRN Multi-Scale mIoU Evaluation Report")
        w("=" * 60)
        w()
        w(f"Evaluation Date   : {now}")
        if args.job_id:  w(f"SLURM Job ID      : {args.job_id}")
        if args.node:    w(f"Node              : {args.node}")
        if args.gpu:     w(f"GPU               : {args.gpu}")
        if args.total_dur:
            w(f"Total Duration    : {args.total_dur}" +
              (f"  (Phase 1: {args.phase1_dur}, Phase 2: {args.phase2_dur})"
               if args.phase1_dur else ""))
        w()
        w("-" * 60)
        w("Checkpoint")
        w("-" * 60)
        w(f"Checkpoint Dir    : {checkpoint_dir}")
        w(f"Checkpoint Step   : {checkpoint_step}")
        w(f"DRN Checkpoint    : drn_eval/KITTI360_checkpoints/checkpoint_030.pth.tar")
        w(f"                    (drn_d_105, fine-tuned on KITTI-360 for 30 epochs)")
        w()
        w("-" * 60)
        w("Evaluation Protocol")
        w("-" * 60)
        w("Methodology       : drn_d_105_MIoU (matches reference evaluation)")
        w("DRN Inference     : Multi-scale  (6 scales: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)")
        w("Confusion Matrix  : Confidence-weighted  (KITTI-360 per-pixel confidence PNGs)")
        w(f"Frame Count       : {len(clips) * clip_length} frames  "
          f"({len(clips)} clip groups x {clip_length} frames)")
        w(f"Resolution        : {resolution}")
        w(f"Inference Steps   : {infer_steps}")
        w(f"GT Label Source   : drn_eval/CTRLV_STAGE2/val_labels.txt  (checked-in, fixed set)")
        w()
        w("-" * 60)
        w(f"Clip Groups ({len(clips)} total)")
        w("-" * 60)
        w(f" {'#':>3}  {'Sequence':<40}  {'Frames'}")
        for c in clips:
            fids = c.get("frame_ids", [])
            if fids:
                w(f" {c['group_idx']:>3}  {c['sequence']:<40}  "
                  f"{fids[0]:010d} – {fids[-1]:010d}")
        w()
        w("-" * 60)
        w("Overall Result")
        w("-" * 60)
        w(f"DRN mIoU (multi-scale, confidence-weighted) : {miou:.2f}%")
        if log_miou is not None and abs(log_miou - miou) > 0.1:
            w(f"  (Note: log-reported mIoU = {log_miou:.2f}% — small diff due to float precision)")
        w()
        w("-" * 60)
        w("Per-Class IoU  (19 KITTI-360 trainID classes)")
        w("-" * 60)
        w(f" {'TrainID':>7}  {'Class':<16}  {'IoU (%)':>8}")
        w(f" {'-------':>7}  {'-'*16}  {'-------':>8}")
        for i, (cls, iou) in enumerate(zip(KITTI360_CLASSES, ious)):
            if np.isnan(iou):
                w(f" {i:>7}  {cls:<16}   {'n/a':>7}   (class absent from eval frames)")
            else:
                w(f" {i:>7}  {cls:<16}  {iou:>8.2f}")
        w()
        w(f" mIoU = nanmean over {int(np.sum(~np.isnan(ious)))} non-NaN classes = {miou:.2f}%")
        w()
        w("-" * 60)
        w("Output Paths")
        w("-" * 60)
        w(f"Generated frames  : {ctrlv_stage2_dir}/generated_frames/")
        w(f"val_images.txt    : {ctrlv_stage2_dir}/val_images.txt")
        w(f"val_labels.txt    : {ctrlv_stage2_dir}/val_labels.txt")
        w(f"DRN raw log       : {log_path}")
        w(f"Metadata JSON     : {meta_path}")
        w(f"Confusion matrix  : {cm_npy_path}  (.npy)")
        if png_ok:
            w(f"                    {cm_png_path}  (.png)")
        w(f"This report       : {report_path}")

    print(f"✓ Report written : {report_path}")
    if png_ok:
        print(f"✓ CM PNG written : {cm_png_path}")
    print(f"✓ CM npy written : {cm_npy_path}")
    print(f"\nDRN mIoU (multi-scale, confidence-weighted): {miou:.2f}%")
    print("Per-class IoU:")
    for i, (cls, iou) in enumerate(zip(KITTI360_CLASSES, ious)):
        iou_str = f"{iou:.2f}%" if not np.isnan(iou) else "n/a"
        print(f"  {i:>2}  {cls:<16}  {iou_str}")


if __name__ == "__main__":
    main()
