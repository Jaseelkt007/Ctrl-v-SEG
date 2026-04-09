"""
SVS-GAN Evaluation: FVD-I3D only
=================================
Evaluates SVS-GAN baseline using pre-generated frames already on disk.

Directory layout (per clip subdir):
  <svs_dir>/<seq_N>/fake_B_XXXXXXXXXX.jpg   — generated RGB
  <svs_dir>/<seq_N>/real_B_XXXXXXXXXX.png   — ground-truth RGB (already copied)

No external dataset paths needed — GT is co-located with predictions.

Metric computed:
  - FVD-I3D  (cdfvd / I3D)
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# ---------------------------------------------------------------------------
# FVD frame sampling
# ---------------------------------------------------------------------------

def sample_video_frames(video, target_frames=16):
    """video: (T, H, W, C) uint8 → (target_frames, H, W, C) uint8"""
    T = video.shape[0]
    if T >= target_frames:
        idxs = np.linspace(0, T - 1, target_frames, dtype=int)
        return video[idxs]
    pad = [video[-1]] * (target_frames - T)
    return np.concatenate([video, np.stack(pad)], axis=0)


# ---------------------------------------------------------------------------
# Build clip list directly from directory structure
# ---------------------------------------------------------------------------

def build_clip_list(svs_base, clip_length=25):
    """
    Scan svs_base for subdirs containing fake_B_*.jpg files.
    For each subdir, pair fake_B_<frame>.jpg with real_B_<frame>.png.
    Only frames where both files exist are kept.
    Non-overlapping clips of `clip_length` are formed per subdir.

    Returns:
        clips   : list of dicts with keys seq, start_frame, fake_paths, real_paths
        skipped : number of incomplete clips (frames missing real_B counterpart)
        total_possible : total clips before filtering
    """
    clips = []
    skipped = 0
    total_possible = 0

    for sub in sorted(os.listdir(svs_base)):
        sub_path = os.path.join(svs_base, sub)
        if not os.path.isdir(sub_path) or not sub.startswith("2013"):
            continue

        # Collect all fake_B frames and check for matching real_B
        fake_files = sorted(
            f for f in os.listdir(sub_path)
            if f.startswith("fake_B_") and f.endswith(".jpg")
        )

        paired = []
        for fake_fname in fake_files:
            frame_id = fake_fname.replace("fake_B_", "").replace(".jpg", "")
            real_fname = f"real_B_{frame_id}.png"
            real_path = os.path.join(sub_path, real_fname)
            if os.path.isfile(real_path):
                paired.append((frame_id,
                               os.path.join(sub_path, fake_fname),
                               real_path))

        n_clips = len(paired) // clip_length
        total_possible += n_clips

        for clip_i in range(n_clips):
            entries = paired[clip_i * clip_length: (clip_i + 1) * clip_length]
            clips.append({
                "seq":         sub,
                "start_frame": entries[0][0],
                "fake_paths":  [e[1] for e in entries],
                "real_paths":  [e[2] for e in entries],
            })

    return clips, skipped, total_possible


# ---------------------------------------------------------------------------
# Load frames from disk
# ---------------------------------------------------------------------------

def load_clip_frames(paths, target_hw=None):
    """Load a list of image paths → (T, H, W, 3) uint8."""
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if target_hw is not None:
            img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
        frames.append(np.array(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SVS-GAN FVD-I3D evaluation (fake_B vs real_B, co-located)"
    )
    p.add_argument("--svs_dir",     required=True,
                   help="Root of val_latest directory containing seq subdirs")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save eval_results.json and eval_summary.txt")
    p.add_argument("--clip_length", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap number of clips evaluated (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SVS-GAN Evaluation: FVD-I3D")
    print("=" * 70)

    # ---------------------------------------------------------- [1] Clip list
    print("\n[1/3] Scanning clip directories ...")
    clips, skipped, total_possible = build_clip_list(args.svs_dir, args.clip_length)
    print(f"  Clip dirs scanned  : {total_possible + skipped}")
    print(f"  Complete clips     : {len(clips)}")

    if args.num_samples is not None:
        clips = clips[: args.num_samples]
        print(f"  Capped at          : {len(clips)}")

    if not clips:
        print("ERROR: no complete clips found. Check --svs_dir.")
        sys.exit(1)

    # Detect resolution from first fake_B frame
    sample_img = Image.open(clips[0]["fake_paths"][0])
    fake_W, fake_H = sample_img.size
    target_hw = (fake_H, fake_W)
    print(f"  fake_B resolution  : {fake_W}×{fake_H}  (W×H)")
    print(f"  real_B resized to  : {fake_H}×{fake_W}  (H×W)")
    print(f"  clip_length        : {args.clip_length}  frames per clip")

    # ---------------------------------------------------------- [2] Accumulate
    print(f"\n[2/3] Loading {len(clips)} clips ...")

    all_real_clips = []
    all_fake_clips = []

    for clip in tqdm(clips, desc="Loading clips"):
        fake_hwc = load_clip_frames(clip["fake_paths"])
        real_hwc = load_clip_frames(clip["real_paths"], target_hw=target_hw)
        all_fake_clips.append(sample_video_frames(fake_hwc))
        all_real_clips.append(sample_video_frames(real_hwc))

    real_fvd_np = np.stack(all_real_clips, axis=0)   # (N, 16, H, W, 3)
    fake_fvd_np = np.stack(all_fake_clips, axis=0)
    print(f"  FVD array shape: {fake_fvd_np.shape}  dtype={fake_fvd_np.dtype}")

    del all_real_clips, all_fake_clips

    # ---------------------------------------------------------- [3] FVD-I3D
    print("\n[3/3] Computing FVD-I3D (cdfvd / I3D) ...")
    fvd_i3d = None
    try:
        from cdfvd import fvd as cdfvd_lib
        ev = cdfvd_lib.cdfvd(model="i3d", device="cuda")
        fvd_i3d = ev.compute_fvd(real_fvd_np, fake_fvd_np)
        print(f"  ✓ FVD-I3D: {fvd_i3d:.4f}")
    except Exception as e:
        print(f"  ERROR: FVD-I3D failed: {e}")

    del real_fvd_np, fake_fvd_np
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- Results
    n_evaluated = len(clips)
    print("\n" + "=" * 70)
    print(f"SVS-GAN FVD RESULTS  |  {n_evaluated} clips")
    print("=" * 70)
    if fvd_i3d is not None:
        print(f"  FVD-I3D (cdfvd/I3D): {fvd_i3d:.4f}")
    else:
        print("  FVD-I3D: FAILED")
    print("=" * 70)

    # -- Save JSON --
    json_out = {
        "model":           "SVS-GAN",
        "svs_dir":         args.svs_dir,
        "num_clips":       n_evaluated,
        "clip_length":     args.clip_length,
        "fake_resolution": f"{fake_H}x{fake_W}",
        "fvd_i3d":         fvd_i3d,
    }
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n✓ JSON saved:     {json_path}")

    # -- Save text summary --
    summary_path = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"SVS-GAN FVD Evaluation Summary\n{'='*60}\n\n")
        f.write(f"Model           : SVS-GAN\n")
        f.write(f"SVS output dir  : {args.svs_dir}\n")
        f.write(f"Clips evaluated : {n_evaluated}\n")
        f.write(f"Clip length     : {args.clip_length}\n")
        f.write(f"Resolution      : {fake_H}x{fake_W}\n\n")
        f.write("Video Quality:\n")
        if fvd_i3d is not None:
            f.write(f"  FVD-I3D (cdfvd/I3D): {fvd_i3d:.4f}\n")
        else:
            f.write("  FVD-I3D: FAILED\n")
    print(f"✓ Summary saved:  {summary_path}")
    print("\n✓ SVS-GAN FVD evaluation complete!")


if __name__ == "__main__":
    main()
