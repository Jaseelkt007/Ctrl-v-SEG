"""
Original Ctrl-V Evaluation: FID only
=====================================
Evaluates the original Ctrl-V (Semantic→RGB) baseline using pre-generated
frames already on disk.

Directory layout (per clip subdir):
  <data_dir>/<seq_N>/frame_XXXX.png     — generated RGB
  <data_dir>/<seq_N>/real_B_XXXX.png   — ground-truth RGB

Metric computed:
  - FID (Frechet Inception Distance, Inception-v3 via torch-fidelity)
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Collect paired frame paths from directory structure
# ---------------------------------------------------------------------------

def collect_frame_pairs(data_dir):
    """
    Scan data_dir for subdirs containing frame_*.png files.
    For each subdir, pair frame_XXXX.png with real_B_XXXX.png.
    Only frames where both files exist are kept.

    Returns:
        pairs : list of (fake_path, real_path) tuples
    """
    pairs = []

    for sub in sorted(os.listdir(data_dir)):
        sub_path = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        fake_files = sorted(
            f for f in os.listdir(sub_path)
            if f.startswith("frame_") and f.endswith(".png")
        )

        for fake_fname in fake_files:
            frame_id = fake_fname.replace("frame_", "").replace(".png", "")
            real_fname = f"real_B_{frame_id}.png"
            real_path = os.path.join(sub_path, real_fname)
            if os.path.isfile(real_path):
                pairs.append((
                    os.path.join(sub_path, fake_fname),
                    real_path,
                ))

    return pairs


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Original Ctrl-V FID evaluation (frame_* vs real_B_*, co-located)"
    )
    p.add_argument("--data_dir",    required=True,
                   help="Root directory containing sequence subdirs")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save eval_results.json and eval_summary.txt")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap number of frame pairs evaluated (default: all)")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="Batch size for torch-fidelity FID computation (default: 64)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Original Ctrl-V Evaluation: FID (Inception-v3)")
    print("=" * 70)

    # ---------------------------------------------------------- [1] Collect pairs
    print("\n[1/3] Scanning frame directories ...")
    pairs = collect_frame_pairs(args.data_dir)
    print(f"  Total paired frames found : {len(pairs)}")

    if not pairs:
        print("ERROR: no paired frames found. Check --data_dir.")
        sys.exit(1)

    if args.num_samples is not None:
        pairs = pairs[: args.num_samples]
        print(f"  Capped at               : {len(pairs)}")

    # Detect resolution from first fake frame
    sample_img = Image.open(pairs[0][0])
    fake_W, fake_H = sample_img.size
    print(f"  Generated resolution    : {fake_W}×{fake_H}  (W×H)")
    print(f"  GT resized to match     : {fake_W}×{fake_H}  (W×H)")

    # ---------------------------------------------------------- [2] Copy to temp dirs
    print(f"\n[2/3] Copying {len(pairs)} frame pairs to temp dirs ...")

    fid_temp    = os.path.join(args.output_dir, '_fid_tmp')
    fid_gen_dir = os.path.join(fid_temp, 'gen')
    fid_gt_dir  = os.path.join(fid_temp, 'gt')
    os.makedirs(fid_gen_dir, exist_ok=True)
    os.makedirs(fid_gt_dir,  exist_ok=True)

    for idx, (fake_path, real_path) in enumerate(tqdm(pairs, desc="Copying frames")):
        # Generated frame — copy as-is
        Image.open(fake_path).convert("RGB").save(
            os.path.join(fid_gen_dir, f'{idx:06d}.png'))

        # GT frame — resize to match generated resolution
        gt_img = Image.open(real_path).convert("RGB")
        if gt_img.size != (fake_W, fake_H):
            gt_img = gt_img.resize((fake_W, fake_H), Image.LANCZOS)
        gt_img.save(os.path.join(fid_gt_dir, f'{idx:06d}.png'))

    print(f"  Frames written to: {fid_temp}")

    # ---------------------------------------------------------- [3] FID
    print("\n[3/3] Computing FID (Inception-v3 via torch-fidelity) ...")
    fid_score = None
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=fid_gen_dir, input2=fid_gt_dir,
            cuda=True, isc=False, fid=True, kid=False, prc=False,
            verbose=False, batch_size=args.batch_size,
        )
        fid_score = metrics['frechet_inception_distance']
        print(f"  ✓ FID (Inception-v3): {fid_score:.4f}")
    except ImportError:
        print("  ERROR: torch-fidelity not installed. Run: pip install torch-fidelity")
    except Exception as e:
        print(f"  ERROR: FID computation failed: {e}")
    finally:
        shutil.rmtree(fid_temp, ignore_errors=True)
        print(f"  Temp dir cleaned up.")

    # ---------------------------------------------------------- Results
    n_evaluated = len(pairs)
    print("\n" + "=" * 70)
    print(f"Ctrl-V FID RESULTS  |  {n_evaluated} frame pairs")
    print("=" * 70)
    if fid_score is not None:
        print(f"  FID (Inception-v3): {fid_score:.4f}")
    else:
        print("  FID: FAILED")
    print("=" * 70)

    # -- Save JSON --
    json_out = {
        "model":            "Ctrl-V (original)",
        "data_dir":         args.data_dir,
        "num_frame_pairs":  n_evaluated,
        "resolution":       f"{fake_H}x{fake_W}",
        "fid":              fid_score,
    }
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n✓ JSON saved:     {json_path}")

    # -- Save text summary --
    summary_path = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Ctrl-V (Original) FID Evaluation Summary\n{'='*60}\n\n")
        f.write(f"Model           : Ctrl-V (original)\n")
        f.write(f"Data dir        : {args.data_dir}\n")
        f.write(f"Frames evaluated: {n_evaluated}\n")
        f.write(f"Resolution      : {fake_H}x{fake_W}\n\n")
        f.write("Image Quality:\n")
        if fid_score is not None:
            f.write(f"  FID (Inception-v3): {fid_score:.4f}\n")
        else:
            f.write("  FID: FAILED\n")
    print(f"✓ Summary saved:  {summary_path}")
    print("\n✓ Ctrl-V FID evaluation complete!")


if __name__ == "__main__":
    main()
