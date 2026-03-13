#!/usr/bin/env python3
"""
Compute FID, FVD-I3D, FVD-VideoMAE, LPIPS, SSIM, and PSNR for Stage 2 generated RGB frames.

Reads from the saved frame directories produced by eval_stage2_rgb.py:
  frames/video_XXX/gt_rgb/frame_YYY.png
  frames/video_XXX/gen_rgb/frame_YYY.png

Metrics:
  - FID            : Frechet Inception Distance (torch-fidelity, Inception-v3 features)
  - FVD-I3D        : Frechet Video Distance using cdfvd I3D backbone (i3d_pretrained_400.pt)
  - FVD-VideoMAE   : Frechet Video Distance using VideoMAE-Base (MCG-NJU/videomae-base, ViT-B/16)
  - LPIPS          : Learned Perceptual Image Patch Similarity
  - SSIM           : Structural Similarity Index
  - PSNR           : Peak Signal-to-Noise Ratio

FVD-I3D and FVD-VideoMAE are evaluated SEPARATELY and logged with their own sections.

Usage:
    python tools/compute_stage2_fid_fvd.py \\
        --frames_dir /no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/frames \\
        --output_file /no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/fid_fvd_results.txt
"""

import os
import sys
import argparse
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def sample_video_frames(video, target_frames=16):
    """Uniformly sample target_frames from a (T, H, W, C) uint8 video array."""
    T = video.shape[0]
    if T >= target_frames:
        idxs = np.linspace(0, T - 1, target_frames, dtype=int)
        return video[idxs]
    else:
        pad = [video[-1]] * (target_frames - T)
        return np.concatenate([video, np.stack(pad)], axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute FID, FVD-I3D, FVD-C3D, LPIPS, SSIM, PSNR for Stage 2 frames'
    )
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video_XXX/ subdirs with gt_rgb/ and gen_rgb/')
    parser.add_argument('--num_frames', type=int, default=25,
                        help='Number of frames per video (default: 25)')
    parser.add_argument('--fid_batch_size', type=int, default=64,
                        help='Batch size for FID feature extraction')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output .txt file (default: fid_fvd_results.txt beside frames_dir)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Output .json file (default: fid_fvd_results.json beside frames_dir)')
    parser.add_argument('--skip_fvd_i3d', action='store_true',
                        help='Skip FVD-I3D (requires Dropbox download of i3d_torchscript.pt)')
    parser.add_argument('--skip_fvd_videomae', action='store_true',
                        help='Skip FVD-C3D (requires torchvision with Kinetics400 weights)')
    return parser.parse_args()


def load_video_frames(video_dir, subdir, num_frames):
    """Load frames/frame_XXX.png → [T, H, W, 3] uint8."""
    frame_files = sorted(
        f for f in os.listdir(os.path.join(video_dir, subdir)) if f.endswith('.png')
    )[:num_frames]
    frames = [np.array(Image.open(os.path.join(video_dir, subdir, ff)).convert('RGB'))
              for ff in frame_files]
    return np.stack(frames, axis=0)


def check_dependencies():
    """Check & report which optional packages are available."""
    status = {}
    try:
        import torch_fidelity; status['torch_fidelity'] = True
    except ImportError:
        status['torch_fidelity'] = False
        print("  [MISSING] torch-fidelity — install with: pip install torch-fidelity")

    try:
        from cdfvd import fvd; status['fvd_i3d'] = True
    except Exception as e:
        status['fvd_i3d'] = False
        print(f"  [MISSING] FVD-I3D (cdfvd): {e}")

    try:
        from cdfvd import fvd; status['fvd_videomae'] = True
    except Exception as e:
        status['fvd_videomae'] = False
        print(f"  [MISSING] FVD-VideoMAE (cdfvd): {e}")

    try:
        import lpips; status['lpips'] = True
    except ImportError:
        status['lpips'] = False
        print("  [MISSING] lpips — install with: pip install lpips")

    try:
        from skimage.metrics import structural_similarity; status['skimage'] = True
    except ImportError:
        status['skimage'] = False
        print("  [MISSING] scikit-image — install with: pip install scikit-image")

    return status


def main():
    args = parse_args()

    frames_dir = args.frames_dir
    parent_dir = os.path.dirname(frames_dir.rstrip('/'))

    if args.output_file is None:
        args.output_file = os.path.join(parent_dir, 'fid_fvd_results.txt')
    if args.output_json is None:
        args.output_json = os.path.join(parent_dir, 'fid_fvd_results.json')

    # Discover video directories
    video_dirs = sorted(
        d for d in os.listdir(frames_dir)
        if d.startswith('video_') and os.path.isdir(os.path.join(frames_dir, d))
    )
    if not video_dirs:
        print(f"Error: No video_XXX directories found in {frames_dir}")
        sys.exit(1)

    num_videos = len(video_dirs)
    print("=" * 70)
    print("Stage 2: FID / FVD-I3D / FVD-VideoMAE / LPIPS / SSIM / PSNR Evaluation")
    print("=" * 70)
    print(f"Frames directory : {frames_dir}")
    print(f"Videos found     : {num_videos}")
    print(f"Frames per video : {args.num_frames}")
    print()

    # ------------------------------------------------------------------ deps --
    print("[0/7] Checking dependencies...")
    deps = check_dependencies()
    print()

    # ------------------------------------------------------------------ load --
    print("[1/7] Loading video frames...")
    sample_frame = Image.open(
        os.path.join(frames_dir, video_dirs[0], 'gen_rgb', 'frame_000.png'))
    W, H = sample_frame.size
    print(f"  Frame resolution: {H}x{W}")

    all_gt_videos, all_gen_videos = [], []
    for vd in tqdm(video_dirs, desc="  Loading"):
        vpath = os.path.join(frames_dir, vd)
        all_gt_videos.append(load_video_frames(vpath, 'gt_rgb', args.num_frames))
        all_gen_videos.append(load_video_frames(vpath, 'gen_rgb', args.num_frames))
    print(f"  ✓ Loaded {num_videos} videos × {args.num_frames} frames")

    # ------------------------------------------------------------------ FID --
    print("\n[2/7] Computing FID (Inception-v3 via torch-fidelity)...")
    fid_score = None
    if deps['torch_fidelity']:
        try:
            from torch_fidelity import calculate_metrics
            temp_dir = tempfile.mkdtemp(prefix='stage2_fid_')
            gt_dir  = os.path.join(temp_dir, 'gt');  os.makedirs(gt_dir)
            gen_dir = os.path.join(temp_dir, 'gen'); os.makedirs(gen_dir)
            idx = 0
            for gt_vid, gen_vid in zip(all_gt_videos, all_gen_videos):
                for t in range(gt_vid.shape[0]):
                    Image.fromarray(gt_vid[t]).save(os.path.join(gt_dir,  f'{idx:06d}.png'))
                    Image.fromarray(gen_vid[t]).save(os.path.join(gen_dir, f'{idx:06d}.png'))
                    idx += 1
            print(f"  Total frame pairs for FID: {idx}")
            # cuda=True is fine in a standalone script (no ThreadPoolExecutor conflict)
            metrics = calculate_metrics(
                input1=gen_dir, input2=gt_dir,
                cuda=True, isc=False, fid=True, kid=False, prc=False,
                verbose=True, batch_size=args.fid_batch_size,
            )
            fid_score = metrics['frechet_inception_distance']
            print(f"  ✓ FID (Inception-v3): {fid_score:.4f}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"  WARNING: FID failed: {e}")
            import traceback; traceback.print_exc()
            if 'temp_dir' in locals(): shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print("  SKIPPED — torch-fidelity not installed")

    # -------------------------------------------------------- build arrays --
    # Full-length stacks for FID, LPIPS, SSIM, PSNR
    gt_orig_stack  = np.stack(all_gt_videos,  axis=0)[:, :args.num_frames]  # (N, T, H, W, 3)
    gen_orig_stack = np.stack(all_gen_videos, axis=0)[:, :args.num_frames]

    # [N, T, 3, H, W] in [-1, 1]  — original res for LPIPS / SSIM / PSNR
    gt_orig  = gt_orig_stack.transpose(0, 1, 4, 2, 3).astype(np.float32)  / 127.5 - 1.0
    gen_orig = gen_orig_stack.transpose(0, 1, 4, 2, 3).astype(np.float32) / 127.5 - 1.0

    # 16-frame uniform sampling for FVD (cdfvd expects (B, 16, H, W, C) uint8)
    print("\n  Applying 16-frame uniform sampling for FVD...")
    gt_fvd_np  = np.stack([sample_video_frames(v) for v in all_gt_videos],  axis=0)
    gen_fvd_np = np.stack([sample_video_frames(v) for v in all_gen_videos], axis=0)
    print(f"  FVD input shape: {gen_fvd_np.shape}  (N, 16, H, W, C) uint8")

    # FVD evaluators typically resize to 224x224 internally
    fvd_h, fvd_w = 224, 224

    # --------------------------------------------------------------- FVD-I3D --
    print("\n[3/7] Computing FVD-I3D using cd-fvd...")
    fvd_i3d = None
    if deps['fvd_i3d'] and not args.skip_fvd_i3d:
        try:
            from cdfvd import fvd as cdfvd_lib
            evaluator_i3d = cdfvd_lib.cdfvd(model='i3d', device='cuda')
            fvd_i3d = evaluator_i3d.compute_fvd(gt_fvd_np, gen_fvd_np)
            print(f"  ✓ FVD-I3D: {fvd_i3d:.4f}")
        except Exception as e:
            print(f"  WARNING: FVD-I3D failed: {e}")
            import traceback; traceback.print_exc()
    else:
        reason = "skipped by flag" if args.skip_fvd_i3d else "cdfvd unavailable"
        print(f"  SKIPPED — {reason}")

    # --------------------------------------------------------- FVD-VideoMAE --
    print("\n[4/7] Computing FVD-VideoMAE using cd-fvd...")
    fvd_videomae = None
    if deps['fvd_videomae'] and not args.skip_fvd_videomae:
        try:
            from cdfvd import fvd as cdfvd_lib
            evaluator_videomae = cdfvd_lib.cdfvd(model='videomae', device='cuda')
            fvd_videomae = evaluator_videomae.compute_fvd(gt_fvd_np, gen_fvd_np)
            print(f"  ✓ FVD-VideoMAE: {fvd_videomae:.4f}")
        except Exception as e:
            print(f"  WARNING: FVD-VideoMAE failed: {e}")
            import traceback; traceback.print_exc()
    else:
        reason = "skipped by flag" if args.skip_fvd_videomae else "cdfvd unavailable"
        print(f"  SKIPPED — {reason}")

    # ----------------------------------------------------------------- LPIPS --
    print("\n[5/7] Computing LPIPS...")
    lpips_score = None
    if deps['lpips']:
        try:
            import lpips as lpips_lib
            loss_fn = lpips_lib.LPIPS(net='alex').cuda()
            gt_t2  = torch.from_numpy(gt_orig).cuda().float()
            gen_t2 = torch.from_numpy(gen_orig).cuda().float()
            total = 0.0
            for i in range(num_videos):
                with torch.no_grad():
                    total += loss_fn(gt_t2[i], gen_t2[i]).mean().item()
            lpips_score = total / num_videos
            del loss_fn, gt_t2, gen_t2
            print(f"  ✓ LPIPS: {lpips_score:.4f}")
        except Exception as e:
            print(f"  WARNING: LPIPS failed: {e}")
    else:
        print("  SKIPPED — lpips not installed")

    # ---------------------------------------------------------- SSIM & PSNR --
    print("\n[6/7] Computing SSIM and PSNR...")
    ssim_mean = ssim_std = psnr_mean = psnr_std = None
    ssim_per_video = psnr_per_video = None
    if deps['skimage']:
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            ssim_frames = np.zeros((num_videos, args.num_frames))
            psnr_frames = np.zeros((num_videos, args.num_frames))
            for vi in tqdm(range(num_videos), desc="  SSIM/PSNR"):
                for fi in range(args.num_frames):
                    ig, ig2 = gt_orig[vi, fi], gen_orig[vi, fi]
                    dr = max(ig.max(), ig2.max()) - min(ig.min(), ig2.min())
                    if dr < 1e-6: dr = 2.0
                    ssim_frames[vi, fi] = ssim_fn(ig, ig2, channel_axis=0, data_range=dr,
                                                  gaussian_weights=True, sigma=1.5)
                    psnr_frames[vi, fi] = psnr_fn(ig, ig2, data_range=dr)
            ssim_per_video = ssim_frames.mean(axis=1)
            psnr_per_video = psnr_frames.mean(axis=1)
            ssim_mean, ssim_std = ssim_frames.mean(), ssim_frames.std()
            psnr_mean, psnr_std = psnr_frames.mean(), psnr_frames.std()
            print(f"  ✓ SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
            print(f"  ✓ PSNR: {psnr_mean:.4f} ± {psnr_std:.4f} dB")
        except Exception as e:
            print(f"  WARNING: SSIM/PSNR failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  SKIPPED — scikit-image not installed")

    # --------------------------------------------------------- print summary --
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    lines = [
        "=" * 70,
        "Stage 2  FID / FVD-I3D / FVD-VideoMAE / LPIPS / SSIM / PSNR Results",
        "=" * 70,
        "",
        f"Frames directory  : {frames_dir}",
        f"Videos evaluated  : {num_videos}",
        f"Frames per video  : {args.num_frames}",
        f"Frame resolution  : {H}x{W}",
        f"FVD input size    : {fvd_h}x{fvd_w} (resized inside FVD evaluators)",
        "",
        "--- Image Quality ---",
        f"FID (Inception-v3): {fid_score:.4f}" if fid_score is not None else "FID: N/A (torch-fidelity not installed)",
        f"LPIPS             : {lpips_score:.4f}" if lpips_score is not None else "LPIPS: N/A",
        f"SSIM (mean ± std) : {ssim_mean:.4f} ± {ssim_std:.4f}" if ssim_mean is not None else "SSIM: N/A",
        f"PSNR (mean ± std) : {psnr_mean:.4f} ± {psnr_std:.4f} dB" if psnr_mean is not None else "PSNR: N/A",
        "",
        "--- Video Quality (FVD) ---",
        f"FVD-I3D           : {fvd_i3d:.4f}" if fvd_i3d is not None else "FVD-I3D: N/A",
        f"  backbone        : I3D (cdfvd/i3d_pretrained_400.pt, Kinetics400)",
        f"FVD-VideoMAE      : {fvd_videomae:.4f}" if fvd_videomae is not None else "FVD-VideoMAE: N/A",
        f"  backbone        : VideoMAE-Base (MCG-NJU/videomae-base, ViT-B/16)",
        "",
        "Interpretation:",
        "  FID    : lower is better  (<50 excellent, 50-100 good, >200 poor)",
        "  FVD-*  : lower is better  (temporal consistency; backbone affects absolute scale)",
        "  LPIPS  : lower is better  (<0.2 very similar)",
        "  SSIM   : higher is better (>0.8 good)",
        "  PSNR   : higher is better (>30 dB excellent)",
        "",
        "=" * 70,
    ]

    if ssim_per_video is not None:
        lines.insert(-1, "Per-Video Breakdown:")
        lines.insert(-1, f"  {'Video':<12} {'SSIM':>8} {'PSNR':>10}")
        lines.insert(-1, "  " + "-" * 32)
        for i in range(num_videos):
            lines.insert(-1, f"  video_{i:03d}    {ssim_per_video[i]:.4f}  {psnr_per_video[i]:.4f}")
        lines.insert(-1, "")

    output_text = "\n".join(lines)
    print(output_text)

    with open(args.output_file, 'w') as f:
        f.write(output_text)
    print(f"\n✓ Results saved to: {args.output_file}")

    json_results = {
        'frames_dir': frames_dir,
        'num_videos': num_videos,
        'num_frames': args.num_frames,
        'resolution': f'{H}x{W}',
        'fvd_input_size': f'{fvd_h}x{fvd_w}',
        'fid':      float(fid_score)   if fid_score   is not None else None,
        'fvd_i3d':      float(fvd_i3d)      if fvd_i3d      is not None else None,
        'fvd_videomae': float(fvd_videomae) if fvd_videomae  is not None else None,
        'lpips':    float(lpips_score) if lpips_score is not None else None,
        'ssim_mean': float(ssim_mean)  if ssim_mean   is not None else None,
        'ssim_std':  float(ssim_std)   if ssim_std    is not None else None,
        'psnr_mean': float(psnr_mean)  if psnr_mean   is not None else None,
        'psnr_std':  float(psnr_std)   if psnr_std    is not None else None,
    }
    if ssim_per_video is not None:
        json_results['per_video'] = {
            f'video_{i:03d}': {'ssim': float(ssim_per_video[i]), 'psnr': float(psnr_per_video[i])}
            for i in range(num_videos)
        }

    with open(args.output_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ JSON saved to:    {args.output_json}")
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
