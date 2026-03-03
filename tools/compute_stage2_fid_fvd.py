#!/usr/bin/env python3
"""
Compute FID, FVD, LPIPS, SSIM, and PSNR for Stage 2 generated RGB frames.

Reads from the saved frame directories produced by eval_stage2_rgb.py:
  frames/video_XXX/gt_rgb/frame_YYY.png
  frames/video_XXX/gen_rgb/frame_YYY.png

Metrics:
  - FID (Frechet Inception Distance): Image-level quality via torch-fidelity
  - FVD (Frechet Video Distance): Video-level quality via I3D features
  - LPIPS (Learned Perceptual Image Patch Similarity): Perceptual quality
  - SSIM (Structural Similarity Index): Structural similarity
  - PSNR (Peak Signal-to-Noise Ratio): Pixel-level quality

Usage:
    python tools/compute_stage2_fid_fvd.py \
        --frames_dir /no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/frames \
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute FID, FVD, LPIPS, SSIM, PSNR for Stage 2 generated frames'
    )
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video_XXX/ subdirs with gt_rgb/ and gen_rgb/')
    parser.add_argument('--num_frames', type=int, default=25,
                        help='Number of frames per video (default: 25)')
    parser.add_argument('--fid_batch_size', type=int, default=64,
                        help='Batch size for FID feature extraction')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: fid_fvd_results.txt in parent of frames_dir)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Output JSON path (default: fid_fvd_results.json in parent of frames_dir)')
    return parser.parse_args()


def load_video_frames(video_dir, subdir, num_frames):
    """
    Load frames from video_dir/subdir/frame_XXX.png
    
    Returns:
        frames: [T, H, W, 3] uint8 numpy array
    """
    frame_files = sorted([f for f in os.listdir(os.path.join(video_dir, subdir))
                          if f.endswith('.png')])[:num_frames]
    frames = []
    for ff in frame_files:
        img = np.array(Image.open(os.path.join(video_dir, subdir, ff)).convert('RGB'))
        frames.append(img)
    return np.stack(frames, axis=0)  # [T, H, W, 3]


def main():
    args = parse_args()
    
    frames_dir = args.frames_dir
    parent_dir = os.path.dirname(frames_dir.rstrip('/'))
    
    if args.output_file is None:
        args.output_file = os.path.join(parent_dir, 'fid_fvd_results.txt')
    if args.output_json is None:
        args.output_json = os.path.join(parent_dir, 'fid_fvd_results.json')
    
    # Discover video directories
    video_dirs = sorted([d for d in os.listdir(frames_dir)
                         if d.startswith('video_') and os.path.isdir(os.path.join(frames_dir, d))])
    
    if len(video_dirs) == 0:
        print(f"Error: No video_XXX directories found in {frames_dir}")
        sys.exit(1)
    
    num_videos = len(video_dirs)
    print("=" * 60)
    print("Stage 2: FID / FVD / LPIPS / SSIM / PSNR Evaluation")
    print("=" * 60)
    print(f"Frames directory: {frames_dir}")
    print(f"Videos found:     {num_videos}")
    print(f"Frames per video: {args.num_frames}")
    print()
    
    # ================================================================
    # Load all GT and generated frames
    # ================================================================
    print("[1/5] Loading video frames...")
    
    # First, check dimensions from first video
    sample_frame = Image.open(os.path.join(frames_dir, video_dirs[0], 'gen_rgb', 'frame_000.png'))
    W, H = sample_frame.size
    print(f"  Frame resolution: {H}x{W}")
    
    # For FVD: resize to (128, 512) to match the existing pipeline convention
    fvd_h, fvd_w = 128, 512
    
    all_gt_videos = []  # list of [T, H, W, 3] uint8
    all_gen_videos = []
    
    for vd in tqdm(video_dirs, desc="  Loading videos"):
        vpath = os.path.join(frames_dir, vd)
        gt_frames = load_video_frames(vpath, 'gt_rgb', args.num_frames)
        gen_frames = load_video_frames(vpath, 'gen_rgb', args.num_frames)
        all_gt_videos.append(gt_frames)
        all_gen_videos.append(gen_frames)
    
    print(f"  ✓ Loaded {num_videos} videos × {args.num_frames} frames")
    
    # ================================================================
    # FID (torch-fidelity)
    # ================================================================
    print("\n[2/5] Computing FID...")
    fid_score = None
    try:
        from torch_fidelity import calculate_metrics
        
        # Create temp dirs with all individual frames
        temp_dir = tempfile.mkdtemp(prefix='stage2_fid_')
        gt_dir = os.path.join(temp_dir, 'gt')
        gen_dir = os.path.join(temp_dir, 'gen')
        os.makedirs(gt_dir)
        os.makedirs(gen_dir)
        
        idx = 0
        for vid_i, (gt_vid, gen_vid) in enumerate(zip(all_gt_videos, all_gen_videos)):
            for t in range(gt_vid.shape[0]):
                Image.fromarray(gt_vid[t]).save(os.path.join(gt_dir, f'{idx:06d}.png'))
                Image.fromarray(gen_vid[t]).save(os.path.join(gen_dir, f'{idx:06d}.png'))
                idx += 1
        
        total_frames = idx
        print(f"  Total frames for FID: {total_frames} (GT) vs {total_frames} (Gen)")
        
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=gt_dir,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
            batch_size=args.fid_batch_size,
        )
        fid_score = metrics['frechet_inception_distance']
        print(f"  ✓ FID: {fid_score:.4f}")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except ImportError:
        print("  WARNING: torch-fidelity not installed. Skipping FID.")
        print("  Install with: pip install torch-fidelity")
    except Exception as e:
        print(f"  WARNING: FID computation failed: {e}")
        import traceback; traceback.print_exc()
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ================================================================
    # Prepare tensors for FVD / LPIPS / SSIM / PSNR
    # ================================================================
    print("\n  Preparing video tensors...")
    
    # Stack into arrays: [N, T, C, H, W] for FVD/LPIPS
    # Resize to (fvd_h, fvd_w) for FVD
    gt_array = np.zeros((num_videos, args.num_frames, 3, fvd_h, fvd_w), dtype=np.float32)
    gen_array = np.zeros((num_videos, args.num_frames, 3, fvd_h, fvd_w), dtype=np.float32)
    
    # Also keep original resolution for SSIM/PSNR
    gt_orig = np.zeros((num_videos, args.num_frames, 3, H, W), dtype=np.float32)
    gen_orig = np.zeros((num_videos, args.num_frames, 3, H, W), dtype=np.float32)
    
    for i, (gt_vid, gen_vid) in enumerate(zip(all_gt_videos, all_gen_videos)):
        T = min(gt_vid.shape[0], args.num_frames)
        for t in range(T):
            # Original res: [H, W, 3] -> [3, H, W], normalize to [-1, 1]
            gt_chw = gt_vid[t].transpose(2, 0, 1).astype(np.float32)
            gen_chw = gen_vid[t].transpose(2, 0, 1).astype(np.float32)
            gt_orig[i, t] = gt_chw / 127.5 - 1.0
            gen_orig[i, t] = gen_chw / 127.5 - 1.0
            
            # Resized for FVD
            gt_t = torch.from_numpy(gt_chw).unsqueeze(0)
            gen_t = torch.from_numpy(gen_chw).unsqueeze(0)
            gt_resized = F.interpolate(gt_t, size=(fvd_h, fvd_w), mode='bilinear', align_corners=False)
            gen_resized = F.interpolate(gen_t, size=(fvd_h, fvd_w), mode='bilinear', align_corners=False)
            gt_array[i, t] = gt_resized.squeeze(0).numpy() / 127.5 - 1.0
            gen_array[i, t] = gen_resized.squeeze(0).numpy() / 127.5 - 1.0
    
    gt_tensor = torch.from_numpy(gt_array).cuda().float()
    gen_tensor = torch.from_numpy(gen_array).cuda().float()
    
    # ================================================================
    # FVD
    # ================================================================
    print("\n[3/5] Computing FVD...")
    fvd_score = None
    try:
        # Use the original FVD implementation from ctrlv (I3D-based, works with tensors directly)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from ctrlv.metrics.fvd import FVD
        print("  Using I3D-based FVD implementation...")
        print(f"  Video tensor shape: {gen_tensor.shape} (N, T, C, H, W)")
        fvd_evaluator = FVD(device='cuda')
        fvd_score = fvd_evaluator.evaluate(gen_tensor, gt_tensor)
        del fvd_evaluator
        print(f"  ✓ FVD: {fvd_score:.4f}")
    except Exception as e:
        print(f"  WARNING: FVD computation failed: {e}")
        import traceback; traceback.print_exc()
    
    # Free FVD tensors
    del gt_tensor, gen_tensor
    torch.cuda.empty_cache()
    
    # ================================================================
    # LPIPS
    # ================================================================
    print("\n[4/5] Computing LPIPS...")
    lpips_score = None
    try:
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        
        # Use original resolution tensors
        gt_orig_t = torch.from_numpy(gt_orig).cuda().float()
        gen_orig_t = torch.from_numpy(gen_orig).cuda().float()
        
        lpips_total = 0
        for i in range(num_videos):
            with torch.no_grad():
                lpips_total += loss_fn_alex(gt_orig_t[i], gen_orig_t[i]).mean().item()
        lpips_score = lpips_total / num_videos
        
        del loss_fn_alex
        print(f"  ✓ LPIPS: {lpips_score:.4f}")
    except ImportError:
        print("  WARNING: lpips not installed. Skipping.")
    except Exception as e:
        print(f"  WARNING: LPIPS computation failed: {e}")
        import traceback; traceback.print_exc()
    
    # ================================================================
    # SSIM & PSNR
    # ================================================================
    print("\n[5/5] Computing SSIM and PSNR...")
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        ssim_per_frame = np.zeros((num_videos, args.num_frames))
        psnr_per_frame = np.zeros((num_videos, args.num_frames))
        
        for vid_idx in tqdm(range(num_videos), desc="  SSIM/PSNR"):
            for f_idx in range(args.num_frames):
                img_gt = gt_orig[vid_idx, f_idx]   # [3, H, W] in [-1, 1]
                img_gen = gen_orig[vid_idx, f_idx]
                data_range = max(img_gt.max(), img_gen.max()) - min(img_gt.min(), img_gen.min())
                if data_range < 1e-6:
                    data_range = 2.0
                ssim_per_frame[vid_idx, f_idx] = ssim(
                    img_gt, img_gen, channel_axis=0, data_range=data_range,
                    gaussian_weights=True, sigma=1.5
                )
                psnr_per_frame[vid_idx, f_idx] = psnr(img_gt, img_gen, data_range=data_range)
        
        ssim_mean = ssim_per_frame.mean()
        ssim_std = ssim_per_frame.std()
        ssim_error = np.sqrt(((ssim_per_frame - ssim_mean)**2).sum() / num_videos)
        psnr_mean = psnr_per_frame.mean()
        psnr_std = psnr_per_frame.std()
        psnr_error = np.sqrt(((psnr_per_frame - psnr_mean)**2).sum() / num_videos)
        
        # Per-video means
        ssim_per_video = ssim_per_frame.mean(axis=1)
        psnr_per_video = psnr_per_frame.mean(axis=1)
        
        print(f"  ✓ SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
        print(f"  ✓ PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")
    except ImportError:
        print("  WARNING: scikit-image not installed. Skipping SSIM/PSNR.")
        ssim_mean = ssim_std = ssim_error = None
        psnr_mean = psnr_std = psnr_error = None
        ssim_per_frame = psnr_per_frame = None
        ssim_per_video = psnr_per_video = None
    except Exception as e:
        print(f"  WARNING: SSIM/PSNR computation failed: {e}")
        import traceback; traceback.print_exc()
        ssim_mean = ssim_std = ssim_error = None
        psnr_mean = psnr_std = psnr_error = None
        ssim_per_frame = psnr_per_frame = None
        ssim_per_video = psnr_per_video = None
    
    # ================================================================
    # Print & Save Results
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("Stage 2 FID / FVD / LPIPS / SSIM / PSNR Results")
    results_lines.append("=" * 60)
    results_lines.append("")
    results_lines.append(f"Frames directory:  {frames_dir}")
    results_lines.append(f"Videos evaluated:  {num_videos}")
    results_lines.append(f"Frames per video:  {args.num_frames}")
    results_lines.append(f"Frame resolution:  {H}x{W}")
    results_lines.append(f"FVD resolution:    {fvd_h}x{fvd_w}")
    results_lines.append("")
    
    if fid_score is not None:
        results_lines.append(f"FID:               {fid_score:.4f}")
    else:
        results_lines.append(f"FID:               N/A (torch-fidelity not installed)")
    
    if fvd_score is not None:
        results_lines.append(f"FVD:               {fvd_score:.4f}")
    else:
        results_lines.append(f"FVD:               N/A")
    
    if lpips_score is not None:
        results_lines.append(f"LPIPS:             {lpips_score:.4f}")
    else:
        results_lines.append(f"LPIPS:             N/A")
    
    if ssim_mean is not None:
        results_lines.append(f"SSIM (mean):       {ssim_mean:.4f}")
        results_lines.append(f"SSIM (std):        {ssim_std:.4f}")
        results_lines.append(f"SSIM (error):      {ssim_error:.4f}")
    else:
        results_lines.append(f"SSIM:              N/A")
    
    if psnr_mean is not None:
        results_lines.append(f"PSNR (mean):       {psnr_mean:.4f}")
        results_lines.append(f"PSNR (std):        {psnr_std:.4f}")
        results_lines.append(f"PSNR (error):      {psnr_error:.4f}")
    else:
        results_lines.append(f"PSNR:              N/A")
    
    results_lines.append("")
    
    # Per-video breakdown
    if ssim_per_video is not None:
        results_lines.append("Per-Video Breakdown:")
        results_lines.append(f"  {'Video':<12} {'SSIM':>8} {'PSNR':>8}")
        results_lines.append(f"  {'-'*30}")
        for i in range(num_videos):
            results_lines.append(f"  video_{i:03d}    {ssim_per_video[i]:.4f}   {psnr_per_video[i]:.4f}")
        results_lines.append(f"  {'Average':<12} {ssim_per_video.mean():.4f}   {psnr_per_video.mean():.4f}")
        results_lines.append("")
    
    # Interpretation guide
    results_lines.append("Interpretation Guide:")
    results_lines.append("  FID:   Lower is better. <50 excellent, 50-100 good, 100-200 moderate, >200 poor")
    results_lines.append("  FVD:   Lower is better. Measures temporal consistency of generated video")
    results_lines.append("  LPIPS: Lower is better. <0.2 very similar, 0.2-0.4 similar, >0.5 different")
    results_lines.append("  SSIM:  Higher is better. 1.0 = identical, >0.8 good, >0.5 moderate")
    results_lines.append("  PSNR:  Higher is better. >30 dB excellent, 20-30 good, <20 noisy")
    results_lines.append("")
    results_lines.append("=" * 60)
    
    output_text = "\n".join(results_lines)
    print(output_text)
    
    # Save text
    with open(args.output_file, 'w') as f:
        f.write(output_text)
    print(f"\n✓ Results saved to: {args.output_file}")
    
    # Save JSON
    json_results = {
        'frames_dir': frames_dir,
        'num_videos': num_videos,
        'num_frames': args.num_frames,
        'resolution': f'{H}x{W}',
        'fvd_resolution': f'{fvd_h}x{fvd_w}',
        'fid': float(fid_score) if fid_score is not None else None,
        'fvd': float(fvd_score) if fvd_score is not None else None,
        'lpips': float(lpips_score) if lpips_score is not None else None,
        'ssim_mean': float(ssim_mean) if ssim_mean is not None else None,
        'ssim_std': float(ssim_std) if ssim_std is not None else None,
        'ssim_error': float(ssim_error) if ssim_error is not None else None,
        'psnr_mean': float(psnr_mean) if psnr_mean is not None else None,
        'psnr_std': float(psnr_std) if psnr_std is not None else None,
        'psnr_error': float(psnr_error) if psnr_error is not None else None,
    }
    if ssim_per_video is not None:
        json_results['per_video'] = {}
        for i in range(num_videos):
            json_results['per_video'][f'video_{i:03d}'] = {
                'ssim': float(ssim_per_video[i]),
                'psnr': float(psnr_per_video[i]),
            }
    
    with open(args.output_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ JSON saved to: {args.output_json}")
    
    print(f"\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
