#!/usr/bin/env python3
"""
generate_stage2_frames_for_drn.py

Phase 1 of the multi-scale DRN mIoU evaluation pipeline.

Generates Stage 2 ControlNet RGB frames for the fixed 19-clip evaluation set
defined by drn_eval/CTRLV_STAGE2/val_labels.txt (475 public-path GT label
entries: 19 clip groups × 25 frames, covering 9 KITTI-360 sequences).

The GT label paths in drn_eval/CTRLV_STAGE2/val_labels.txt are the single
source of truth for which frames to evaluate.  This file is checked into the
project and must not be modified between runs, ensuring all checkpoints are
compared on exactly the same set of frames.

Outputs (inside --output_dir/CTRLV_STAGE2/):
  val_images.txt        relative paths to generated RGB PNGs (from CTRLV_STAGE2/)
  val_labels.txt        copy of drn_eval/CTRLV_STAGE2/val_labels.txt
  generated_frames/     the generated RGB frame directories
  metadata.json         clip metadata (sequence, frame IDs, clip index)

Confidence map paths are automatically derived by segment.py at evaluation
time by replacing "semantic" → "confidence" in each val_labels.txt entry:
  /misc/data/public/kitti-360/KITTI-360/data_2d_confidences/train/{seq}/...

Usage
-----
  python tools/generate_stage2_frames_for_drn.py \\
      --checkpoint_dir /no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze_reinject \\
      --output_dir     /no_backups/s1492/Ctrl-V/outputs/eval_stage2_drn_ms

Then run Phase 2 (segment.py) via eval_stage2_drn_ms.sh.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Make ctrlv importable
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ctrlv.utils.semantic_preprocessing import (
    KITTI360_LABEL_MAPPING,
    load_and_remap_semantic,
    semantic_ids_to_viz_rgb,
)

KITTI360_ROOT = "/misc/data/public/kitti-360/KITTI-360"


# ============================================================================
# Helpers
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Stage 2 RGB frames for multi-scale DRN evaluation")

    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Stage 2 ControlNet checkpoint directory")
    parser.add_argument("--output_dir", type=str,
                        default="/no_backups/s1492/Ctrl-V/outputs/eval_stage2_drn_ms",
                        help="Root output directory; CTRLV_STAGE2/ will be created inside it")
    parser.add_argument("--drn_eval_dir", type=str,
                        default=os.path.join(os.path.dirname(_script_dir), "drn_eval"),
                        help="Path to drn_eval/ directory; contains the fixed val_labels.txt and info.json")
    parser.add_argument("--kitti360_root", type=str,
                        default=KITTI360_ROOT,
                        help="Public KITTI-360 root (used to derive RGB init-frame paths)")

    # Stage 2 model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--semantic_vae_checkpoint", type=str,
                        default="/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth")

    # Clip settings
    parser.add_argument("--clip_length",  type=int,   default=25,
                        help="Frames per clip; first clip_length frames of each 30-frame group are used")
    parser.add_argument("--train_H",      type=int,   default=192)
    parser.add_argument("--train_W",      type=int,   default=704)

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int,   default=30)
    parser.add_argument("--min_guidance_scale",  type=float, default=1.0)
    parser.add_argument("--max_guidance_scale",  type=float, default=3.0)
    parser.add_argument("--conditioning_scale",  type=float, default=1.0)
    parser.add_argument("--noise_aug_strength",  type=float, default=0.01)
    parser.add_argument("--fps",                 type=int,   default=7)
    parser.add_argument("--seed",                type=int,   default=1234)

    return parser.parse_args()


def load_clip_groups(val_labels_txt: str, kitti360_root: str, clip_length: int):
    """
    Load the fixed evaluation set from drn_eval/CTRLV_STAGE2/val_labels.txt.

    The file contains public-path GT semantic entries, one per line, already
    grouped into consecutive clip_length-frame blocks.  Each block of
    clip_length consecutive lines (same sequence, consecutive frame IDs)
    is one evaluation clip.

    Returns a list of clip-group dicts, each containing:
        {
          'sequence': str,
          'frames': [
              {
                'frame_id':        int,
                'sequence':        str,
                'gt_semantic_path': str,  # public path (from val_labels.txt)
                'rgb_path':        str,  # public raw RGB path
              },
              ...  (clip_length entries)
          ]
        }
    """
    # Read all label paths
    with open(val_labels_txt) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    print(f"  Loaded {len(all_paths)} GT label entries from {val_labels_txt}")

    # Group into clips of clip_length consecutive entries from the same sequence
    groups = []
    i = 0
    while i < len(all_paths):
        chunk_paths = all_paths[i:i + clip_length]
        frames = []
        for gt_path in chunk_paths:
            parts = gt_path.replace("\\", "/").split("/")
            seq = next((p for p in parts if "_sync" in p and "drive" in p), None)
            frame_id = int(os.path.splitext(parts[-1])[0])
            fname = f"{frame_id:010d}.png"
            rgb_path = os.path.join(
                kitti360_root, "data_2d_raw",
                seq, "image_00", "data_rect", fname
            )
            frames.append({
                "frame_id":         frame_id,
                "sequence":         seq,
                "gt_semantic_path": gt_path,   # taken directly from val_labels.txt
                "rgb_path":         rgb_path,
            })
        seq_of_chunk = frames[0]["sequence"]
        groups.append({"sequence": seq_of_chunk, "frames": frames})
        i += clip_length

    print(f"  Grouped into {len(groups)} clips of {clip_length} frames each")
    for g in groups:
        fids = [f["frame_id"] for f in g["frames"]]
        print(f"    {g['sequence']:40s}  frames {fids[0]}–{fids[-1]}")
    return groups


def load_semantic_clip(frames_meta, train_H: int, train_W: int):
    """
    Load GT semantic maps for a clip.

    Returns:
        semantic_ids_np  np.ndarray [T, H, W]  trainIDs (0-18, 255=ignore)
        sem_rgb_tensor   torch.Tensor [T, 3, H, W] float32 in [-1, 1] (for ControlNet cond_images)
        semantic_ids_pt  torch.Tensor [T, H, W] int64
    """
    sem_list = []
    for meta in frames_meta:
        ids = load_and_remap_semantic(meta["gt_semantic_path"], ignore_index=255)
        # Resize to eval resolution using nearest-neighbour
        ids_t = torch.from_numpy(ids.astype(np.int64)).long()
        if ids_t.shape != (train_H, train_W):
            ids_t = F.interpolate(
                ids_t.unsqueeze(0).unsqueeze(0).float(),
                size=(train_H, train_W),
                mode="nearest",
            ).squeeze().long()
        sem_list.append(ids_t)

    semantic_ids_pt = torch.stack(sem_list, dim=0)      # [T, H, W]
    semantic_ids_np = semantic_ids_pt.numpy()

    # Build semantic RGB visualization tensor for cond_images
    rgb_frames = []
    for t in range(len(frames_meta)):
        vis = semantic_ids_to_viz_rgb(semantic_ids_np[t])          # [H, W, 3] uint8
        rgb_t = torch.from_numpy(vis).permute(2, 0, 1).float() / 127.5 - 1.0  # [-1, 1]
        rgb_frames.append(rgb_t)
    sem_rgb_tensor = torch.stack(rgb_frames, dim=0)                # [T, 3, H, W]

    return semantic_ids_np, sem_rgb_tensor, semantic_ids_pt


def load_init_rgb(rgb_path: str, train_H: int, train_W: int) -> Image.Image:
    """Load and resize the first RGB frame (SVD image conditioning)."""
    img = Image.open(rgb_path).convert("RGB")
    if img.size != (train_W, train_H):
        img = img.resize((train_W, train_H), Image.LANCZOS)
    return img


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    ctrlv_stage2_dir = os.path.join(args.output_dir, "CTRLV_STAGE2")
    gen_frames_dir   = os.path.join(ctrlv_stage2_dir, "generated_frames")
    os.makedirs(gen_frames_dir, exist_ok=True)

    # Copy info.json from drn_eval template
    src_info = os.path.join(args.drn_eval_dir, "CTRLV_STAGE2", "info.json")
    dst_info = os.path.join(ctrlv_stage2_dir, "info.json")
    if not os.path.exists(dst_info):
        import shutil
        shutil.copy(src_info, dst_info)
        print(f"  Copied info.json to {dst_info}")
    else:
        print(f"  info.json already present at {dst_info}")

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    print("=" * 70)
    print("Stage 2 Frame Generation for Multi-Scale DRN mIoU Evaluation")
    print("=" * 70)
    print(f"  Checkpoint  : {args.checkpoint_dir}")
    print(f"  Output      : {ctrlv_stage2_dir}")
    print(f"  Resolution  : {args.train_H}×{args.train_W}  clip_length={args.clip_length}")
    print(f"  Infer steps : {args.num_inference_steps}")

    # ------------------------------------------------------------------
    # [1] Load evaluation clips from the fixed val_labels.txt
    # ------------------------------------------------------------------
    # drn_eval/CTRLV_STAGE2/val_labels.txt is the single source of truth:
    # 475 public-path GT label entries (19 clips × 25 frames, 9 sequences).
    # This file is version-controlled — do not modify between runs.
    static_val_labels = os.path.join(args.drn_eval_dir, "CTRLV_STAGE2", "val_labels.txt")
    if not os.path.exists(static_val_labels):
        raise FileNotFoundError(
            f"Fixed val_labels.txt not found at {static_val_labels}. "
            "Run from project root or pass --drn_eval_dir correctly."
        )
    print(f"\n[1/4] Loading evaluation set from {static_val_labels}")
    groups = load_clip_groups(static_val_labels, args.kitti360_root, args.clip_length)

    # ------------------------------------------------------------------
    # [2] Load Stage 2 pipeline
    # ------------------------------------------------------------------
    print("\n[2/4] Loading Stage 2 ControlNet pipeline...")

    from diffusers.models import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    from ctrlv.models import ControlNetModel, DualVAEManager, UNetSpatioTemporalConditionModel
    from ctrlv.pipelines.pipeline_video_control import StableVideoControlPipeline

    # Resolve checkpoint
    ckpt_dir = args.checkpoint_dir
    best_path = os.path.join(ckpt_dir, "best_checkpoint")
    if os.path.exists(best_path):
        ckpt_path = best_path
        ckpt_step = "best"
    else:
        subdirs = sorted(
            [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint")],
            key=lambda x: int(x.split("-")[1]),
        )
        if not subdirs:
            raise ValueError(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, subdirs[-1])
        ckpt_step = subdirs[-1].split("-")[1]
    print(f"  Using checkpoint: {ckpt_path}  (step {ckpt_step})")

    ctrlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="control_net")
    unet    = UNetSpatioTemporalConditionModel.from_pretrained(
        ckpt_path, subfolder="unet",
        low_cpu_mem_usage=True, num_frames=args.clip_length,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant="fp16"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    vae_manager = DualVAEManager(
        rgb_vae=vae,
        semantic_vae_checkpoint=args.semantic_vae_checkpoint,
        num_semantic_classes=19,
        device=device,
        clip_size=args.clip_length,
        verbose=True,
    )
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    ctrlnet.to(device, dtype=weight_dtype)
    ctrlnet.eval()
    unet.eval()

    pipeline = StableVideoControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet, controlnet=ctrlnet,
        image_encoder=image_encoder, vae=vae,
        feature_extractor=feature_extractor,
        variant="fp16", torch_dtype=weight_dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae_manager = vae_manager
    print("  ✓ Pipeline ready")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------
    # [3] Generate RGB frames for each clip group
    # ------------------------------------------------------------------
    print(f"\n[3/4] Generating RGB frames for {len(groups)} clip groups...")

    val_images_entries  = []   # relative paths (from CTRLV_STAGE2/)
    metadata_clips      = []

    # Track per-sequence clip index for consistent naming (e.g. drive_0000_sync_00,_01,_02)
    seq_clip_counters: dict = {}

    for group_idx, group in enumerate(groups):
        seq    = group["sequence"]
        frames = group["frames"]
        n_frames = len(frames)

        # Per-sequence clip index resets to 0 for each new sequence.
        # Matches the naming convention in the reference evaluation folder.
        seq_clip_idx = seq_clip_counters.get(seq, 0)
        seq_clip_counters[seq] = seq_clip_idx + 1

        folder_name = f"{seq}_{seq_clip_idx:02d}"
        frame_out_dir = os.path.join(gen_frames_dir, folder_name)
        os.makedirs(frame_out_dir, exist_ok=True)

        print(f"  [{group_idx+1:2d}/{len(groups)}]  {folder_name}  "
              f"frames {frames[0]['frame_id']}–{frames[-1]['frame_id']}  ({n_frames} frames)  ", end="", flush=True)

        # Load semantic conditioning
        sem_ids_np, sem_rgb_t, sem_ids_pt = load_semantic_clip(
            frames, args.train_H, args.train_W
        )
        # [1, T, 3, H, W]  semantic RGB for cond_images
        cond_images = sem_rgb_t.unsqueeze(0).to(device, dtype=weight_dtype)
        # [1, T, H, W]  trainIDs for semantic VAE
        semantic_ids_cond = sem_ids_pt.unsqueeze(0)

        # Load init RGB frame
        image_init = load_init_rgb(frames[0]["rgb_path"], args.train_H, args.train_W)

        # Run Stage 2 inference
        with torch.autocast(str(device).replace(":0", ""), enabled=True):
            result = pipeline(
                image_init,
                cond_images=cond_images,
                height=args.train_H, width=args.train_W,
                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.clip_length,
                control_condition_scale=args.conditioning_scale,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                noise_aug_strength=args.noise_aug_strength,
                generator=generator,
                output_type="pt",
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )

        gen_frames_pt = result.frames[0]                                     # [T, 3, H, W] float [0,1]
        gen_frames_np = (gen_frames_pt.detach().cpu().numpy() * 255).astype(np.uint8)  # [T, 3, H, W]
        del result, gen_frames_pt
        torch.cuda.empty_cache()

        # Save generated frames and record txt entries
        for t in range(n_frames):
            fname   = f"frame_{t+1:04d}.png"
            abs_out = os.path.join(frame_out_dir, fname)
            rel_out = os.path.join("generated_frames", folder_name, fname)

            img_hwc = gen_frames_np[t].transpose(1, 2, 0)               # [H, W, 3]
            Image.fromarray(img_hwc).save(abs_out)

            val_images_entries.append(rel_out)

        metadata_clips.append({
            "group_idx":  group_idx,
            "folder_name": folder_name,
            "sequence":   seq,
            "frame_ids":  [f["frame_id"] for f in frames],
            "n_frames":   n_frames,
        })
        print(f"✓")

    # ------------------------------------------------------------------
    # [4] Write output files
    # ------------------------------------------------------------------
    print("\n[4/4] Writing val_images.txt, val_labels.txt, metadata.json ...")

    with open(os.path.join(ctrlv_stage2_dir, "val_images.txt"), "w") as f:
        f.write("\n".join(val_images_entries) + "\n")

    # val_labels.txt is copied verbatim from the project's fixed reference file.
    # This guarantees the GT labels in the output are identical across all runs.
    import shutil
    shutil.copy(static_val_labels, os.path.join(ctrlv_stage2_dir, "val_labels.txt"))

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({
            "checkpoint_dir":   args.checkpoint_dir,
            "checkpoint_step":  ckpt_step,
            "clip_length":      args.clip_length,
            "resolution":       f"{args.train_H}x{args.train_W}",
            "num_inference_steps": args.num_inference_steps,
            "seed":             args.seed,
            "num_groups":       len(groups),
            "total_frames":     len(val_images_entries),
            "clips":            metadata_clips,
        }, f, indent=2)

    print(f"\n  ✓ val_images.txt : {len(val_images_entries)} entries")
    print(f"  ✓ val_labels.txt : copied from {static_val_labels}")
    print(f"  ✓ metadata.json  : {args.output_dir}/metadata.json")
    print(f"\n  Confidence maps will be resolved automatically by segment.py")
    print(f"  by replacing 'semantic' → 'confidence' in val_labels.txt paths.")
    print(f"\n  CTRLV_STAGE2 data_dir : {ctrlv_stage2_dir}")
    print(f"\n  ✓ Phase 1 complete. Run Phase 2:")
    print(f"    cd {os.path.join(os.path.dirname(_script_dir), 'drn_eval')}")
    print(f"    python segment.py test -d {ctrlv_stage2_dir} \\")
    print(f"        -c 19 --arch drn_d_105 \\")
    print(f"        --pretrained KITTI360_checkpoints/checkpoint_030.pth.tar \\")
    print(f"        --phase val --batch-size 1 --ms")


if __name__ == "__main__":
    main()
