"""
Stage 1 Evaluation: Semantic Prediction Quality

Evaluates the Stage 1 diffusion model (RGB → Semantic) on validation data.

Metrics computed:
- Per-class IoU (Intersection over Union)
- Mean IoU (mIoU)
- Overall Pixel Accuracy
- Per-class Pixel Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix (saved as image)

Also saves visualization outputs:
- Ground truth semantic frames (colorized PNG)
- Generated semantic frames (colorized PNG)
- Side-by-side comparison images

Usage:
    python tools/eval_stage1_semantic.py \
        --checkpoint_dir /path/to/stage1/checkpoint \
        --output_dir /path/to/eval_output \
        --num_samples 10 \
        --clip_length 25
"""

from accelerate.utils import write_basic_config
write_basic_config()

import warnings
import logging
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.utils.torch_utils import is_compiled_module

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import get_dataloader, encode_video_image, get_add_time_ids, get_n_training_samples, get_model_attr, eval_samples_generator
    from ctrlv.models import UNetSpatioTemporalConditionModel
    from ctrlv.pipelines import VideoDiffusionPipeline
    from ctrlv.utils.semantic_preprocessing import (
        KITTI360_LABEL_MAPPING,
        KITTI360_CLASS_NAMES,
        KITTI360_VIZ_COLORS,
        semantic_ids_to_viz_rgb,
    )

logger = get_logger(__name__, log_level="INFO")

# ============================================================================
# Metrics
# ============================================================================

class SemanticMetrics:
    """Accumulates predictions and computes semantic segmentation metrics."""
    
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Confusion matrix: rows=GT, cols=Predicted
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """
        Update confusion matrix with a batch of predictions.
        
        Args:
            pred: [H, W] or [T, H, W] predicted trainIDs (0-18)
            gt: [H, W] or [T, H, W] ground truth trainIDs (0-18, 255=ignore)
        """
        if pred.ndim == 3:
            for t in range(pred.shape[0]):
                self._update_single(pred[t], gt[t])
        else:
            self._update_single(pred, gt)
    
    def _update_single(self, pred: np.ndarray, gt: np.ndarray):
        """Update with a single frame pair [H, W]."""
        valid_mask = (gt != self.ignore_index) & (gt < self.num_classes)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        for g, p in zip(gt_valid.flatten(), pred_valid.flatten()):
            self.confusion_matrix[g, p] += 1
    
    def compute(self):
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix
        
        # Per-class metrics
        tp = np.diag(cm)  # True positives
        fp = cm.sum(axis=0) - tp  # False positives (predicted as class but GT is different)
        fn = cm.sum(axis=1) - tp  # False negatives (GT is class but predicted different)
        
        # IoU per class
        denominator = tp + fp + fn
        iou_per_class = np.where(denominator > 0, tp / denominator, np.nan)
        
        # Pixel accuracy per class (recall)
        gt_per_class = cm.sum(axis=1)
        acc_per_class = np.where(gt_per_class > 0, tp / gt_per_class, np.nan)
        
        # Precision per class
        pred_per_class = cm.sum(axis=0)
        precision_per_class = np.where(pred_per_class > 0, tp / pred_per_class, np.nan)
        
        # F1 per class
        f1_per_class = np.where(
            (precision_per_class + acc_per_class) > 0,
            2 * precision_per_class * acc_per_class / (precision_per_class + acc_per_class),
            np.nan
        )
        
        # Mean IoU (only over classes present in GT)
        valid_classes = ~np.isnan(iou_per_class)
        miou = np.nanmean(iou_per_class) if valid_classes.any() else 0.0
        
        # Overall pixel accuracy
        total_correct = tp.sum()
        total_pixels = cm.sum()
        overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        # Mean accuracy (average per-class recall)
        mean_accuracy = np.nanmean(acc_per_class) if valid_classes.any() else 0.0
        
        # Frequency-weighted IoU
        freq = gt_per_class / gt_per_class.sum() if gt_per_class.sum() > 0 else np.zeros(self.num_classes)
        fwiou = np.nansum(freq * iou_per_class)
        
        return {
            'miou': float(miou),
            'overall_accuracy': float(overall_accuracy),
            'mean_accuracy': float(mean_accuracy),
            'fwiou': float(fwiou),
            'iou_per_class': iou_per_class,
            'accuracy_per_class': acc_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': acc_per_class,  # recall = accuracy per class
            'f1_per_class': f1_per_class,
            'gt_pixels_per_class': gt_per_class,
            'pred_pixels_per_class': pred_per_class,
            'confusion_matrix': cm,
        }


def save_confusion_matrix_image(cm, class_names, output_path):
    """Save confusion matrix as a PNG image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Normalize by row (GT class) for better visualization
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(row_sums > 0, cm / row_sums, 0)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Normalized Confusion Matrix (Row-Normalized)')
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_normalized[i, j]
                if val > 0.01:
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved confusion matrix to: {output_path}")
    except ImportError:
        print("  Warning: matplotlib not available, skipping confusion matrix image")


def save_semantic_frame(semantic_ids, output_path):
    """Save a single semantic ID frame as colorized PNG."""
    rgb = semantic_ids_to_viz_rgb(semantic_ids)  # [H, W] -> [H, W, 3]
    Image.fromarray(rgb).save(output_path)


def save_side_by_side(gt_ids, pred_ids, output_path):
    """Save GT and predicted semantic frames side by side."""
    gt_rgb = semantic_ids_to_viz_rgb(gt_ids)
    pred_rgb = semantic_ids_to_viz_rgb(pred_ids)
    
    H, W, _ = gt_rgb.shape
    # Add a small separator
    sep = np.ones((H, 4, 3), dtype=np.uint8) * 255
    combined = np.concatenate([gt_rgb, sep, pred_rgb], axis=1)
    Image.fromarray(combined).save(output_path)


def _save_clip_frames(pred_np, gt_np, video_dir):
    """Save colorized + grayscale frames for a clip. Returns number of frames saved."""
    gt_dir = os.path.join(video_dir, 'gt')
    pred_dir = os.path.join(video_dir, 'pred')
    compare_dir = os.path.join(video_dir, 'comparison')
    gt_gray_dir = os.path.join(video_dir, 'gt_grayscale')
    pred_gray_dir = os.path.join(video_dir, 'pred_grayscale')
    for d in [gt_dir, pred_dir, compare_dir, gt_gray_dir, pred_gray_dir]:
        os.makedirs(d, exist_ok=True)

    trainid_to_original = {tid: kid for kid, tid in KITTI360_LABEL_MAPPING.items()}
    T = min(pred_np.shape[0], gt_np.shape[0])
    for t in range(T):
        save_semantic_frame(gt_np[t], os.path.join(gt_dir, f'frame_{t:03d}.png'))
        save_semantic_frame(pred_np[t], os.path.join(pred_dir, f'frame_{t:03d}.png'))
        save_side_by_side(gt_np[t], pred_np[t], os.path.join(compare_dir, f'frame_{t:03d}.png'))

        gt_original = np.zeros_like(gt_np[t], dtype=np.uint8)
        pred_original = np.zeros_like(pred_np[t], dtype=np.uint8)
        for train_id, orig_id in trainid_to_original.items():
            gt_original[gt_np[t] == train_id] = orig_id
            pred_original[pred_np[t] == train_id] = orig_id
        gt_original[gt_np[t] == 255] = 0
        Image.fromarray(gt_original, mode='L').save(os.path.join(gt_gray_dir, f'frame_{t:03d}.png'))
        Image.fromarray(pred_original, mode='L').save(os.path.join(pred_gray_dir, f'frame_{t:03d}.png'))
    return T


def save_legend_image(output_path):
    """Save a class color legend image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        patches = []
        for i, name in enumerate(KITTI360_CLASS_NAMES):
            color = np.array(KITTI360_VIZ_COLORS[i]) / 255.0
            patches.append(mpatches.Patch(color=color, label=f'{i}: {name}'))
        
        ax.legend(handles=patches, loc='center', fontsize=9, frameon=False)
        ax.axis('off')
        ax.set_title('Semantic Class Legend', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_eval_args():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Semantic Evaluation")
    
    # Required paths
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to Stage 1 checkpoint directory (parent containing checkpoint-XXXXX)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # Model config
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='stabilityai/stable-video-diffusion-img2vid-xt',
                        help='Base SVD model path')
    parser.add_argument('--semantic_vae_checkpoint', type=str,
                        default='/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth',
                        help='Path to semantic VAE checkpoint')
    
    # Data config
    parser.add_argument('--dataset_name', type=str, default='kitti360')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--clip_length', type=int, default=25)
    parser.add_argument('--train_H', type=int, default=192)
    parser.add_argument('--train_W', type=int, default=704)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Eval config
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of video clips to evaluate')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--min_guidance_scale', type=float, default=3.0)
    parser.add_argument('--max_guidance_scale', type=float, default=7.0)
    parser.add_argument('--noise_aug_strength', type=float, default=0.01)
    parser.add_argument('--fps', type=int, default=7)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_cond_bbox_frames', type=int, default=1)
    
    # Output config
    parser.add_argument('--save_frames', action='store_true', default=True,
                        help='Save GT and generated frames as PNG')
    parser.add_argument('--num_save_videos', type=int, default=10,
                        help='Number of first-N clips to save frames for (quick reference)')
    parser.add_argument('--num_worst_videos', type=int, default=20,
                        help='Number of worst-mIoU clips to save frames for after all inference')
    
    return parser.parse_args()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    args = parse_eval_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    tmp_npz_dir = os.path.join(args.output_dir, '_tmp_npz')
    os.makedirs(tmp_npz_dir, exist_ok=True)
    worst_dir = os.path.join(args.output_dir, 'worst_clips')
    os.makedirs(worst_dir, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("=" * 80)
    print("Stage 1 Evaluation: Semantic Prediction Quality")
    print("=" * 80)
    print(f"Checkpoint:       {args.checkpoint_dir}")
    print(f"Output:           {args.output_dir}")
    print(f"Num samples:      {args.num_samples}")
    print(f"Clip length:      {args.clip_length}")
    print(f"Resolution:       {args.train_H}x{args.train_W}")
    print(f"Inference steps:  {args.num_inference_steps}")
    print(f"Device:           {device}")
    print("=" * 80)
    
    # ====================================================================
    # Load Models
    # ====================================================================
    print("\n[1/5] Loading models...")
    
    # Find best checkpoint first, then fall back to latest checkpoint
    checkpoint_dir = args.checkpoint_dir
    
    # Check for best_checkpoint first
    best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint")
    if os.path.exists(best_ckpt_path):
        ckpt_path = best_ckpt_path
        ckpt_step = "best"
        print(f"  Using best checkpoint: best_checkpoint")
    else:
        # Fall back to latest numbered checkpoint
        checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
        if checkpoint_subdirs:
            checkpoint_subdirs = sorted(checkpoint_subdirs, key=lambda x: int(x.split("-")[1]))
            latest_ckpt = checkpoint_subdirs[-1]
            ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
            ckpt_step = int(latest_ckpt.split("-")[1])
            print(f"  Using latest checkpoint: {latest_ckpt} (step {ckpt_step})")
        else:
            print(f"  ERROR: No checkpoints found in {checkpoint_dir}")
            return
    
    # Load base models
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", variant="fp16",
        low_cpu_mem_usage=True, num_frames=args.clip_length
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
    )
    
    # Load DualVAEManager
    from ctrlv.models import DualVAEManager
    vae_manager = DualVAEManager(
        rgb_vae=vae,
        semantic_vae_checkpoint=args.semantic_vae_checkpoint,
        num_semantic_classes=19,
        device=device,
        clip_size=args.clip_length,
        verbose=True
    )
    
    # Load UNet checkpoint weights using from_pretrained (matches save_pretrained in training)
    print(f"  Loading UNet weights from {ckpt_path}/unet/...")
    unet_ckpt_path = os.path.join(ckpt_path, "unet")
    if os.path.exists(unet_ckpt_path):
        load_model = UNetSpatioTemporalConditionModel.from_pretrained(ckpt_path, subfolder="unet")
        unet.register_to_config(**load_model.config)
        unet.load_state_dict(load_model.state_dict())
        del load_model
        print(f"  ✓ Loaded UNet weights via from_pretrained")
    else:
        print(f"  WARNING: No UNet directory found at {unet_ckpt_path}")
        print(f"  Contents of {ckpt_path}: {os.listdir(ckpt_path)}")
    
    # Move to device
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    unet.eval()
    
    # Build pipeline
    pipeline = VideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        feature_extractor=feature_extractor,
        revision=None,
        variant="fp16",
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    # Attach vae_manager so pipeline._encode_vae_condition can use semantic VAE
    pipeline.vae_manager = vae_manager
    print("  ✓ Pipeline constructed (with vae_manager for semantic conditioning)")
    
    # ====================================================================
    # Load Data
    # ====================================================================
    print("\n[2/5] Loading validation data...")
    
    train_dataset, train_loader = get_dataloader(
        args.data_root, args.dataset_name, if_train=False,
        clip_length=args.clip_length,
        batch_size=1, num_workers=args.num_workers,
        data_type='clip', use_default_collate=True, tokenizer=None, shuffle=False,
        if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
        use_segmentation=True, use_preplotted_bbox=True,
        if_last_frame_traj=False, non_overlapping_clips=True,
        return_semantic_ids=True
    )
    
    total_clips = len(train_dataset)
    num_eval = min(args.num_samples, total_clips)
    print(f"  Dataset: {total_clips} clips")
    print(f"  Evaluating: {num_eval} clips  (streaming — no preload)")

    # ====================================================================
    # Run Inference & Compute Metrics
    # ====================================================================
    print("\n[3/5] Running inference and computing metrics...")

    metrics = SemanticMetrics(num_classes=19, ignore_index=255)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    per_sample_results = []

    sample_stream = eval_samples_generator(train_loader)
    for sample_i, sample in enumerate(tqdm(sample_stream, total=num_eval, desc="Evaluating")):
        if sample_i >= num_eval:
            break
        # bbox_img is [T, 3, H, W] float32 RGB semantic visualization (from _semantic_ids_to_rgb)
        # semantic_ids is [T, H, W] int64 trainIDs 0-18
        # 
        # Training loop encodes conditioning via semantic VAE (encode_semantic_from_ids),
        # so we must pass semantic_ids + use_semantic_vae=True to the pipeline.
        # bbox_images is still needed for shape inference in the pipeline.
        bbox_img_rgb = sample['bbox_img'].unsqueeze(0)  # [1, T, 3, H, W] float32 RGB
        
        # Get semantic_ids for semantic VAE conditioning
        semantic_ids_cond = sample['semantic_ids'].unsqueeze(0)  # [1, T, H, W] int64 trainIDs
        
        # Run diffusion inference (output latents for semantic decoding)
        with torch.autocast(str(device).replace(":0", ""), enabled=True):
            result = pipeline(
                sample['image_init'],
                height=train_dataset.train_H, width=train_dataset.train_W,
                bbox_images=bbox_img_rgb,
                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.clip_length,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                noise_aug_strength=args.noise_aug_strength,
                generator=generator,
                output_type='latent',
                num_cond_bbox_frames=args.num_cond_bbox_frames,
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )
        
        # Decode semantic latents
        # IMPORTANT: Cast latents to float32 - semantic VAE decoder is float32,
        # but pipeline outputs fp16 latents. Without this cast, conv2d fails with
        # "Input type (c10::Half) and bias type (float) should be the same"
        latents = result.frames[0].to(torch.float32)  # [T, C, H, W]
        del result
        torch.cuda.empty_cache()
        pred_semantic_ids = vae_manager.decode_semantic(latents)  # [T, H, W] trainIDs 0-18
        pred_np = pred_semantic_ids.cpu().numpy()
        
        # Get GT semantic IDs
        if 'semantic_ids' in sample:
            gt_np = sample['semantic_ids'].cpu().numpy()  # [T, H, W] trainIDs 0-18
        else:
            print(f"  WARNING: Sample {sample_i} has no semantic_ids, skipping")
            continue
        
        # Resize pred to GT size if they differ
        if pred_np.shape != gt_np.shape:
            T_pred, H_pred, W_pred = pred_np.shape
            T_gt, H_gt, W_gt = gt_np.shape
            # Resize pred to match GT
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(1).float()  # [T, 1, H, W]
            pred_resized = F.interpolate(pred_tensor, size=(H_gt, W_gt), mode='nearest')
            pred_np = pred_resized.squeeze(1).numpy().astype(np.int64)
        
        # Update metrics
        metrics.update(pred_np, gt_np)
        
        # Per-sample IoU
        sample_metrics = SemanticMetrics(num_classes=19, ignore_index=255)
        sample_metrics.update(pred_np, gt_np)
        sample_result = sample_metrics.compute()

        # Extract sequence name and start frame from image_paths
        seq_name = 'unknown'
        start_frame = str(sample_i)
        if 'image_paths' in sample and sample['image_paths']:
            first_path = str(sample['image_paths'][0])
            # KITTI-360 path: .../data_2d_raw/2013_05_28_drive_0000_sync/image_00/.../0000000123.png
            path_parts = Path(first_path).parts
            seq_candidates = [p for p in path_parts if 'drive' in p]
            if seq_candidates:
                seq_name = seq_candidates[0]
            start_frame = Path(first_path).stem  # e.g. '0000000123'

        per_sample_results.append({
            'sample_idx': sample_i,
            'sequence': seq_name,
            'start_frame': start_frame,
            'miou': sample_result['miou'],
            'pixel_accuracy': sample_result['overall_accuracy'],
            'iou_per_class': sample_result['iou_per_class'].tolist(),
        })

        # Always save pred/gt as compressed numpy for post-hoc worst-clip selection
        np.savez_compressed(
            os.path.join(tmp_npz_dir, f'clip_{sample_i:04d}.npz'),
            pred=pred_np, gt=gt_np
        )

        # Save frames for the first num_save_videos clips (quick inline reference)
        if args.save_frames and sample_i < args.num_save_videos:
            video_dir = os.path.join(frames_dir, f'video_{sample_i:03d}_{seq_name}_f{start_frame}')
            n_saved = _save_clip_frames(pred_np, gt_np, video_dir)
            print(f"  Saved {n_saved} frames for clip {sample_i} ({seq_name} @ {start_frame})")
    
    # ====================================================================
    # Identify and Save Worst-Performing Clips
    # ====================================================================
    print(f"\n[3b/5] Saving {args.num_worst_videos} worst clips by mIoU...")

    worst_clips = sorted(per_sample_results, key=lambda x: x['miou'])[:args.num_worst_videos]

    worst_report_lines = [
        "Worst Clips Report",
        "=" * 80,
        f"Bottom {args.num_worst_videos} clips by mIoU (ascending order)\n",
        f"{'Rank':<6} {'ClipIdx':<10} {'Sequence':<35} {'StartFrame':<14} {'mIoU':>8} {'PixAcc':>10}",
        "-" * 85,
    ]

    for rank, clip_info in enumerate(worst_clips):
        clip_idx = clip_info['sample_idx']
        npz_path = os.path.join(tmp_npz_dir, f'clip_{clip_idx:04d}.npz')
        seq = clip_info['sequence']
        sf = clip_info['start_frame']
        safe_seq = seq.replace('/', '_')[:30]
        video_dir = os.path.join(
            worst_dir,
            f'rank{rank+1:02d}_miou{clip_info["miou"]*100:.1f}_{safe_seq}_f{sf}'
        )
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            _save_clip_frames(data['pred'], data['gt'], video_dir)
            # Per-class IoU for this clip
            iou_arr = np.array(clip_info['iou_per_class'])
            worst_class_idx = int(np.nanargmin(iou_arr))
            worst_class_iou = iou_arr[worst_class_idx]
            worst_class_name = KITTI360_CLASS_NAMES[worst_class_idx]
            worst_report_lines.append(
                f"  {rank+1:<4} {clip_idx:<10} {seq:<35} {sf:<14} "
                f"{clip_info['miou']*100:>7.2f}% {clip_info['pixel_accuracy']*100:>9.2f}%"
                f"  [worst class: {worst_class_name} {worst_class_iou*100:.1f}%]"
            )
            print(f"  [{rank+1:2d}/{args.num_worst_videos}] clip {clip_idx:04d} "
                  f"({seq}) frame {sf} → mIoU={clip_info['miou']*100:.1f}%")
        else:
            print(f"  WARNING: npz not found for clip {clip_idx}")

    # Append per-clip table sorted by mIoU (all clips)
    worst_report_lines += [
        "\n" + "=" * 80,
        "All Clips Sorted by mIoU (ascending)",
        f"{'Rank':<6} {'ClipIdx':<10} {'Sequence':<35} {'StartFrame':<14} {'mIoU':>8} {'PixAcc':>10}",
        "-" * 85,
    ]
    for rank, clip_info in enumerate(sorted(per_sample_results, key=lambda x: x['miou'])):
        worst_report_lines.append(
            f"  {rank+1:<4} {clip_info['sample_idx']:<10} {clip_info['sequence']:<35} "
            f"{clip_info['start_frame']:<14} {clip_info['miou']*100:>7.2f}% "
            f"{clip_info['pixel_accuracy']*100:>9.2f}%"
        )

    worst_report_path = os.path.join(args.output_dir, 'worst_clips_report.txt')
    with open(worst_report_path, 'w') as f:
        f.write('\n'.join(worst_report_lines) + '\n')
    print(f"  Saved worst clips report: {worst_report_path}")
    print(f"  Worst clip frames: {worst_dir}/")

    # Clean up temporary npz files
    import shutil
    shutil.rmtree(tmp_npz_dir, ignore_errors=True)

    # ====================================================================
    # Compute Final Metrics
    # ====================================================================
    print("\n[4/5] Computing final metrics...")
    
    results = metrics.compute()
    
    # ====================================================================
    # Print & Save Results
    # ====================================================================
    print("\n[5/5] Saving results...")
    
    print("\n" + "=" * 80)
    print(f"STAGE 1 EVALUATION RESULTS (Checkpoint: step {ckpt_step})")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Mean IoU (mIoU)':<30} {results['miou']*100:>10.2f}%")
    print(f"{'Overall Pixel Accuracy':<30} {results['overall_accuracy']*100:>10.2f}%")
    print(f"{'Mean Class Accuracy':<30} {results['mean_accuracy']*100:>10.2f}%")
    print(f"{'Frequency-Weighted IoU':<30} {results['fwiou']*100:>10.2f}%")
    
    print(f"\n{'Per-Class Results':}")
    print(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'GT Pix':>12} {'Pred Pix':>12}")
    print("-" * 80)
    
    for i, name in enumerate(KITTI360_CLASS_NAMES):
        iou = results['iou_per_class'][i]
        prec = results['precision_per_class'][i]
        recall = results['recall_per_class'][i]
        f1 = results['f1_per_class'][i]
        gt_pix = results['gt_pixels_per_class'][i]
        pred_pix = results['pred_pixels_per_class'][i]
        
        iou_str = f"{iou*100:.2f}%" if not np.isnan(iou) else "N/A"
        prec_str = f"{prec*100:.2f}%" if not np.isnan(prec) else "N/A"
        recall_str = f"{recall*100:.2f}%" if not np.isnan(recall) else "N/A"
        f1_str = f"{f1*100:.2f}%" if not np.isnan(f1) else "N/A"
        
        print(f"  {name:<18} {iou_str:>8} {prec_str:>8} {recall_str:>8} {f1_str:>8} {gt_pix:>12,d} {pred_pix:>12,d}")
    
    print("-" * 80)
    
    # Per-sample summary (sorted by mIoU ascending so bad clips stand out)
    sorted_results = sorted(per_sample_results, key=lambda x: x['miou'])
    print(f"\nPer-Sample Summary (sorted by mIoU ascending — worst first):")
    print(f"{'Rank':<6} {'ClipIdx':<10} {'Sequence':<35} {'StartFrame':<14} {'mIoU':>8} {'PixAcc':>10}")
    print("-" * 85)
    for rank, r in enumerate(sorted_results):
        print(f"  {rank+1:<4} {r['sample_idx']:<10} {r['sequence']:<35} "
              f"{r['start_frame']:<14} {r['miou']*100:>7.2f}% {r['pixel_accuracy']*100:>9.2f}%")

    avg_sample_miou = np.mean([r['miou'] for r in per_sample_results])
    avg_sample_acc = np.mean([r['pixel_accuracy'] for r in per_sample_results])
    miou_std = np.std([r['miou'] for r in per_sample_results])
    print(f"\n  Average mIoU: {avg_sample_miou*100:.2f}% ± {miou_std*100:.2f}%  |  "
          f"Avg Pixel Acc: {avg_sample_acc*100:.2f}%")
    
    print("\n" + "=" * 80)
    
    # Save JSON results
    json_results = {
        'checkpoint_step': ckpt_step,
        'checkpoint_dir': args.checkpoint_dir,
        'num_samples': args.num_samples,
        'clip_length': args.clip_length,
        'resolution': f'{args.train_H}x{args.train_W}',
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': f'{args.min_guidance_scale}-{args.max_guidance_scale}',
        'metrics': {
            'miou': float(results['miou']),
            'overall_pixel_accuracy': float(results['overall_accuracy']),
            'mean_class_accuracy': float(results['mean_accuracy']),
            'frequency_weighted_iou': float(results['fwiou']),
        },
        'per_class': {},
        'per_sample': [
            {k: v for k, v in r.items() if k != 'iou_per_class'}
            for r in per_sample_results
        ],
    }
    
    for i, name in enumerate(KITTI360_CLASS_NAMES):
        json_results['per_class'][name] = {
            'iou': float(results['iou_per_class'][i]) if not np.isnan(results['iou_per_class'][i]) else None,
            'precision': float(results['precision_per_class'][i]) if not np.isnan(results['precision_per_class'][i]) else None,
            'recall': float(results['recall_per_class'][i]) if not np.isnan(results['recall_per_class'][i]) else None,
            'f1': float(results['f1_per_class'][i]) if not np.isnan(results['f1_per_class'][i]) else None,
            'gt_pixels': int(results['gt_pixels_per_class'][i]),
            'pred_pixels': int(results['pred_pixels_per_class'][i]),
        }
    
    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ JSON results saved to: {json_path}")
    
    # Save confusion matrix image
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    save_confusion_matrix_image(results['confusion_matrix'], KITTI360_CLASS_NAMES, cm_path)
    
    # Save legend
    legend_path = os.path.join(args.output_dir, 'class_legend.png')
    save_legend_image(legend_path)
    
    # Save raw confusion matrix as numpy
    cm_npy_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
    np.save(cm_npy_path, results['confusion_matrix'])
    print(f"✓ Confusion matrix saved to: {cm_npy_path}")
    
    # Save a summary text file
    summary_path = os.path.join(args.output_dir, 'eval_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage 1 Evaluation Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Checkpoint: step {ckpt_step}\n")
        f.write(f"Checkpoint dir: {args.checkpoint_dir}\n")
        f.write(f"Num samples: {args.num_samples}\n")
        f.write(f"Clip length: {args.clip_length}\n")
        f.write(f"Resolution: {args.train_H}x{args.train_W}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  mIoU:              {results['miou']*100:.2f}%\n")
        f.write(f"  Pixel Accuracy:    {results['overall_accuracy']*100:.2f}%\n")
        f.write(f"  Mean Accuracy:     {results['mean_accuracy']*100:.2f}%\n")
        f.write(f"  FW-IoU:            {results['fwiou']*100:.2f}%\n\n")
        f.write(f"Per-Class IoU:\n")
        for i, name in enumerate(KITTI360_CLASS_NAMES):
            iou = results['iou_per_class'][i]
            iou_str = f"{iou*100:.2f}%" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<18} {iou_str}\n")
        f.write(f"\nPer-Clip mIoU Summary (sorted worst first):\n")
        f.write(f"  {'Rank':<6} {'ClipIdx':<10} {'Sequence':<35} {'StartFrame':<14} {'mIoU':>8} {'PixAcc':>10}\n")
        f.write(f"  {'-'*80}\n")
        for rank, r in enumerate(sorted(per_sample_results, key=lambda x: x['miou'])):
            f.write(f"  {rank+1:<6} {r['sample_idx']:<10} {r['sequence']:<35} "
                    f"{r['start_frame']:<14} {r['miou']*100:>7.2f}% {r['pixel_accuracy']*100:>9.2f}%\n")
        miou_std = np.std([r['miou'] for r in per_sample_results])
        f.write(f"\n  Average mIoU: {avg_sample_miou*100:.2f}% ± {miou_std*100:.2f}%\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    print(f"\n✓ Frames saved to: {frames_dir}/")
    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
