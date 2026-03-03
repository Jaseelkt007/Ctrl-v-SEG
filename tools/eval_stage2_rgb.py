"""
Stage 2 Evaluation: Semantic-to-RGB Video Generation

Evaluates Stage 2 ControlNet model that generates RGB video from semantic conditioning.

Metrics:
  - DRN-based mIoU: Run DRN segmentation on generated RGB, compare with GT semantic labels
  - Per-class IoU, pixel accuracy, precision, recall, F1
  - FID: Frechet Inception Distance between generated and GT frames
  - Per-frame and per-video metrics

Outputs:
  - JSON results with all metrics
  - Text summary
  - Saved GT RGB frames, generated RGB frames, GT semantic frames, side-by-side comparisons
  - Confusion matrix from DRN predictions
"""

from accelerate.utils import write_basic_config
write_basic_config()
import warnings
import argparse
import json
import logging
import os
import sys

import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as T
from einops import rearrange

from accelerate.logging import get_logger

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import get_dataloader, get_n_training_samples
    from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
    from ctrlv.pipelines import StableVideoControlPipeline
    from ctrlv.utils.semantic_preprocessing import (
        KITTI360_LABEL_MAPPING,
        KITTI360_CLASS_NAMES,
        KITTI360_VIZ_COLORS,
        semantic_ids_to_viz_rgb,
    )

logger = get_logger(__name__, log_level="INFO")

# ============================================================================
# DRN Model Loading
# ============================================================================

def load_drn_model(drn_dir, checkpoint_path, num_classes=19, arch='drn_d_105'):
    """Load pretrained DRN segmentation model."""
    sys.path.insert(0, drn_dir)
    import drn as drn_module
    from segment import DRNSeg

    model = DRNSeg(arch, num_classes, pretrained_model=None, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location='cpu')["state_dict"]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    return model


def drn_predict(model, rgb_frame_np, info_json_path):
    """
    Run DRN inference on a single RGB frame.
    
    Args:
        model: DRN model
        rgb_frame_np: [H, W, 3] uint8 numpy array
        info_json_path: Path to info.json with mean/std
    
    Returns:
        pred_trainids: [H, W] numpy array of trainIDs (0-18)
    """
    with open(info_json_path) as f:
        info = json.load(f)
    
    normalize = T.Normalize(mean=info['mean'], std=info['std'])
    transform = T.Compose([T.ToTensor(), normalize])
    
    img_pil = Image.fromarray(rgb_frame_np)
    img_tensor = transform(img_pil).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model(img_tensor)[0]
    _, pred = torch.max(output, 1)
    return pred.squeeze().cpu().numpy()


def drn_predict_batch(model, rgb_frames_np, info_json_path):
    """
    Run DRN inference on multiple frames.
    
    Args:
        model: DRN model
        rgb_frames_np: [T, H, W, 3] uint8 numpy array
        info_json_path: path to info.json
    
    Returns:
        preds: [T, H, W] numpy array of trainIDs
    """
    with open(info_json_path) as f:
        info = json.load(f)
    
    normalize = T.Normalize(mean=info['mean'], std=info['std'])
    transform = T.Compose([T.ToTensor(), normalize])
    
    preds = []
    for t in range(rgb_frames_np.shape[0]):
        img_pil = Image.fromarray(rgb_frames_np[t])
        img_tensor = transform(img_pil).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img_tensor)[0]
        _, pred = torch.max(output, 1)
        preds.append(pred.squeeze().cpu().numpy())
    
    return np.stack(preds, axis=0)


# ============================================================================
# Metrics (same as Stage 1 but reusable)
# ============================================================================

class SemanticMetrics:
    """Accumulates predictions and computes semantic segmentation metrics."""
    
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        if pred.ndim == 3:
            for t in range(pred.shape[0]):
                self._update_single(pred[t], gt[t])
        else:
            self._update_single(pred, gt)
    
    def _update_single(self, pred: np.ndarray, gt: np.ndarray):
        valid_mask = (gt != self.ignore_index) & (gt < self.num_classes)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        for g, p in zip(gt_valid.flatten(), pred_valid.flatten()):
            self.confusion_matrix[g, p] += 1
    
    def compute(self):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        gt_per_class = cm.sum(axis=1)
        pred_per_class = cm.sum(axis=0)
        
        denominator = tp + fp + fn
        iou_per_class = np.where(denominator > 0, tp / denominator, np.nan)
        acc_per_class = np.where(gt_per_class > 0, tp / gt_per_class, np.nan)
        precision_per_class = np.where(pred_per_class > 0, tp / pred_per_class, np.nan)
        f1_per_class = np.where(
            (precision_per_class + acc_per_class) > 0,
            2 * precision_per_class * acc_per_class / (precision_per_class + acc_per_class),
            np.nan
        )
        
        valid_classes = ~np.isnan(iou_per_class)
        miou = np.nanmean(iou_per_class) if valid_classes.any() else 0.0
        total_correct = tp.sum()
        total_pixels = gt_per_class.sum()
        overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        mean_accuracy = np.nanmean(acc_per_class) if valid_classes.any() else 0.0
        freq = gt_per_class / total_pixels if total_pixels > 0 else np.zeros(self.num_classes)
        fw_iou = np.nansum(freq * iou_per_class)
        
        return {
            'miou': float(miou),
            'overall_accuracy': float(overall_accuracy),
            'mean_class_accuracy': float(mean_accuracy),
            'frequency_weighted_iou': float(fw_iou),
            'iou_per_class': iou_per_class,
            'accuracy_per_class': acc_per_class,
            'precision_per_class': precision_per_class,
            'f1_per_class': f1_per_class,
            'gt_per_class': gt_per_class,
            'pred_per_class': pred_per_class,
        }

    def save_confusion_matrix(self, output_path, class_names):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            cm = self.confusion_matrix.astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.where(row_sums > 0, cm / row_sums, 0)
            
            present = cm.sum(axis=1) > 0
            indices = np.where(present)[0]
            if len(indices) == 0:
                return
            
            cm_sub = cm_normalized[np.ix_(indices, indices)]
            names_sub = [class_names[i] for i in indices]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
            ax.set_xticks(range(len(names_sub)))
            ax.set_yticks(range(len(names_sub)))
            ax.set_xticklabels(names_sub, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(names_sub, fontsize=8)
            ax.set_xlabel('Predicted (DRN)')
            ax.set_ylabel('Ground Truth')
            ax.set_title('Stage 2 Confusion Matrix (DRN on Generated RGB)')
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not save confusion matrix plot: {e}")


# ============================================================================
# Visualization helpers
# ============================================================================

def save_rgb_frame(frame_np, output_path):
    """Save a [H, W, 3] uint8 numpy array as PNG."""
    Image.fromarray(frame_np).save(output_path)

def save_comparison_rgb_sem(gt_rgb, gen_rgb, gt_sem_viz, drn_sem_viz, output_path):
    """Save a 4-panel comparison: GT RGB | Generated RGB | GT Semantic | DRN Predicted Semantic."""
    H, W = gt_rgb.shape[:2]
    canvas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    canvas[:H, :W] = gt_rgb
    canvas[:H, W:2*W] = gen_rgb
    canvas[H:2*H, :W] = gt_sem_viz
    canvas[H:2*H, W:2*W] = drn_sem_viz
    Image.fromarray(canvas).save(output_path)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Stage 2 Evaluation: Semantic-to-RGB")
    
    # Model paths
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Stage 2 ControlNet checkpoint directory')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='stabilityai/stable-video-diffusion-img2vid-xt')
    parser.add_argument('--semantic_vae_checkpoint', type=str,
                        default='/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth')
    
    # DRN paths
    parser.add_argument('--drn_dir', type=str, default='/usrhomes/s1492/drn',
                        help='Path to DRN repository')
    parser.add_argument('--drn_checkpoint', type=str,
                        default='/usrhomes/s1492/drn/KITTI360_checkpoints/checkpoint_030.pth.tar')
    parser.add_argument('--drn_info_json', type=str,
                        default='/usrhomes/s1492/drn/CTRLV_BBOX/info.json')
    parser.add_argument('--drn_arch', type=str, default='drn_d_105')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='kitti360')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--clip_length', type=int, default=25)
    parser.add_argument('--train_H', type=int, default=192)
    parser.add_argument('--train_W', type=int, default=704)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Inference
    parser.add_argument('--num_samples', type=int, default=15)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--min_guidance_scale', type=float, default=1.0)
    parser.add_argument('--max_guidance_scale', type=float, default=3.0)
    parser.add_argument('--conditioning_scale', type=float, default=1.0)
    parser.add_argument('--noise_aug_strength', type=float, default=0.01)
    parser.add_argument('--fps', type=int, default=7)
    parser.add_argument('--seed', type=int, default=1234)
    
    # Saving
    parser.add_argument('--save_frames', action='store_true', default=True)
    parser.add_argument('--num_save_videos', type=int, default=10)
    
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_eval_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("=" * 80)
    print("Stage 2 Evaluation: Semantic-to-RGB Video Generation")
    print("=" * 80)
    
    # Find latest checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    if checkpoint_subdirs:
        checkpoint_subdirs = sorted(checkpoint_subdirs, key=lambda x: int(x.split("-")[1]))
        latest_ckpt = checkpoint_subdirs[-1]
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        ckpt_step = int(latest_ckpt.split("-")[1])
    else:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    print(f"Checkpoint:       {checkpoint_dir}")
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
    print("\n[1/6] Loading models...")
    print(f"  Using checkpoint: {latest_ckpt} (step {ckpt_step})")
    
    # Load ControlNet from checkpoint
    ctrlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="control_net")
    print(f"  ✓ ControlNet loaded from {ckpt_path}/control_net/")
    
    # Load UNet from checkpoint
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        ckpt_path, subfolder="unet",
        low_cpu_mem_usage=True, num_frames=args.clip_length
    )
    print(f"  ✓ UNet loaded from {ckpt_path}/unet/")
    
    # Load VAE, image encoder, feature extractor from base SVD
    from diffusers.models import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant="fp16"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=None, variant="fp16"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=None
    )
    
    # Load DualVAEManager for semantic conditioning
    from ctrlv.models import DualVAEManager
    vae_manager = DualVAEManager(
        rgb_vae=vae,
        semantic_vae_checkpoint=args.semantic_vae_checkpoint,
        num_semantic_classes=19,
        device=device,
        clip_size=args.clip_length,
        verbose=True
    )
    
    # Move to device
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    ctrlnet.to(device, dtype=weight_dtype)
    ctrlnet.eval()
    unet.eval()
    
    # Build pipeline
    pipeline = StableVideoControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=ctrlnet,
        image_encoder=image_encoder,
        vae=vae,
        feature_extractor=feature_extractor,
        revision=None,
        variant="fp16",
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    # Attach vae_manager for semantic VAE encoding of conditioning
    pipeline.vae_manager = vae_manager
    print("  ✓ Pipeline constructed (with vae_manager for semantic conditioning)")
    
    # Load DRN model
    print(f"\n  Loading DRN model ({args.drn_arch})...")
    drn_model = load_drn_model(args.drn_dir, args.drn_checkpoint, num_classes=19, arch=args.drn_arch)
    print(f"  ✓ DRN model loaded from {args.drn_checkpoint}")
    
    # ====================================================================
    # Load Data
    # ====================================================================
    print("\n[2/6] Loading validation data...")
    
    train_dataset, train_loader = get_dataloader(
        args.data_root, args.dataset_name, if_train=True,
        clip_length=args.clip_length,
        batch_size=1, num_workers=args.num_workers,
        data_type='clip', use_default_collate=True, tokenizer=None, shuffle=False,
        if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
        use_segmentation=True, use_preplotted_bbox=True,
        if_last_frame_traj=False, non_overlapping_clips=True,
        return_semantic_ids=True
    )
    
    print(f"  Dataset: {len(train_dataset)} clips")
    print(f"  Evaluating: {args.num_samples} clips")
    
    # Collect samples
    demo_samples = get_n_training_samples(train_loader, args.num_samples, show_progress=True)
    print(f"  ✓ Collected {len(demo_samples)} validation samples")
    
    # ====================================================================
    # Run Inference & Compute Metrics
    # ====================================================================
    print("\n[3/6] Running inference...")
    
    drn_metrics = SemanticMetrics(num_classes=19, ignore_index=255)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    per_sample_results = []
    all_gt_frames = []
    all_gen_frames = []
    
    # Build reverse mapping: trainID -> original KITTI-360 ID
    trainid_to_original = {train_id: kitti_id for kitti_id, train_id in KITTI360_LABEL_MAPPING.items()}
    
    for sample_i, sample in enumerate(tqdm(demo_samples, desc="Evaluating")):
        # bbox_img is [T, 3, H, W] float32 RGB semantic visualization (from _semantic_ids_to_rgb)
        # semantic_ids is [T, H, W] int64 trainIDs 0-18
        bbox_img_rgb = sample['bbox_img'].unsqueeze(0)  # [1, T, 3, H, W] float32 RGB
        semantic_ids_cond = sample['semantic_ids'].unsqueeze(0)  # [1, T, H, W] int64 trainIDs
        
        # Run ControlNet pipeline (semantic -> RGB)
        with torch.autocast(str(device).replace(":0", ""), enabled=True):
            result = pipeline(
                sample['image_init'],
                cond_images=bbox_img_rgb,
                height=train_dataset.train_H, width=train_dataset.train_W,
                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.clip_length,
                control_condition_scale=args.conditioning_scale,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                noise_aug_strength=args.noise_aug_strength,
                generator=generator,
                output_type='pt',
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )
        
        # Generated RGB frames: [T, C, H, W] float [0, 1]
        gen_frames_pt = result.frames[0]  # [T, 3, H, W]
        gen_frames_np = (gen_frames_pt.detach().cpu().numpy() * 255).astype(np.uint8)  # [T, 3, H, W]
        gen_frames_hwc = np.transpose(gen_frames_np, (0, 2, 3, 1))  # [T, H, W, 3]
        
        # GT RGB frames: from gt_clip_np [T, 3, H, W] uint8
        gt_frames_np = sample['gt_clip_np']  # [T, 3, H, W] uint8
        gt_frames_hwc = np.transpose(gt_frames_np, (0, 2, 3, 1))  # [T, H, W, 3]
        
        # GT semantic trainIDs
        gt_sem_np = sample['semantic_ids'].cpu().numpy()  # [T, H, W] int64 trainIDs
        
        # Run DRN on generated RGB frames -> predicted semantic trainIDs
        drn_pred = drn_predict_batch(drn_model, gen_frames_hwc, args.drn_info_json)  # [T, H, W]
        
        # Resize DRN predictions if needed (DRN output size may differ)
        if drn_pred.shape[1:] != gt_sem_np.shape[1:]:
            drn_pred_t = torch.from_numpy(drn_pred).unsqueeze(1).float()
            drn_pred_t = F.interpolate(drn_pred_t, size=gt_sem_np.shape[1:], mode='nearest')
            drn_pred = drn_pred_t.squeeze(1).numpy().astype(np.int64)
        
        # Update DRN metrics
        drn_metrics.update(drn_pred, gt_sem_np)
        
        # Per-sample DRN mIoU
        sample_drn_metrics = SemanticMetrics(num_classes=19, ignore_index=255)
        sample_drn_metrics.update(drn_pred, gt_sem_np)
        sample_result = sample_drn_metrics.compute()
        
        per_sample_results.append({
            'sample_idx': sample_i,
            'drn_miou': sample_result['miou'],
            'drn_pixel_accuracy': sample_result['overall_accuracy'],
        })
        
        # Collect frames for FID (flatten all frames)
        all_gt_frames.append(gt_frames_hwc)
        all_gen_frames.append(gen_frames_hwc)
        
        # Save frames
        if args.save_frames and sample_i < args.num_save_videos:
            video_dir = os.path.join(frames_dir, f'video_{sample_i:03d}')
            gt_rgb_dir = os.path.join(video_dir, 'gt_rgb')
            gen_rgb_dir = os.path.join(video_dir, 'gen_rgb')
            gt_sem_dir = os.path.join(video_dir, 'gt_semantic')
            drn_sem_dir = os.path.join(video_dir, 'drn_semantic')
            gt_sem_gray_dir = os.path.join(video_dir, 'gt_semantic_grayscale')
            drn_sem_gray_dir = os.path.join(video_dir, 'drn_semantic_grayscale')
            compare_dir = os.path.join(video_dir, 'comparison')
            for d in [gt_rgb_dir, gen_rgb_dir, gt_sem_dir, drn_sem_dir,
                       gt_sem_gray_dir, drn_sem_gray_dir, compare_dir]:
                os.makedirs(d, exist_ok=True)
            
            T = min(gen_frames_hwc.shape[0], gt_frames_hwc.shape[0])
            for t in range(T):
                # GT and generated RGB
                save_rgb_frame(gt_frames_hwc[t], os.path.join(gt_rgb_dir, f'frame_{t:03d}.png'))
                save_rgb_frame(gen_frames_hwc[t], os.path.join(gen_rgb_dir, f'frame_{t:03d}.png'))
                
                # GT semantic (colorized + grayscale)
                gt_sem_viz = semantic_ids_to_viz_rgb(gt_sem_np[t])
                Image.fromarray(gt_sem_viz).save(os.path.join(gt_sem_dir, f'frame_{t:03d}.png'))
                gt_original = np.zeros_like(gt_sem_np[t], dtype=np.uint8)
                for train_id, orig_id in trainid_to_original.items():
                    gt_original[gt_sem_np[t] == train_id] = orig_id
                gt_original[gt_sem_np[t] == 255] = 0
                Image.fromarray(gt_original, mode='L').save(os.path.join(gt_sem_gray_dir, f'frame_{t:03d}.png'))
                
                # DRN predicted semantic (colorized + grayscale)
                drn_sem_viz = semantic_ids_to_viz_rgb(drn_pred[t])
                Image.fromarray(drn_sem_viz).save(os.path.join(drn_sem_dir, f'frame_{t:03d}.png'))
                drn_original = np.zeros_like(drn_pred[t], dtype=np.uint8)
                for train_id, orig_id in trainid_to_original.items():
                    drn_original[drn_pred[t] == train_id] = orig_id
                Image.fromarray(drn_original, mode='L').save(os.path.join(drn_sem_gray_dir, f'frame_{t:03d}.png'))
                
                # 4-panel comparison
                save_comparison_rgb_sem(
                    gt_frames_hwc[t], gen_frames_hwc[t],
                    gt_sem_viz, drn_sem_viz,
                    os.path.join(compare_dir, f'frame_{t:03d}.png')
                )
            
            print(f"  Saved {T} frames for video {sample_i}")
    
    # ====================================================================
    # Compute FID
    # ====================================================================
    print("\n[4/6] Computing FID...")
    
    fid_score = None
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        
        # Save all GT and generated frames to temp dirs for FID calculation
        fid_gt_dir = os.path.join(args.output_dir, '_fid_gt')
        fid_gen_dir = os.path.join(args.output_dir, '_fid_gen')
        os.makedirs(fid_gt_dir, exist_ok=True)
        os.makedirs(fid_gen_dir, exist_ok=True)
        
        frame_idx = 0
        for gt_clip, gen_clip in zip(all_gt_frames, all_gen_frames):
            for t in range(gt_clip.shape[0]):
                Image.fromarray(gt_clip[t]).save(os.path.join(fid_gt_dir, f'{frame_idx:06d}.png'))
                Image.fromarray(gen_clip[t]).save(os.path.join(fid_gen_dir, f'{frame_idx:06d}.png'))
                frame_idx += 1
        
        fid_score = calculate_fid_given_paths(
            [fid_gt_dir, fid_gen_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        print(f"  ✓ FID: {fid_score:.2f}")
        
        # Cleanup temp dirs
        import shutil
        shutil.rmtree(fid_gt_dir, ignore_errors=True)
        shutil.rmtree(fid_gen_dir, ignore_errors=True)
        
    except ImportError:
        print("  WARNING: pytorch_fid not installed. Skipping FID calculation.")
        print("  Install with: pip install pytorch-fid")
    except Exception as e:
        print(f"  WARNING: FID calculation failed: {e}")
    
    # ====================================================================
    # Compute Final DRN Metrics
    # ====================================================================
    print("\n[5/6] Computing final DRN metrics...")
    
    drn_results = drn_metrics.compute()
    
    # ====================================================================
    # Print & Save Results
    # ====================================================================
    print("\n[6/6] Saving results...")
    
    print("\n" + "=" * 80)
    print(f"STAGE 2 EVALUATION RESULTS (Checkpoint: step {ckpt_step})")
    print("=" * 80)
    
    print(f"\nDRN Semantic Segmentation on Generated RGB:")
    print(f"  mIoU (DRN):        {drn_results['miou']*100:.2f}%")
    print(f"  Pixel Accuracy:    {drn_results['overall_accuracy']*100:.2f}%")
    print(f"  Mean Accuracy:     {drn_results['mean_class_accuracy']*100:.2f}%")
    print(f"  FW-IoU:            {drn_results['frequency_weighted_iou']*100:.2f}%")
    
    if fid_score is not None:
        print(f"\nImage Quality:")
        print(f"  FID:               {fid_score:.2f}")
    
    print(f"\nPer-Class IoU (DRN on Generated RGB vs GT Semantic):")
    class_names = KITTI360_CLASS_NAMES
    for i, name in enumerate(class_names):
        iou = drn_results['iou_per_class'][i]
        if np.isnan(iou):
            print(f"  {name:<18} N/A")
        else:
            print(f"  {name:<18} {iou*100:.2f}%")
    
    print(f"\nPer-Sample DRN mIoU:")
    print(f"  {'Sample':<10} {'DRN mIoU':>10} {'DRN PixAcc':>12}")
    print(f"  {'-'*35}")
    for sr in per_sample_results:
        print(f"  {sr['sample_idx']:<10} {sr['drn_miou']*100:>9.2f}% {sr['drn_pixel_accuracy']*100:>11.2f}%")
    avg_miou = np.mean([sr['drn_miou'] for sr in per_sample_results])
    avg_pix = np.mean([sr['drn_pixel_accuracy'] for sr in per_sample_results])
    print(f"  {'Average':<10} {avg_miou*100:>9.2f}% {avg_pix*100:>11.2f}%")
    
    print("\n" + "=" * 80)
    
    # Save JSON
    json_results = {
        'checkpoint': ckpt_path,
        'checkpoint_step': ckpt_step,
        'num_samples': len(demo_samples),
        'clip_length': args.clip_length,
        'resolution': f'{args.train_H}x{args.train_W}',
        'drn_metrics': {
            'miou': drn_results['miou'],
            'overall_pixel_accuracy': drn_results['overall_accuracy'],
            'mean_class_accuracy': drn_results['mean_class_accuracy'],
            'frequency_weighted_iou': drn_results['frequency_weighted_iou'],
        },
        'fid': fid_score,
        'per_class': {},
        'per_sample': per_sample_results,
    }
    for i, name in enumerate(class_names):
        json_results['per_class'][name] = {
            'iou': None if np.isnan(drn_results['iou_per_class'][i]) else float(drn_results['iou_per_class'][i]),
            'precision': None if np.isnan(drn_results['precision_per_class'][i]) else float(drn_results['precision_per_class'][i]),
            'recall': None if np.isnan(drn_results['accuracy_per_class'][i]) else float(drn_results['accuracy_per_class'][i]),
            'f1': None if np.isnan(drn_results['f1_per_class'][i]) else float(drn_results['f1_per_class'][i]),
            'gt_pixels': int(drn_results['gt_per_class'][i]),
            'pred_pixels': int(drn_results['pred_per_class'][i]),
        }
    
    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ JSON results saved to: {json_path}")
    
    # Save confusion matrix
    drn_metrics.save_confusion_matrix(
        os.path.join(args.output_dir, 'confusion_matrix_drn.png'), class_names
    )
    np.save(os.path.join(args.output_dir, 'confusion_matrix_drn.npy'), drn_metrics.confusion_matrix)
    print(f"✓ Confusion matrix saved")
    
    # Save summary text
    summary_path = os.path.join(args.output_dir, 'eval_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage 2 Evaluation Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Checkpoint: step {ckpt_step}\n")
        f.write(f"Checkpoint dir: {checkpoint_dir}\n")
        f.write(f"Num samples: {len(demo_samples)}\n")
        f.write(f"Clip length: {args.clip_length}\n")
        f.write(f"Resolution: {args.train_H}x{args.train_W}\n\n")
        f.write(f"DRN Metrics (on Generated RGB):\n")
        f.write(f"  mIoU:              {drn_results['miou']*100:.2f}%\n")
        f.write(f"  Pixel Accuracy:    {drn_results['overall_accuracy']*100:.2f}%\n")
        f.write(f"  Mean Accuracy:     {drn_results['mean_class_accuracy']*100:.2f}%\n")
        f.write(f"  FW-IoU:            {drn_results['frequency_weighted_iou']*100:.2f}%\n\n")
        if fid_score is not None:
            f.write(f"FID: {fid_score:.2f}\n\n")
        f.write(f"Per-Class IoU (DRN):\n")
        for i, name in enumerate(class_names):
            iou = drn_results['iou_per_class'][i]
            f.write(f"  {name:<18} {'N/A' if np.isnan(iou) else f'{iou*100:.2f}%'}\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    print(f"\n✓ Frames saved to: {frames_dir}")
    print(f"\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
