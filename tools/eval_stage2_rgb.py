"""
Stage 2 Evaluation: Semantic-to-RGB Video Generation

Evaluates the Stage 2 ControlNet model (semantic maps → RGB video).

Metrics computed in a single pass:
  - DRN mIoU       : DRN segmentation on generated RGB vs GT semantic labels
  - FID            : Frechet Inception Distance (torch-fidelity, Inception-v3)
  - FVD-I3D        : Frechet Video Distance via cdfvd I3D backbone
  - FVD-VideoMAE   : Frechet Video Distance via cdfvd VideoMAE backbone
  - LPIPS          : Learned Perceptual Image Patch Similarity (AlexNet)
  - SSIM           : Structural Similarity Index
  - PSNR           : Peak Signal-to-Noise Ratio

Dataset: KITTI-360 official validation split, non-overlapping clips.

Outputs:
  - eval_results.json       : all metrics in structured JSON
  - eval_summary.txt        : human-readable summary
  - confusion_matrix_drn.png / .npy
"""

from accelerate.utils import write_basic_config
write_basic_config()
import warnings
import argparse
import json
import logging
import os
import sys
import shutil
import tempfile

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from accelerate.logging import get_logger

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import get_dataloader, eval_samples_generator
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
# DRN helpers
# ============================================================================

def load_drn_model(drn_dir, checkpoint_path, num_classes=19, arch='drn_d_105'):
    sys.path.insert(0, drn_dir)
    import drn as drn_module
    from segment import DRNSeg
    model = DRNSeg(arch, num_classes, pretrained_model=None, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location='cpu')["state_dict"]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model.cuda().eval()


def drn_predict_batch(model, rgb_frames_np, info_json_path):
    """rgb_frames_np: [T, H, W, 3] uint8 → preds: [T, H, W] trainIDs"""
    with open(info_json_path) as f:
        info = json.load(f)
    normalize = T.Normalize(mean=info['mean'], std=info['std'])
    transform = T.Compose([T.ToTensor(), normalize])
    preds = []
    for t in range(rgb_frames_np.shape[0]):
        img_tensor = transform(Image.fromarray(rgb_frames_np[t])).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img_tensor)[0]
        _, pred = torch.max(output, 1)
        preds.append(pred.squeeze().cpu().numpy())
    return np.stack(preds, axis=0)


# ============================================================================
# Semantic metrics
# ============================================================================

class SemanticMetrics:
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

    def _update_single(self, pred, gt):
        valid_mask = (gt != self.ignore_index) & (gt < self.num_classes)
        for g, p in zip(gt[valid_mask].flatten(), pred[valid_mask].flatten()):
            self.confusion_matrix[g, p] += 1

    def compute(self):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        gt_per_class   = cm.sum(axis=1)
        pred_per_class = cm.sum(axis=0)
        denom = tp + fp + fn
        iou_per_class       = np.where(denom > 0,           tp / denom,           np.nan)
        acc_per_class       = np.where(gt_per_class > 0,    tp / gt_per_class,    np.nan)
        precision_per_class = np.where(pred_per_class > 0,  tp / pred_per_class,  np.nan)
        f1_per_class = np.where(
            (precision_per_class + acc_per_class) > 0,
            2 * precision_per_class * acc_per_class / (precision_per_class + acc_per_class),
            np.nan
        )
        valid = ~np.isnan(iou_per_class)
        total_pixels = gt_per_class.sum()
        freq = gt_per_class / total_pixels if total_pixels > 0 else np.zeros(self.num_classes)
        return {
            'miou':                    float(np.nanmean(iou_per_class) if valid.any() else 0.0),
            'overall_accuracy':        float(tp.sum() / total_pixels if total_pixels > 0 else 0.0),
            'mean_class_accuracy':     float(np.nanmean(acc_per_class) if valid.any() else 0.0),
            'frequency_weighted_iou':  float(np.nansum(freq * iou_per_class)),
            'iou_per_class':           iou_per_class,
            'accuracy_per_class':      acc_per_class,
            'precision_per_class':     precision_per_class,
            'f1_per_class':            f1_per_class,
            'gt_per_class':            gt_per_class,
            'pred_per_class':          pred_per_class,
        }

    def save_confusion_matrix(self, output_path, class_names):
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            cm = self.confusion_matrix.astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
            present = np.where(cm.sum(axis=1) > 0)[0]
            if len(present) == 0:
                return
            cm_sub   = cm_norm[np.ix_(present, present)]
            names_sub = [class_names[i] for i in present]
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
            ax.set_xticks(range(len(names_sub))); ax.set_yticks(range(len(names_sub)))
            ax.set_xticklabels(names_sub, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(names_sub, fontsize=8)
            ax.set_xlabel('Predicted (DRN)'); ax.set_ylabel('Ground Truth')
            ax.set_title('Stage 2 Confusion Matrix (DRN on Generated RGB)')
            plt.colorbar(im); plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not save confusion matrix: {e}")


# ============================================================================
# FVD frame sampling
# ============================================================================

def sample_video_frames(video, target_frames=16):
    """Uniformly sample target_frames from (T, H, W, C) uint8 array."""
    T = video.shape[0]
    if T >= target_frames:
        idxs = np.linspace(0, T - 1, target_frames, dtype=int)
        return video[idxs]
    pad = [video[-1]] * (target_frames - T)
    return np.concatenate([video, np.stack(pad)], axis=0)


# ============================================================================
# Argument parser
# ============================================================================

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Stage 2 Evaluation: Semantic-to-RGB")

    # Model paths
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='stabilityai/stable-video-diffusion-img2vid-xt')
    parser.add_argument('--semantic_vae_checkpoint', type=str,
                        default='/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth')

    # DRN paths
    parser.add_argument('--drn_dir',        type=str, default='/usrhomes/s1492/drn')
    parser.add_argument('--drn_checkpoint', type=str,
                        default='/usrhomes/s1492/drn/KITTI360_checkpoints/checkpoint_030.pth.tar')
    parser.add_argument('--drn_info_json',  type=str,
                        default='/usrhomes/s1492/drn/CTRLV_BBOX/info.json')
    parser.add_argument('--drn_arch',       type=str, default='drn_d_105')

    # Output
    parser.add_argument('--output_dir', type=str, required=True)

    # Dataset  — validation split, non-overlapping clips by default
    parser.add_argument('--dataset_name', type=str, default='kitti360')
    parser.add_argument('--data_root',    type=str, default='')
    parser.add_argument('--clip_length',  type=int, default=25)
    parser.add_argument('--train_H',      type=int, default=192)
    parser.add_argument('--train_W',      type=int, default=704)
    parser.add_argument('--num_workers',  type=int, default=4)

    # Inference
    parser.add_argument('--num_samples',          type=int,   default=487)
    parser.add_argument('--num_inference_steps',  type=int,   default=30)
    parser.add_argument('--min_guidance_scale',   type=float, default=1.0)
    parser.add_argument('--max_guidance_scale',   type=float, default=3.0)
    parser.add_argument('--conditioning_scale',   type=float, default=1.0)
    parser.add_argument('--noise_aug_strength',   type=float, default=0.01)
    parser.add_argument('--fps',                  type=int,   default=7)
    parser.add_argument('--seed',                 type=int,   default=1234)

    # Optional frame saving (disabled by default — saves time)
    parser.add_argument('--save_frames',    action='store_true', default=False,
                        help='Save GT/gen RGB frames to disk (slows evaluation)')
    parser.add_argument('--num_save_videos', type=int, default=10)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_eval_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16

    print("=" * 80)
    print("Stage 2 Evaluation: Semantic-to-RGB Video Generation")
    print("=" * 80)

    # ------------------------------------------------------------------ checkpoint
    checkpoint_dir  = args.checkpoint_dir
    ckpt_subdirs    = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1])
    )
    if not ckpt_subdirs:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = ckpt_subdirs[-1]
    ckpt_path   = os.path.join(checkpoint_dir, latest_ckpt)
    ckpt_step   = int(latest_ckpt.split("-")[1])

    print(f"Checkpoint:      {ckpt_path}  (step {ckpt_step})")
    print(f"Output:          {args.output_dir}")
    print(f"Dataset:         {args.dataset_name}  (val split, non-overlapping)")
    print(f"Num samples:     {args.num_samples}")
    print(f"Resolution:      {args.train_H}x{args.train_W}  clip_length={args.clip_length}")
    print(f"Infer steps:     {args.num_inference_steps}")
    print(f"Device:          {device}")
    print("=" * 80)

    # ====================================================================
    # [1/5] Load models
    # ====================================================================
    print("\n[1/5] Loading models...")

    ctrlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="control_net")
    unet    = UNetSpatioTemporalConditionModel.from_pretrained(
        ckpt_path, subfolder="unet", low_cpu_mem_usage=True, num_frames=args.clip_length
    )
    print(f"  ✓ ControlNet + UNet loaded from {latest_ckpt}")

    from diffusers.models import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant="fp16"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor"
    )

    from ctrlv.models import DualVAEManager
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
    ctrlnet.eval(); unet.eval()

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

    print(f"  Loading DRN ({args.drn_arch})...")
    drn_model = load_drn_model(args.drn_dir, args.drn_checkpoint, num_classes=19, arch=args.drn_arch)
    print(f"  ✓ DRN loaded")

    # Load LPIPS model
    lpips_fn = None
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net='alex').cuda()
        print("  ✓ LPIPS (AlexNet) loaded")
    except ImportError:
        print("  WARNING: lpips not installed — LPIPS will be skipped")

    # Check skimage for SSIM/PSNR
    ssim_fn = psnr_fn = None
    try:
        from skimage.metrics import structural_similarity as _ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
        ssim_fn, psnr_fn = _ssim_fn, _psnr_fn
        print("  ✓ scikit-image loaded for SSIM/PSNR")
    except ImportError:
        print("  WARNING: scikit-image not installed — SSIM/PSNR will be skipped")

    # ====================================================================
    # [2/5] Load validation data  (non-overlapping clips, val split)
    # ====================================================================
    print("\n[2/5] Loading validation data...")

    val_dataset, val_loader = get_dataloader(
        args.data_root, args.dataset_name, if_train=False,
        clip_length=args.clip_length,
        batch_size=1, num_workers=args.num_workers,
        data_type='clip', use_default_collate=True, tokenizer=None, shuffle=False,
        if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
        use_segmentation=True, use_preplotted_bbox=True,
        if_last_frame_traj=False, non_overlapping_clips=True,
        return_semantic_ids=True,
    )
    total_clips = len(val_dataset)
    num_eval    = min(args.num_samples, total_clips)
    print(f"  Val dataset: {total_clips} non-overlapping clips")
    print(f"  Evaluating:  {num_eval} clips  (streaming — no preload)")

    # ====================================================================
    # [3/5] Run inference + compute per-sample metrics
    # ====================================================================
    print("\n[3/5] Running inference and computing per-sample metrics...")

    drn_metrics = SemanticMetrics(num_classes=19, ignore_index=255)
    generator   = torch.Generator(device=device).manual_seed(args.seed)

    per_sample_results = []
    all_lpips, all_ssim, all_psnr = [], [], []
    all_gt_frames  = []   # for FVD — stores 16-frame clips (N, 16, H, W, 3) uint8
    all_gen_frames = []   # for FVD

    # FID temp dirs on real disk (not /tmp which is RAM-backed tmpfs on SLURM nodes)
    fid_temp    = os.path.join(args.output_dir, '_fid_tmp')
    fid_gt_dir  = os.path.join(fid_temp, 'gt');  os.makedirs(fid_gt_dir,  exist_ok=True)
    fid_gen_dir = os.path.join(fid_temp, 'gen'); os.makedirs(fid_gen_dir, exist_ok=True)
    fid_frame_idx = 0

    # Optional frame saving
    frames_dir = os.path.join(args.output_dir, 'frames')
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    trainid_to_original = {tid: kid for kid, tid in KITTI360_LABEL_MAPPING.items()}

    sample_stream = eval_samples_generator(val_loader)
    for sample_i, sample in enumerate(tqdm(sample_stream, total=num_eval, desc="Evaluating")):
        if sample_i >= num_eval:
            break
        bbox_img_rgb      = sample['bbox_img'].unsqueeze(0)          # [1, T, 3, H, W]
        semantic_ids_cond = sample['semantic_ids'].unsqueeze(0)      # [1, T, H, W]

        with torch.autocast(str(device).replace(":0", ""), enabled=True):
            result = pipeline(
                sample['image_init'],
                cond_images=bbox_img_rgb,
                height=val_dataset.train_H, width=val_dataset.train_W,
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

        # Decode frames — free the pipeline result immediately to reclaim GPU/CPU memory
        gen_frames_pt  = result.frames[0]                                           # [T, 3, H, W] float [0,1]
        gen_frames_np  = (gen_frames_pt.detach().cpu().numpy() * 255).astype(np.uint8)  # [T, 3, H, W]
        gen_frames_hwc = np.transpose(gen_frames_np, (0, 2, 3, 1))                 # [T, H, W, 3]
        del result, gen_frames_pt, gen_frames_np
        torch.cuda.empty_cache()

        gt_frames_np   = sample['gt_clip_np']                                       # [T, 3, H, W] uint8
        gt_frames_hwc  = np.transpose(gt_frames_np, (0, 2, 3, 1))                  # [T, H, W, 3]
        gt_sem_np      = sample['semantic_ids'].cpu().numpy()                       # [T, H, W] int64

        T_clip = gt_frames_hwc.shape[0]

        # -- DRN semantic segmentation on generated RGB --
        drn_pred = drn_predict_batch(drn_model, gen_frames_hwc, args.drn_info_json)
        if drn_pred.shape[1:] != gt_sem_np.shape[1:]:
            drn_pred_t = torch.from_numpy(drn_pred).unsqueeze(1).float()
            drn_pred_t = F.interpolate(drn_pred_t, size=gt_sem_np.shape[1:], mode='nearest')
            drn_pred   = drn_pred_t.squeeze(1).numpy().astype(np.int64)
        drn_metrics.update(drn_pred, gt_sem_np)

        sample_drn = SemanticMetrics(num_classes=19, ignore_index=255)
        sample_drn.update(drn_pred, gt_sem_np)
        s = sample_drn.compute()

        # -- LPIPS (per video, mean over frames) --
        lpips_val = None
        if lpips_fn is not None:
            gt_t  = torch.from_numpy(gt_frames_hwc.transpose(0, 3, 1, 2)).float().cuda() / 127.5 - 1.0
            gen_t = torch.from_numpy(gen_frames_hwc.transpose(0, 3, 1, 2)).float().cuda() / 127.5 - 1.0
            with torch.no_grad():
                lpips_val = lpips_fn(gt_t, gen_t).mean().item()
            all_lpips.append(lpips_val)
            del gt_t, gen_t

        # -- SSIM & PSNR (per frame, float [0,1]) --
        if ssim_fn is not None:
            for t in range(T_clip):
                ig  = gt_frames_hwc[t].astype(np.float32)  / 255.0
                ig2 = gen_frames_hwc[t].astype(np.float32) / 255.0
                all_ssim.append(ssim_fn(ig, ig2, channel_axis=2, data_range=1.0,
                                        gaussian_weights=True, sigma=1.5))
                all_psnr.append(psnr_fn(ig, ig2, data_range=1.0))

        # -- Save frames to FID temp dir --
        for t in range(T_clip):
            Image.fromarray(gt_frames_hwc[t]).save(
                os.path.join(fid_gt_dir,  f'{fid_frame_idx:06d}.png'))
            Image.fromarray(gen_frames_hwc[t]).save(
                os.path.join(fid_gen_dir, f'{fid_frame_idx:06d}.png'))
            fid_frame_idx += 1

        # -- Collect for FVD (sample to 16 frames immediately to keep RAM low) --
        all_gt_frames.append(sample_video_frames(gt_frames_hwc))
        all_gen_frames.append(sample_video_frames(gen_frames_hwc))

        per_sample_results.append({
            'sample_idx':        sample_i,
            'drn_miou':          s['miou'],
            'drn_pixel_accuracy': s['overall_accuracy'],
            'lpips':             lpips_val,
        })

        # -- Optional frame saving --
        if args.save_frames and sample_i < args.num_save_videos:
            video_dir  = os.path.join(frames_dir, f'video_{sample_i:03d}')
            gt_rgb_dir = os.path.join(video_dir, 'gt_rgb')
            gen_rgb_dir = os.path.join(video_dir, 'gen_rgb')
            gt_sem_dir  = os.path.join(video_dir, 'gt_semantic')
            drn_sem_dir = os.path.join(video_dir, 'drn_semantic')
            for d in [gt_rgb_dir, gen_rgb_dir, gt_sem_dir, drn_sem_dir]:
                os.makedirs(d, exist_ok=True)
            for t in range(T_clip):
                Image.fromarray(gt_frames_hwc[t]).save(
                    os.path.join(gt_rgb_dir,  f'frame_{t:03d}.png'))
                Image.fromarray(gen_frames_hwc[t]).save(
                    os.path.join(gen_rgb_dir, f'frame_{t:03d}.png'))
                Image.fromarray(semantic_ids_to_viz_rgb(gt_sem_np[t])).save(
                    os.path.join(gt_sem_dir,  f'frame_{t:03d}.png'))
                Image.fromarray(semantic_ids_to_viz_rgb(drn_pred[t])).save(
                    os.path.join(drn_sem_dir, f'frame_{t:03d}.png'))

    n_evaluated = len(per_sample_results)
    print(f"\n  Inference done. {n_evaluated} clips, {fid_frame_idx} frame pairs collected for FID.")

    # ====================================================================
    # [4/5] Global image/video metrics  (FID, FVD-I3D, FVD-VideoMAE)
    # ====================================================================
    print("\n[4/5] Computing global image/video metrics...")

    # -- FID --
    fid_score = None
    print(f"\n  [4a] FID (Inception-v3 via torch-fidelity)...")
    try:
        from torch_fidelity import calculate_metrics
        metrics_fid = calculate_metrics(
            input1=fid_gen_dir, input2=fid_gt_dir,
            cuda=True, isc=False, fid=True, kid=False, prc=False,
            verbose=False, batch_size=64,
        )
        fid_score = metrics_fid['frechet_inception_distance']
        print(f"  ✓ FID: {fid_score:.4f}")
    except ImportError:
        print("  SKIPPED — torch-fidelity not installed")
    except Exception as e:
        print(f"  WARNING: FID failed: {e}")
    finally:
        shutil.rmtree(fid_temp, ignore_errors=True)

    # -- Build FVD arrays (already sampled to 16 frames during inference loop) --
    print("\n  Stacking FVD arrays (16-frame clips)...")
    gt_fvd_np  = np.stack(all_gt_frames,  axis=0)   # (N, 16, H, W, 3) uint8
    gen_fvd_np = np.stack(all_gen_frames, axis=0)
    print(f"  FVD input shape: {gen_fvd_np.shape}  (N, 16, H, W, C) uint8")

    # -- FVD-I3D --
    fvd_i3d_score = None
    print("\n  [4b] FVD-I3D (cd-fvd)...")
    try:
        from cdfvd import fvd as cdfvd_lib
        evaluator_i3d = cdfvd_lib.cdfvd(model='i3d', device='cuda')
        fvd_i3d_score = evaluator_i3d.compute_fvd(gt_fvd_np, gen_fvd_np)
        print(f"  ✓ FVD-I3D: {fvd_i3d_score:.4f}")
    except Exception as e:
        print(f"  WARNING: FVD-I3D failed: {e}")

    # -- FVD-VideoMAE --
    fvd_videomae_score = None
    print("\n  [4c] FVD-VideoMAE (cd-fvd)...")
    try:
        from cdfvd import fvd as cdfvd_lib
        evaluator_vmae = cdfvd_lib.cdfvd(model='videomae', device='cuda')
        fvd_videomae_score = evaluator_vmae.compute_fvd(gt_fvd_np, gen_fvd_np)
        print(f"  ✓ FVD-VideoMAE: {fvd_videomae_score:.4f}")
    except Exception as e:
        print(f"  WARNING: FVD-VideoMAE failed: {e}")

    # Free FVD arrays
    del gt_fvd_np, gen_fvd_np, all_gt_frames, all_gen_frames
    torch.cuda.empty_cache()

    # -- Aggregate LPIPS / SSIM / PSNR --
    lpips_mean = float(np.mean(all_lpips)) if all_lpips else None
    lpips_std  = float(np.std(all_lpips))  if all_lpips else None
    ssim_mean  = float(np.mean(all_ssim))  if all_ssim  else None
    ssim_std   = float(np.std(all_ssim))   if all_ssim  else None
    psnr_mean  = float(np.mean(all_psnr))  if all_psnr  else None
    psnr_std   = float(np.std(all_psnr))   if all_psnr  else None

    if lpips_mean is not None: print(f"  ✓ LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")
    if ssim_mean  is not None: print(f"  ✓ SSIM:  {ssim_mean:.4f} ± {ssim_std:.4f}")
    if psnr_mean  is not None: print(f"  ✓ PSNR:  {psnr_mean:.4f} ± {psnr_std:.4f} dB")

    # ====================================================================
    # [5/5] Finalise DRN metrics + save all results
    # ====================================================================
    print("\n[5/5] Finalising DRN metrics and saving results...")

    drn_results = drn_metrics.compute()
    class_names = KITTI360_CLASS_NAMES

    # ---- Console summary ----
    print("\n" + "=" * 80)
    print(f"STAGE 2 RESULTS  —  Checkpoint step {ckpt_step}  |  {n_evaluated} clips evaluated")
    print("=" * 80)

    print("\n  Semantic Quality (DRN on Generated RGB vs GT):")
    print(f"    mIoU:              {drn_results['miou']*100:.2f}%")
    print(f"    Pixel Accuracy:    {drn_results['overall_accuracy']*100:.2f}%")
    print(f"    Mean Accuracy:     {drn_results['mean_class_accuracy']*100:.2f}%")
    print(f"    FW-IoU:            {drn_results['frequency_weighted_iou']*100:.2f}%")

    print("\n  Image Quality:")
    if fid_score    is not None: print(f"    FID (Inception-v3): {fid_score:.4f}")
    if lpips_mean   is not None: print(f"    LPIPS (AlexNet):    {lpips_mean:.4f} ± {lpips_std:.4f}")
    if ssim_mean    is not None: print(f"    SSIM:               {ssim_mean:.4f} ± {ssim_std:.4f}")
    if psnr_mean    is not None: print(f"    PSNR:               {psnr_mean:.4f} ± {psnr_std:.4f} dB")

    print("\n  Video Quality (FVD):")
    if fvd_i3d_score      is not None: print(f"    FVD-I3D   (cdfvd/I3D):     {fvd_i3d_score:.4f}")
    if fvd_videomae_score is not None: print(f"    FVD-VideoMAE (cdfvd/ViT):  {fvd_videomae_score:.4f}")

    print("\n  Per-Class IoU (DRN on Generated RGB):")
    for i, name in enumerate(class_names):
        iou = drn_results['iou_per_class'][i]
        val = 'N/A' if np.isnan(iou) else f'{iou*100:.2f}%'
        print(f"    {name:<18} {val}")

    print("\n  Per-Sample Summary:")
    print(f"    {'Sample':<8} {'DRN mIoU':>10} {'PixAcc':>9} {'LPIPS':>8}")
    print(f"    {'-'*38}")
    for sr in per_sample_results:
        lpips_str = f"{sr['lpips']:.4f}" if sr['lpips'] is not None else '  N/A '
        print(f"    {sr['sample_idx']:<8} {sr['drn_miou']*100:>9.2f}%"
              f" {sr['drn_pixel_accuracy']*100:>8.2f}%  {lpips_str}")
    avg_miou = np.mean([sr['drn_miou'] for sr in per_sample_results])
    avg_pix  = np.mean([sr['drn_pixel_accuracy'] for sr in per_sample_results])
    avg_lpips_str = f"{np.mean([sr['lpips'] for sr in per_sample_results if sr['lpips'] is not None]):.4f}" \
                    if any(sr['lpips'] is not None for sr in per_sample_results) else '  N/A '
    print(f"    {'Average':<8} {avg_miou*100:>9.2f}%  {avg_pix*100:>8.2f}%  {avg_lpips_str}")
    print("=" * 80)

    # ---- Save JSON ----
    json_results = {
        'checkpoint':       ckpt_path,
        'checkpoint_step':  ckpt_step,
        'num_samples':      n_evaluated,
        'clip_length':      args.clip_length,
        'resolution':       f'{args.train_H}x{args.train_W}',
        'dataset':          args.dataset_name,
        'split':            'val_non_overlapping',
        'drn_metrics': {
            'miou':                   drn_results['miou'],
            'overall_pixel_accuracy': drn_results['overall_accuracy'],
            'mean_class_accuracy':    drn_results['mean_class_accuracy'],
            'frequency_weighted_iou': drn_results['frequency_weighted_iou'],
        },
        'fid':               fid_score,
        'fvd_i3d':           fvd_i3d_score,
        'fvd_videomae':      fvd_videomae_score,
        'lpips_mean':        lpips_mean,
        'lpips_std':         lpips_std,
        'ssim_mean':         ssim_mean,
        'ssim_std':          ssim_std,
        'psnr_mean':         psnr_mean,
        'psnr_std':          psnr_std,
        'per_class':  {},
        'per_sample': per_sample_results,
    }
    for i, name in enumerate(class_names):
        json_results['per_class'][name] = {
            'iou':       None if np.isnan(drn_results['iou_per_class'][i])       else float(drn_results['iou_per_class'][i]),
            'precision': None if np.isnan(drn_results['precision_per_class'][i]) else float(drn_results['precision_per_class'][i]),
            'recall':    None if np.isnan(drn_results['accuracy_per_class'][i])  else float(drn_results['accuracy_per_class'][i]),
            'f1':        None if np.isnan(drn_results['f1_per_class'][i])        else float(drn_results['f1_per_class'][i]),
            'gt_pixels': int(drn_results['gt_per_class'][i]),
        }

    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ JSON saved:            {json_path}")

    # ---- Save confusion matrix ----
    drn_metrics.save_confusion_matrix(
        os.path.join(args.output_dir, 'confusion_matrix_drn.png'), class_names)
    np.save(os.path.join(args.output_dir, 'confusion_matrix_drn.npy'),
            drn_metrics.confusion_matrix)
    print(f"✓ Confusion matrix saved: {args.output_dir}/confusion_matrix_drn.png")

    # ---- Save text summary ----
    summary_path = os.path.join(args.output_dir, 'eval_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage 2 Evaluation Summary\n{'='*60}\n\n")
        f.write(f"Checkpoint step : {ckpt_step}\n")
        f.write(f"Checkpoint dir  : {checkpoint_dir}\n")
        f.write(f"Dataset         : {args.dataset_name} (val, non-overlapping)\n")
        f.write(f"Num samples     : {n_evaluated}\n")
        f.write(f"Clip length     : {args.clip_length}\n")
        f.write(f"Resolution      : {args.train_H}x{args.train_W}\n\n")
        f.write("DRN Metrics (Generated RGB → Semantic):\n")
        f.write(f"  mIoU              : {drn_results['miou']*100:.2f}%\n")
        f.write(f"  Pixel Accuracy    : {drn_results['overall_accuracy']*100:.2f}%\n")
        f.write(f"  Mean Accuracy     : {drn_results['mean_class_accuracy']*100:.2f}%\n")
        f.write(f"  FW-IoU            : {drn_results['frequency_weighted_iou']*100:.2f}%\n\n")
        f.write("Image Quality:\n")
        if fid_score  is not None: f.write(f"  FID (Inception-v3): {fid_score:.4f}\n")
        if lpips_mean is not None: f.write(f"  LPIPS (AlexNet)   : {lpips_mean:.4f} ± {lpips_std:.4f}\n")
        if ssim_mean  is not None: f.write(f"  SSIM              : {ssim_mean:.4f} ± {ssim_std:.4f}\n")
        if psnr_mean  is not None: f.write(f"  PSNR              : {psnr_mean:.4f} ± {psnr_std:.4f} dB\n")
        f.write("\nVideo Quality (FVD):\n")
        if fvd_i3d_score      is not None: f.write(f"  FVD-I3D (cdfvd/I3D)       : {fvd_i3d_score:.4f}\n")
        if fvd_videomae_score is not None: f.write(f"  FVD-VideoMAE (cdfvd/ViT)  : {fvd_videomae_score:.4f}\n")
        f.write("\nPer-Class IoU (DRN):\n")
        for i, name in enumerate(class_names):
            iou = drn_results['iou_per_class'][i]
            f.write(f"  {name:<18} {'N/A' if np.isnan(iou) else f'{iou*100:.2f}%'}\n")
    print(f"✓ Summary saved:         {summary_path}")
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
