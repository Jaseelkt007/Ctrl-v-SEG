#!/usr/bin/env python3
"""
eval_stage1_semantic_new.py

Evaluates Stage 1 (RGB → Semantic) on the same 19 clip groups used in the
Stage 2 DRN mIoU evaluation.

Clip selection
--------------
Reads drn_eval/CTRLV_STAGE2/val_labels.txt (475 lines = 19 groups × 25 frames).
Derives the first RGB frame for each clip from:
  data_2d_raw/{seq}/image_00/data_rect/{frame:010d}.png

Confidence weighting (--use_confidence_weighting)
--------------------------------------------------
When enabled, each pixel is weighted by its KITTI-360 confidence map value
(16-bit PNG at data_2d_confidences/.../confidence/{frame}.png).
Provides the same weighted mIoU used in the Stage 2 DRN eval.

Outputs
-------
  {output_dir}/eval_report.txt          — structured report matching Stage 2 format
  {output_dir}/eval_results.json        — full numeric results
  {output_dir}/confusion_matrix.png     — row-normalised confusion matrix
  {output_dir}/confusion_matrix.npy     — raw confusion matrix
  {output_dir}/frames/                  — GT vs predicted visualisations (optional)

Usage
-----
  python tools/eval_stage1_semantic_new.py \\
      --checkpoint_dir /no_backups/.../kitti360_semantic_predict_vae \\
      --output_dir     /no_backups/.../eval_stage1_19clips \\
      --drn_eval_dir   drn_eval \\
      --use_confidence_weighting
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir    = os.path.join(os.path.dirname(_script_dir), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ctrlv.utils.semantic_preprocessing import (
    KITTI360_LABEL_MAPPING,
    KITTI360_CLASS_NAMES,
    KITTI360_VIZ_COLORS,
    load_and_remap_semantic,
    semantic_ids_to_viz_rgb,
)

logger = logging.getLogger(__name__)

KITTI360_ROOT  = "/misc/data/public/kitti-360/KITTI-360"
NUM_CLASSES    = 19
IGNORE_INDEX   = 255


# ============================================================================
# Clip group loading  (mirrors generate_stage2_frames_for_drn.py)
# ============================================================================

def load_clip_groups(val_labels_txt: str, clip_length: int = 25):
    """Parse val_labels.txt into a list of clip group dicts."""
    with open(val_labels_txt) as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) % clip_length != 0:
        raise ValueError(
            f"val_labels.txt has {len(lines)} lines — not divisible by clip_length={clip_length}"
        )

    groups = []
    seq_clip_counters: dict = {}
    for g in range(len(lines) // clip_length):
        chunk = lines[g * clip_length: (g + 1) * clip_length]
        # Derive sequence name from path structure
        parts = chunk[0].split(os.sep)
        seq = next((p for p in parts if "drive" in p), "unknown")
        frame_ids = []
        for path in chunk:
            fname = os.path.splitext(os.path.basename(path))[0]
            frame_ids.append(int(fname))

        seq_idx = seq_clip_counters.get(seq, 0)
        seq_clip_counters[seq] = seq_idx + 1

        # Derive RGB path for the first frame
        first_rgb = os.path.join(
            KITTI360_ROOT, "data_2d_raw", seq,
            "image_00", "data_rect", f"{frame_ids[0]:010d}.png"
        )

        groups.append({
            "group_idx":   g,
            "folder_name": f"{seq}_{seq_idx:02d}",
            "sequence":    seq,
            "frame_ids":   frame_ids,
            "label_paths": chunk,
            "first_rgb":   first_rgb,
        })

    return groups


# ============================================================================
# Confidence map loading
# ============================================================================

def load_confidence_map(semantic_path: str, target_H: int, target_W: int) -> np.ndarray:
    """
    Load a KITTI-360 confidence PNG (16-bit I;16) and return float32 [H, W] in [0, 1].
    Returns None if the file does not exist.
    """
    conf_path = semantic_path.replace("semantic", "confidence")
    if not os.path.exists(conf_path):
        return None
    img = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)  # uint16 [H, W]
    if img is None:
        return None
    if img.shape[:2] != (target_H, target_W):
        img = cv2.resize(img, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 65535.0


# ============================================================================
# Weighted confusion matrix
# ============================================================================

def update_confusion_matrix(
    cm: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    confidence: np.ndarray = None,
):
    """
    Update an [N, N] confusion matrix with a [H, W] or [T, H, W] pred/gt pair.
    If confidence [H, W] or [T, H, W] is provided, weight each pixel accordingly.
    """
    if pred.ndim == 3:
        T = pred.shape[0]
        conf_seq = None
        for t in range(T):
            c = confidence[t] if (confidence is not None and confidence.ndim == 3) else confidence
            _update_single(cm, pred[t], gt[t], c)
    else:
        _update_single(cm, pred, gt, confidence)


def _update_single(cm, pred, gt, confidence=None):
    valid = (gt != IGNORE_INDEX) & (gt < NUM_CLASSES) & (pred < NUM_CLASSES)
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    if confidence is not None:
        w = confidence[valid].astype(np.float64)
        for cls_g in range(NUM_CLASSES):
            for cls_p in range(NUM_CLASSES):
                mask = (g == cls_g) & (p == cls_p)
                if mask.any():
                    cm[cls_g, cls_p] += w[mask].sum()
    else:
        np.add.at(cm, (g, p), 1)


def compute_metrics_from_cm(cm: np.ndarray) -> dict:
    tp  = np.diag(cm)
    fp  = cm.sum(0) - tp
    fn  = cm.sum(1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_per_class = np.where(denom > 0, tp / denom, np.nan)
        acc_per_class = np.where(cm.sum(1) > 0, tp / cm.sum(1), np.nan)
    miou            = float(np.nanmean(iou_per_class))
    overall_acc     = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
    mean_acc        = float(np.nanmean(acc_per_class))
    freq            = cm.sum(1) / cm.sum() if cm.sum() > 0 else np.zeros(NUM_CLASSES)
    fwiou           = float(np.nansum(freq * iou_per_class))
    return {
        "miou":          miou,
        "overall_accuracy": overall_acc,
        "mean_accuracy": mean_acc,
        "fwiou":         fwiou,
        "iou_per_class": iou_per_class,
        "acc_per_class": acc_per_class,
    }


# ============================================================================
# Visualisation helpers  (reused from eval_stage1_semantic.py)
# ============================================================================

def save_side_by_side(gt_ids, pred_ids, path):
    gt_rgb   = semantic_ids_to_viz_rgb(gt_ids)
    pred_rgb = semantic_ids_to_viz_rgb(pred_ids)
    H        = gt_rgb.shape[0]
    sep      = np.ones((H, 4, 3), dtype=np.uint8) * 255
    Image.fromarray(np.concatenate([gt_rgb, sep, pred_rgb], axis=1)).save(path)


def save_confusion_matrix_png(cm: np.ndarray, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        row_sums = cm.sum(1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(KITTI360_CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(KITTI360_CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        ax.set_title("Stage 1 Confusion Matrix (row-normalised)\n19-clip evaluation set")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = cm_norm[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Confusion matrix saved: {output_path}")
    except Exception as e:
        print(f"  Warning: could not save confusion matrix PNG: {e}")


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--drn_eval_dir",   default="drn_eval",
                        help="Path to drn_eval/ (contains CTRLV_STAGE2/val_labels.txt)")
    parser.add_argument("--kitti360_root",  default=KITTI360_ROOT)
    parser.add_argument("--pretrained_model_name_or_path",
                        default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--semantic_vae_checkpoint",
                        default="/usrhomes/s1492/vae_semantic/checkpoints/"
                                "semantic_vae_native/best_model_with_dice_boundaryweight.pth")
    parser.add_argument("--clip_length",         type=int,   default=25)
    parser.add_argument("--train_H",             type=int,   default=192)
    parser.add_argument("--train_W",             type=int,   default=704)
    parser.add_argument("--num_inference_steps", type=int,   default=30)
    parser.add_argument("--min_guidance_scale",  type=float, default=3.0)
    parser.add_argument("--max_guidance_scale",  type=float, default=7.0)
    parser.add_argument("--noise_aug_strength",  type=float, default=0.01)
    parser.add_argument("--fps",                 type=int,   default=7)
    parser.add_argument("--seed",                type=int,   default=1234)
    parser.add_argument("--num_cond_bbox_frames",type=int,   default=1)
    parser.add_argument("--use_confidence_weighting", action="store_true",
                        help="Weight confusion matrix by KITTI-360 per-pixel confidence maps.")
    parser.add_argument("--save_frames", action="store_true", default=False,
                        help="Save GT vs predicted visualisations for each clip.")
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    # ------------------------------------------------------------------ #
    # [1/4]  Load clip groups from static val_labels.txt
    # ------------------------------------------------------------------ #
    val_labels_txt = os.path.join(args.drn_eval_dir, "CTRLV_STAGE2", "val_labels.txt")
    print(f"\n[1/4] Loading clip groups from {val_labels_txt}")
    groups = load_clip_groups(val_labels_txt, clip_length=args.clip_length)
    print(f"  Loaded {len(groups)} clip groups × {args.clip_length} frames "
          f"= {len(groups) * args.clip_length} frames total")
    for g in groups:
        print(f"    {g['folder_name']:<45}  frames {g['frame_ids'][0]}–{g['frame_ids'][-1]}")

    # Confidence check
    if args.use_confidence_weighting:
        first_conf = groups[0]["label_paths"][0].replace("semantic", "confidence")
        if os.path.exists(first_conf):
            print(f"  ✓ Confidence maps found — confidence-weighted mIoU enabled")
        else:
            print(f"  WARNING: confidence map not found at {first_conf}")
            print(f"           Falling back to unweighted mIoU.")
            args.use_confidence_weighting = False

    # ------------------------------------------------------------------ #
    # [2/4]  Load Stage 1 pipeline
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading Stage 1 pipeline...")

    from diffusers import EulerDiscreteScheduler
    from diffusers.models import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from ctrlv.models import UNetSpatioTemporalConditionModel, DualVAEManager
    from ctrlv.pipelines import VideoDiffusionPipeline

    # Resolve checkpoint
    best_path = os.path.join(args.checkpoint_dir, "best_checkpoint")
    if os.path.exists(best_path):
        ckpt_path = best_path; ckpt_step = "best"
    else:
        subdirs = sorted(
            [d for d in os.listdir(args.checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        if not subdirs:
            raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")
        ckpt_path = os.path.join(args.checkpoint_dir, subdirs[-1])
        ckpt_step = subdirs[-1].split("-")[1]
    print(f"  Checkpoint: {ckpt_path}  (step {ckpt_step})")

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant="fp16")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        ckpt_path, subfolder="unet", low_cpu_mem_usage=True,
        num_frames=args.clip_length)

    vae_manager = DualVAEManager(
        rgb_vae=vae, semantic_vae_checkpoint=args.semantic_vae_checkpoint,
        num_semantic_classes=19, device=device,
        clip_size=args.clip_length, verbose=True)

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    unet.eval()

    pipeline = VideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, unet=unet, image_encoder=image_encoder,
        vae=vae, feature_extractor=feature_extractor,
        revision=None, variant="fp16", torch_dtype=weight_dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae_manager = vae_manager
    print("  ✓ Pipeline ready")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------ #
    # [3/4]  Inference + metric accumulation
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Running inference on {len(groups)} clips...")
    print(f"      Confidence weighting: {'ON' if args.use_confidence_weighting else 'OFF'}")

    # Separate unweighted and weighted confusion matrices so we can report both
    cm_unweighted = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    cm_weighted   = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    per_clip_results = []
    frames_dir = os.path.join(args.output_dir, "frames")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    for g in tqdm(groups, desc="Clips"):
        seq        = g["sequence"]
        frame_ids  = g["frame_ids"]
        label_paths= g["label_paths"]

        # ---- Load first RGB frame for image conditioning ----
        first_rgb_path = os.path.join(
            args.kitti360_root, "data_2d_raw", seq,
            "image_00", "data_rect", f"{frame_ids[0]:010d}.png"
        )
        if not os.path.exists(first_rgb_path):
            print(f"  WARNING: RGB not found: {first_rgb_path}  — skipping clip")
            continue
        image_init = Image.open(first_rgb_path).convert("RGB").resize(
            (args.train_W, args.train_H), Image.LANCZOS)

        # ---- Load GT semantic maps [T, H, W] trainIDs ----
        gt_list = []
        for lp in label_paths:
            sem = load_and_remap_semantic(lp)  # [H_orig, W_orig] trainIDs
            # Resize to training resolution using nearest-neighbour (preserves integer IDs)
            sem_img = Image.fromarray(sem.astype(np.uint8), mode='L')
            sem_img = sem_img.resize((args.train_W, args.train_H), Image.NEAREST)
            sem = np.array(sem_img, dtype=np.int64)
            gt_list.append(sem)
        gt_np = np.stack(gt_list, axis=0)  # [T, H, W]

        # ---- Build semantic conditioning tensors ----
        sem_rgb_list = []
        for t in range(args.clip_length):
            rgb = semantic_ids_to_viz_rgb(gt_np[t])          # [H, W, 3] uint8
            t_f = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3, H, W]
            sem_rgb_list.append(t_f)
        sem_rgb = torch.stack(sem_rgb_list, dim=0).unsqueeze(0).to(device)  # [1, T, 3, H, W]

        semantic_ids_cond = torch.from_numpy(gt_np).unsqueeze(0).to(device)  # [1, T, H, W]

        # ---- Run Stage 1 inference ----
        with torch.no_grad(), torch.autocast(str(device).replace(":0", ""), enabled=True):
            result = pipeline(
                image_init,
                height=args.train_H, width=args.train_W,
                bbox_images=sem_rgb,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.clip_length,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                noise_aug_strength=args.noise_aug_strength,
                generator=generator,
                output_type="latent",
                num_cond_bbox_frames=args.num_cond_bbox_frames,
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )

        latents   = result.frames[0].to(torch.float32)          # [T, C, H, W]
        pred_ids  = vae_manager.decode_semantic(latents)         # [T, H, W] trainIDs
        pred_np   = pred_ids.cpu().numpy()
        del result, latents, pred_ids
        torch.cuda.empty_cache()

        # ---- Load confidence maps [T, H, W] float32 [0,1] if available ----
        conf_np = None
        if args.use_confidence_weighting:
            conf_list = []
            for lp in label_paths:
                c = load_confidence_map(lp, args.train_H, args.train_W)
                conf_list.append(c if c is not None else np.ones((args.train_H, args.train_W), np.float32))
            conf_np = np.stack(conf_list, axis=0)   # [T, H, W]

        # ---- Update confusion matrices ----
        update_confusion_matrix(cm_unweighted, pred_np, gt_np, confidence=None)
        update_confusion_matrix(cm_weighted,   pred_np, gt_np, confidence=conf_np)

        # Per-clip unweighted mIoU for ranking
        cm_clip = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        update_confusion_matrix(cm_clip, pred_np, gt_np, confidence=None)
        clip_m  = compute_metrics_from_cm(cm_clip)

        per_clip_results.append({
            "group_idx":    g["group_idx"],
            "folder_name":  g["folder_name"],
            "sequence":     seq,
            "start_frame":  frame_ids[0],
            "end_frame":    frame_ids[-1],
            "miou":         clip_m["miou"],
            "pixel_accuracy": clip_m["overall_accuracy"],
        })

        if args.save_frames:
            clip_dir = os.path.join(frames_dir, g["folder_name"])
            os.makedirs(os.path.join(clip_dir, "comparison"), exist_ok=True)
            T = min(pred_np.shape[0], gt_np.shape[0])
            for t in range(T):
                save_side_by_side(
                    gt_np[t], pred_np[t],
                    os.path.join(clip_dir, "comparison", f"frame_{t:03d}.png"))

    # ------------------------------------------------------------------ #
    # [4/4]  Compute metrics, save outputs
    # ------------------------------------------------------------------ #
    print("\n[4/4] Computing metrics and saving outputs...")

    # Choose which CM to report based on flag
    cm_report = cm_weighted if args.use_confidence_weighting else cm_unweighted
    m         = compute_metrics_from_cm(cm_report)

    # Always also compute unweighted for reference
    m_uw = compute_metrics_from_cm(cm_unweighted)

    ious = m["iou_per_class"]
    miou = m["miou"] * 100

    # Save CMs
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm_report)
    save_confusion_matrix_png(cm_report, os.path.join(args.output_dir, "confusion_matrix.png"))
    if args.use_confidence_weighting:
        np.save(os.path.join(args.output_dir, "confusion_matrix_unweighted.npy"), cm_unweighted)

    # Save JSON
    results_json = {
        "checkpoint_dir":      args.checkpoint_dir,
        "checkpoint_step":     ckpt_step,
        "clip_groups":         len(groups),
        "total_frames":        len(groups) * args.clip_length,
        "confidence_weighted": args.use_confidence_weighting,
        "miou":                float(m["miou"]),
        "pixel_accuracy":      float(m["overall_accuracy"]),
        "mean_accuracy":       float(m["mean_accuracy"]),
        "fwiou":               float(m["fwiou"]),
        "miou_unweighted":     float(m_uw["miou"]),
        "iou_per_class": {
            KITTI360_CLASS_NAMES[i]: (None if np.isnan(v) else float(v))
            for i, v in enumerate(ious)
        },
        "per_clip": per_clip_results,
    }
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Write report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(args.output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        def w(line=""):
            f.write(line + "\n")
        w("Stage 1 Semantic Evaluation Report  (19-Clip Set)")
        w("=" * 60)
        w(f"Evaluation Date   : {now}")
        w(f"Checkpoint Dir    : {args.checkpoint_dir}")
        w(f"Checkpoint Step   : {ckpt_step}")
        w()
        w("-" * 60)
        w("Evaluation Protocol")
        w("-" * 60)
        w(f"Clip Set          : drn_eval/CTRLV_STAGE2/val_labels.txt  (same 19 clips as Stage 2)")
        w(f"Frame Count       : {len(groups)} groups × {args.clip_length} = {len(groups)*args.clip_length} frames")
        w(f"Resolution        : {args.train_H}×{args.train_W}")
        w(f"Inference Steps   : {args.num_inference_steps}")
        w(f"Guidance Scale    : {args.min_guidance_scale} – {args.max_guidance_scale}")
        w(f"Confidence Weight : {'ON' if args.use_confidence_weighting else 'OFF'}")
        w()
        w("-" * 60)
        w("Clip Groups (19 total)")
        w("-" * 60)
        w(f" {'#':>3}  {'Sequence':<40}  Frames")
        for g in groups:
            w(f" {g['group_idx']:>3}  {g['sequence']:<40}  "
              f"{g['frame_ids'][0]:010d} – {g['frame_ids'][-1]:010d}")
        w()
        w("-" * 60)
        w("Overall Results")
        w("-" * 60)
        conf_tag = " (confidence-weighted)" if args.use_confidence_weighting else " (unweighted)"
        w(f"mIoU{conf_tag:<28}: {miou:.2f}%")
        if args.use_confidence_weighting:
            w(f"mIoU (unweighted)                    : {m_uw['miou']*100:.2f}%")
        w(f"Pixel Accuracy                       : {m['overall_accuracy']*100:.2f}%")
        w(f"Mean Accuracy                        : {m['mean_accuracy']*100:.2f}%")
        w(f"FW-IoU                               : {m['fwiou']*100:.2f}%")
        w()
        w("-" * 60)
        w("Per-Class IoU")
        w("-" * 60)
        w(f" {'ID':>4}  {'Class':<16}  {'IoU (%)':>8}")
        w(f" {'----':>4}  {'-'*16}  {'--------':>8}")
        for i, (cls, iou) in enumerate(zip(KITTI360_CLASS_NAMES, ious)):
            if np.isnan(iou):
                w(f" {i:>4}  {cls:<16}     n/a   (absent)")
            else:
                w(f" {i:>4}  {cls:<16}  {iou*100:>8.2f}")
        num_valid = int(np.sum(~np.isnan(ious)))
        w(f"\n mIoU = nanmean over {num_valid} non-NaN classes = {miou:.2f}%")
        w()
        w("-" * 60)
        w("Per-Clip mIoU")
        w("-" * 60)
        w(f" {'#':>3}  {'Folder':<46}  {'mIoU':>8}  {'PixAcc':>8}")
        for r in sorted(per_clip_results, key=lambda x: x["group_idx"]):
            w(f" {r['group_idx']:>3}  {r['folder_name']:<46}  "
              f"{r['miou']*100:>7.2f}%  {r['pixel_accuracy']*100:>7.2f}%")
        w()
        w("-" * 60)
        w("Output Paths")
        w("-" * 60)
        w(f"Report            : {report_path}")
        w(f"JSON              : {json_path}")
        w(f"Confusion matrix  : {args.output_dir}/confusion_matrix.npy  /  .png")

    print(f"\n{'='*60}")
    print(f"  mIoU{conf_tag}: {miou:.2f}%")
    if args.use_confidence_weighting:
        print(f"  mIoU (unweighted)        : {m_uw['miou']*100:.2f}%")
    print(f"  Pixel Accuracy           : {m['overall_accuracy']*100:.2f}%")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
