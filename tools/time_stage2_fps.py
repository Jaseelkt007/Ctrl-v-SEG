"""
Quick Stage 2 inference timing: runs 2 samples and reports FPS/wall-time.
No metrics, no DRN — purely generation speed measurement.
Mirrors the exact loading/inference pattern from eval_stage2_rgb.py.
"""
import time, os, sys, warnings
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "/usrhomes/s1492/Ctrl-V-seg/src")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import get_dataloader, eval_samples_generator
    from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel, DualVAEManager
    from ctrlv.pipelines import StableVideoControlPipeline
    from diffusers.models import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = "/no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze_reinject"
SVD_BASE           = "stabilityai/stable-video-diffusion-img2vid-xt"
SEMANTIC_VAE_CKPT  = "/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth"
CLIP_LENGTH        = 25
H, W               = 192, 704
NUM_SAMPLES        = 2
NUM_STEPS          = 30
SEED               = 1234
# ─────────────────────────────────────────────────────────────────────────────

device       = torch.device("cuda")
weight_dtype = torch.float16

print("=" * 60)
print("Stage 2 Speed Benchmark")
print(f"  Checkpoint : {CHECKPOINT_DIR}")
print(f"  Resolution : {H}x{W}  |  clip_length={CLIP_LENGTH}")
print(f"  Inf steps  : {NUM_STEPS}  |  samples={NUM_SAMPLES}")
print("=" * 60)

# resolve best/latest checkpoint
best = os.path.join(CHECKPOINT_DIR, "best_checkpoint")
if os.path.exists(best):
    ckpt_path = best
else:
    subs = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, subs[-1])
print(f"  Using ckpt : {ckpt_path}\n")

# ── load models ───────────────────────────────────────────────────────────────
print("[1/3] Loading models …")
t0 = time.time()

ctrlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="control_net")
unet    = UNetSpatioTemporalConditionModel.from_pretrained(
    ckpt_path, subfolder="unet", low_cpu_mem_usage=True, num_frames=CLIP_LENGTH
)
vae = AutoencoderKLTemporalDecoder.from_pretrained(SVD_BASE, subfolder="vae", variant="fp16")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(SVD_BASE, subfolder="image_encoder", variant="fp16")
feature_extractor = CLIPImageProcessor.from_pretrained(SVD_BASE, subfolder="feature_extractor")

vae_manager = DualVAEManager(
    rgb_vae=vae,
    semantic_vae_checkpoint=SEMANTIC_VAE_CKPT,
    num_semantic_classes=19,
    device=device,
    clip_size=CLIP_LENGTH,
    verbose=True,
)

vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
image_encoder.to(device, dtype=weight_dtype)
ctrlnet.to(device, dtype=weight_dtype)
ctrlnet.eval(); unet.eval()

pipeline = StableVideoControlPipeline.from_pretrained(
    SVD_BASE,
    unet=unet, controlnet=ctrlnet,
    image_encoder=image_encoder, vae=vae,
    feature_extractor=feature_extractor,
    variant="fp16", torch_dtype=weight_dtype,
).to(device)
pipeline.set_progress_bar_config(disable=True)
pipeline.vae_manager = vae_manager

print(f"  Model load time: {time.time()-t0:.1f}s")

# ── load dataset ──────────────────────────────────────────────────────────────
print("\n[2/3] Loading val dataloader …")

val_dataset, val_loader = get_dataloader(
    dset_root            = "",
    dset_name            = "kitti360",
    if_train             = False,
    batch_size           = 1,
    num_workers          = 0,
    data_type            = "clip",
    clip_length          = CLIP_LENGTH,
    train_H              = H,
    train_W              = W,
    use_segmentation     = True,
    return_semantic_ids  = True,
    if_return_bbox_im    = True,
    non_overlapping_clips= True,
    shuffle              = False,
)
print(f"  Val clips: {len(val_dataset)}\n")

# ── run timing ────────────────────────────────────────────────────────────────
print(f"[3/3] Running {NUM_SAMPLES} inference samples …\n")

generator = torch.Generator(device=device).manual_seed(SEED)
sample_times, frame_counts = [], []

sample_stream = eval_samples_generator(val_loader)
for i, sample in enumerate(sample_stream):
    if i >= NUM_SAMPLES:
        break

    bbox_img_rgb      = sample['bbox_img'].unsqueeze(0)        # [1, T, 3, H, W]
    semantic_ids_cond = sample['semantic_ids'].unsqueeze(0)    # [1, T, H, W]

    torch.cuda.synchronize()
    t_start = time.time()

    with torch.autocast(str(device).replace(":0", ""), enabled=True):
        result = pipeline(
            sample['image_init'],
            cond_images             = bbox_img_rgb,
            height                  = H,
            width                   = W,
            decode_chunk_size       = 8,
            motion_bucket_id        = 127,
            fps                     = 7,
            num_inference_steps     = NUM_STEPS,
            num_frames              = CLIP_LENGTH,
            control_condition_scale = 1.0,
            min_guidance_scale      = 1.0,
            max_guidance_scale      = 3.0,
            noise_aug_strength      = 0.01,
            generator               = generator,
            output_type             = 'pt',
            semantic_ids            = semantic_ids_cond,
            use_semantic_vae        = True,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    del result
    torch.cuda.empty_cache()

    n_frames = CLIP_LENGTH
    fps_val  = n_frames / elapsed
    sample_times.append(elapsed)
    frame_counts.append(n_frames)

    print(f"  Sample {i+1}/{NUM_SAMPLES}: {elapsed:.2f}s  →  {fps_val:.3f} frames/s  ({n_frames} frames @ {H}x{W})")

# ── summary ───────────────────────────────────────────────────────────────────
total_time   = sum(sample_times)
total_frames = sum(frame_counts)
avg_fps      = total_frames / total_time
avg_per_clip = total_time / len(sample_times)

print("\n" + "=" * 60)
print("RESULTS")
print(f"  Samples           : {NUM_SAMPLES}")
print(f"  Frames per clip   : {CLIP_LENGTH}")
print(f"  Inference steps   : {NUM_STEPS}")
print(f"  Resolution        : {H}x{W}")
print(f"  Avg time/clip     : {avg_per_clip:.2f}s")
print(f"  Avg FPS           : {avg_fps:.3f} frames/s")
print(f"  Sec/frame         : {1/avg_fps:.3f}s")
print(f"  Total wall time   : {total_time:.2f}s")
print("=" * 60)
