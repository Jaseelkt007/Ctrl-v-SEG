#!/bin/bash
#SBATCH --job-name=full_inf_viz
#SBATCH --output=/usrhomes/s1492/Ctrl-V-seg/logs/full_inference_%j.out
#SBATCH --error=/usrhomes/s1492/Ctrl-V-seg/logs/full_inference_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=03:00:00

set -e
set -u

echo "========================================="
echo "Full Inference Test with Semantic VAE + Visualization"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-localhost}"
echo "Started at: $(date)"
echo ""

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:/usrhomes/s1492/vae_semantic:${PYTHONPATH:-}"

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Configuration
DATASET_PATH="/no_backups/s1492/"
OUT_DIR="/usrhomes/s1492/Ctrl-V-seg/tests/full_inference_output"
mkdir -p "$OUT_DIR"
mkdir -p "/usrhomes/s1492/Ctrl-V-seg/logs"

# Checkpoints
BBOX_MODEL="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict"
SEM2VID_MODEL="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video/checkpoint-77300"
SEMANTIC_VAE_CKPT="/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth"

echo "Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  BBox Model: $BBOX_MODEL"
echo "  Sem2Vid Model: $SEM2VID_MODEL"
echo "  Semantic VAE: $SEMANTIC_VAE_CKPT"
echo "  Output: $OUT_DIR"
echo ""

# Run full inference with visualization
python3 << 'PYTEST'
import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V-seg/src')
sys.path.insert(0, '/usrhomes/s1492/vae_semantic')

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os
from einops import rearrange
from torchvision.utils import save_image
import torch.nn.functional as F

print("\n" + "="*80)
print("FULL 2-STAGE INFERENCE WITH SEMANTIC VAE")
print("="*80)

OUT_DIR = Path("/usrhomes/s1492/Ctrl-V-seg/tests/full_inference_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(OUT_DIR / "ground_truth").mkdir(exist_ok=True)
(OUT_DIR / "stage1_bbox").mkdir(exist_ok=True)
(OUT_DIR / "stage2_generated").mkdir(exist_ok=True)
(OUT_DIR / "semantic_viz").mkdir(exist_ok=True)
(OUT_DIR / "latents").mkdir(exist_ok=True)

print("\n[1/7] Loading dataset...")
from ctrlv.datasets.kitti360_inference import KITTI360InferenceDataset

dataset = KITTI360InferenceDataset(
    root='/no_backups/s1492/',
    train=False,
    data_type='clip',
    clip_length=25,
    if_return_bbox_im=True,
    use_segmentation=True,
    return_semantic_ids=True,
    train_H=128,
    train_W=512
)

print(f"✓ Dataset loaded: {len(dataset)} samples")

# Get one sample
# Dataset returns: (images, targets, prompt, index, bboxes, semantic_ids)
sample_data = dataset[0]
print(f"Sample data: {len(sample_data)} items")

# Unpack: (rgb_clips, targets, prompt, index, bbox_images, semantic_ids)
rgb_clips = sample_data[0]    # [25, 3, 128, 512]
targets = sample_data[1]       # List of bbox annotations
prompt = sample_data[2]        # Text prompt (string)
sample_index = sample_data[3]  # int - sample index
bbox_images = sample_data[4]   # [25, 3, 128, 512] - semantic RGB viz
semantic_ids = sample_data[5] if len(sample_data) > 5 else None  # [25, 128, 512] - trainIDs

print(f"✓ Sample loaded (index {sample_index}):")
print(f"  RGB clips: {rgb_clips.shape}")
print(f"  Semantic RGB viz: {bbox_images.shape}")
if semantic_ids is not None:
    print(f"  Semantic IDs: {semantic_ids.shape} (range: [{semantic_ids.min()}, {semantic_ids.max()}])")
else:
    print(f"  WARNING: No semantic IDs in sample!")

# Save ground truth frames
print("\n[2/7] Saving ground truth frames...")
for i in range(rgb_clips.shape[0]):
    save_image(rgb_clips[i], OUT_DIR / "ground_truth" / f"frame_{i:03d}.png")
print(f"✓ Saved {rgb_clips.shape[0]} ground truth frames")

# Save semantic visualizations
print("\n[3/7] Saving semantic visualizations...")
if semantic_ids is not None:
    from ctrlv.utils.semantic_preprocessing import semantic_ids_to_viz_rgb
    for i in range(semantic_ids.shape[0]):
        semantic_viz = semantic_ids_to_viz_rgb(semantic_ids[i].numpy())
        Image.fromarray(semantic_viz).save(OUT_DIR / "semantic_viz" / f"semantic_{i:03d}.png")
    print(f"✓ Saved {semantic_ids.shape[0]} semantic visualizations")
else:
    print("⚠ Skipping semantic visualizations (no semantic IDs)")

print("\n[4/7] Initializing DualVAEManager...")
from ctrlv.models import DualVAEManager
from diffusers.models import AutoencoderKLTemporalDecoder

rgb_vae = AutoencoderKLTemporalDecoder.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    subfolder='vae',
    variant='fp16'
).to('cuda')

vae_manager = DualVAEManager(
    rgb_vae=rgb_vae,
    semantic_vae_checkpoint='/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth',
    num_semantic_classes=19,
    device='cuda',
    clip_size=4,
    verbose=True
)
print("✓ DualVAEManager initialized")

# Encode RGB and Semantic
print("\n[5/7] Encoding through VAEs...")
rgb_frames = rgb_clips.unsqueeze(0).cuda()  # [1, 25, 3, 128, 512]
if semantic_ids is not None:
    semantic_ids_batch = semantic_ids.unsqueeze(0).cuda()  # [1, 25, 128, 512]
else:
    semantic_ids_batch = None

# RGB encoding
rgb_flat = rearrange(rgb_frames, "b f c h w -> (b f) c h w")
with torch.no_grad():
    rgb_latents = vae_manager.encode_rgb(rgb_flat)
rgb_latents = rearrange(rgb_latents, "(b f) c h w -> b f c h w", b=1)
print(f"✓ RGB encoding: {rgb_frames.shape} -> {rgb_latents.shape}")

# Semantic encoding
if semantic_ids_batch is not None:
    semantic_flat = rearrange(semantic_ids_batch, "b f h w -> (b f) h w")
    with torch.no_grad():
        semantic_latents = vae_manager.encode_semantic_from_ids(semantic_flat)
    semantic_latents = rearrange(semantic_latents, "(b f) c h w -> b f c h w", b=1)
    print(f"✓ Semantic encoding: {semantic_ids_batch.shape} -> {semantic_latents.shape}")
else:
    semantic_latents = None
    print("⚠ Skipping semantic encoding (no semantic IDs)")

# Save latent visualizations (as normalized images)
print("\n[6/7] Saving latent visualizations...")
# Normalize latents to [0, 1] for visualization
rgb_latent_vis = (rgb_latents[0, 0] - rgb_latents[0, 0].min()) / (rgb_latents[0, 0].max() - rgb_latents[0, 0].min() + 1e-8)
save_image(rgb_latent_vis, OUT_DIR / "latents" / "rgb_latent_frame0.png")

if semantic_latents is not None:
    semantic_latent_vis = (semantic_latents[0, 0] - semantic_latents[0, 0].min()) / (semantic_latents[0, 0].max() - semantic_latents[0, 0].min() + 1e-8)
    save_image(semantic_latent_vis, OUT_DIR / "latents" / "semantic_latent_frame0.png")

print(f"✓ Saved latent visualizations")

print("\n[7/7] Loading inference pipelines...")
from ctrlv.pipelines import VideoDiffusionPipeline, StableVideoControlPipeline
from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel

# Stage 1: BBox Prediction Pipeline (EXACTLY like eval_overall.py)
print("\n  Loading Stage 1 (BBox Prediction)...")
bbox_unet = UNetSpatioTemporalConditionModel.from_pretrained(
    '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict/checkpoint-52800',
    subfolder="unet",
    low_cpu_mem_usage=True,
    num_frames=25
).to('cuda')

bbox_pipeline = VideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=bbox_unet
).to('cuda')
bbox_pipeline.vae_manager = vae_manager
print("✓ Stage 1 pipeline loaded")

# Stage 2: Semantic2Video Pipeline (EXACTLY like eval_overall.py)
print("\n  Loading Stage 2 (Semantic2Video)...")
ctrlnet = ControlNetModel.from_pretrained(
    '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video/checkpoint-77300',
    subfolder="control_net"
).to('cuda')

unet = UNetSpatioTemporalConditionModel.from_pretrained(
    '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video/checkpoint-77300',
    subfolder="unet"
).to('cuda')

s2v_pipeline = StableVideoControlPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    controlnet=ctrlnet,
    unet=unet
).to('cuda')
s2v_pipeline.vae_manager = vae_manager
print("✓ Stage 2 pipeline loaded")

# Run Stage 1 inference
print("\n" + "="*80)
print("STAGE 1: BBOX/SEMANTIC PREDICTION")
print("="*80)

# Prepare initial frame
from torchvision import transforms
to_pil = transforms.ToPILImage()
init_frame_pil = to_pil(rgb_clips[0])

print(f"Initial frame shape: {rgb_clips[0].shape}")
if semantic_ids is not None:
    print(f"Semantic IDs for conditioning: {semantic_ids.shape}")
    semantic_for_stage1 = semantic_ids.unsqueeze(0).cuda()
    use_sem_vae = True
else:
    print("⚠ No semantic IDs available, using RGB bbox images")
    semantic_for_stage1 = None
    use_sem_vae = False

# Run bbox prediction
with torch.no_grad():
    bbox_output = bbox_pipeline(
        init_frame_pil,
        height=128,
        width=512,
        bbox_images=bbox_images.unsqueeze(0).cuda(),
        semantic_ids=semantic_for_stage1,
        use_semantic_vae=use_sem_vae,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        num_inference_steps=30,
        num_frames=25,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        noise_aug_strength=0.02,
        output_type='pt'
    )

stage1_frames = bbox_output.frames[0]  # [25, 3, 128, 512]
print(f"✓ Stage 1 output: {stage1_frames.shape}")

# Save Stage 1 outputs
print("\nSaving Stage 1 outputs...")
for i in range(stage1_frames.shape[0]):
    save_image(stage1_frames[i], OUT_DIR / "stage1_bbox" / f"bbox_pred_{i:03d}.png")
print(f"✓ Saved {stage1_frames.shape[0]} bbox prediction frames")

# Run Stage 2 inference
print("\n" + "="*80)
print("STAGE 2: VIDEO GENERATION")
print("="*80)

with torch.no_grad():
    video_output = s2v_pipeline(
        init_frame_pil,
        cond_images=stage1_frames.unsqueeze(0).cuda(),
        semantic_ids=semantic_for_stage1,
        use_semantic_vae=use_sem_vae,
        height=128,
        width=512,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        num_inference_steps=25,
        num_frames=25,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        noise_aug_strength=0.02,
        output_type='pt',
        control_condition_scale=1.0
    )

stage2_frames = video_output.frames[0]  # [25, 3, 128, 512]
print(f"✓ Stage 2 output: {stage2_frames.shape}")

# Save Stage 2 outputs
print("\nSaving Stage 2 (final) outputs...")
for i in range(stage2_frames.shape[0]):
    save_image(stage2_frames[i], OUT_DIR / "stage2_generated" / f"generated_{i:03d}.png")
print(f"✓ Saved {stage2_frames.shape[0]} generated video frames")

# Create GIF files
print("\n" + "="*80)
print("CREATING GIF ANIMATIONS")
print("="*80)

def create_gif(frames_tensor, output_path, fps=7):
    """Create GIF from tensor frames [T, C, H, W]"""
    frames_list = []
    for i in range(frames_tensor.shape[0]):
        frame_np = (frames_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames_list.append(Image.fromarray(frame_np))
    
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=int(1000/fps),
        loop=0
    )

create_gif(rgb_clips, OUT_DIR / "ground_truth.gif", fps=7)
print("✓ Saved ground_truth.gif")

create_gif(stage1_frames.cpu(), OUT_DIR / "stage1_bbox_prediction.gif", fps=7)
print("✓ Saved stage1_bbox_prediction.gif")

create_gif(stage2_frames.cpu(), OUT_DIR / "stage2_generated_video.gif", fps=7)
print("✓ Saved stage2_generated_video.gif")

print("\n" + "="*80)
print("✅ FULL INFERENCE TEST COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"\nContents:")
print(f"  - ground_truth.gif (original RGB frames)")
print(f"  - stage1_bbox_prediction.gif (semantic predictions)")
print(f"  - stage2_generated_video.gif (final generated video)")
print(f"  - ground_truth/ ({rgb_clips.shape[0]} frames)")
print(f"  - stage1_bbox/ ({stage1_frames.shape[0]} frames)")
print(f"  - stage2_generated/ ({stage2_frames.shape[0]} frames)")
print(f"  - semantic_viz/ ({semantic_ids.shape[0]} frames)")
print(f"  - latents/ (RGB + Semantic latent visualizations)")

PYTEST

END_TIME=$(date +%s)
echo ""
echo "========================================="
echo "Full Inference Test Complete!"
echo "========================================="
echo "Finished at: $(date)"
echo "Results: /usrhomes/s1492/Ctrl-V-seg/tests/full_inference_output/"
echo "========================================="
