#!/bin/bash
#SBATCH --job-name=single_sample_inf
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/single_sample_inf_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/single_sample_inf_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G 
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=02:00:00

set -e
set -u

echo "========================================="
echo "Single Sample Inference Test with Semantic VAE"
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
OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/single_sample_test"
mkdir -p "$OUT_DIR"

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

# Run single sample inference test
python3 << 'PYTEST'
import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V-seg/src')
sys.path.insert(0, '/usrhomes/s1492/vae_semantic')

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os

print("\n[1/5] Loading dataset...")
from ctrlv.datasets.kitti360_inference import KITTI360InferenceDataset
from ctrlv.utils import get_dataloader

dataset = KITTI360InferenceDataset(
    root='/no_backups/s1492/',
    train=False,  # Use val set
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
sample = dataset[0]
print(f"✓ Sample loaded:")
print(f"  RGB clips: {sample[0].shape}")
print(f"  BBox images (RGB, viz): {sample[4].shape}")
if len(sample) > 5:
    print(f"  Semantic IDs (trainIDs): {sample[5].shape}")
    print(f"  Semantic ID range: [{sample[5].min()}, {sample[5].max()}]")

print("\n[2/5] Initializing DualVAEManager...")
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

print("\n[3/5] Testing encoding...")
from einops import rearrange

# Test RGB encoding
rgb_frames = sample[0].unsqueeze(0).cuda()  # [1, 25, 3, 128, 512]
rgb_flat = rearrange(rgb_frames, "b f c h w -> (b f) c h w")
rgb_latents = vae_manager.encode_rgb(rgb_flat)
rgb_latents = rearrange(rgb_latents, "(b f) c h w -> b f c h w", b=1)
print(f"✓ RGB encoding: {rgb_frames.shape} -> {rgb_latents.shape}")

# Test Semantic encoding
if len(sample) > 5:
    semantic_ids = sample[5].unsqueeze(0).cuda()  # [1, 25, 128, 512]
    semantic_flat = rearrange(semantic_ids, "b f h w -> (b f) h w")
    semantic_latents = vae_manager.encode_semantic_from_ids(semantic_flat)
    semantic_latents = rearrange(semantic_latents, "(b f) c h w -> b f c h w", b=1)
    print(f"✓ Semantic encoding: {semantic_ids.shape} -> {semantic_latents.shape}")
    
    assert rgb_latents.shape == semantic_latents.shape, "Latent shapes should match"
    print(f"✓ Latent shapes match: {rgb_latents.shape}")
    
    # Save visualization
    print("\n[4/5] Saving visualizations...")
    out_dir = Path("/no_backups/s1492/Ctrl-V/outputs/single_sample_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first RGB frame
    from torchvision.utils import save_image
    save_image(rgb_frames[0, 0], out_dir / "rgb_frame_0.png")
    print(f"✓ Saved RGB frame: {out_dir / 'rgb_frame_0.png'}")
    
    # Save semantic visualization
    from ctrlv.utils.semantic_preprocessing import semantic_ids_to_viz_rgb
    semantic_viz = semantic_ids_to_viz_rgb(semantic_ids[0, 0].cpu().numpy())
    Image.fromarray(semantic_viz).save(out_dir / "semantic_frame_0.png")
    print(f"✓ Saved semantic frame: {out_dir / 'semantic_frame_0.png'}")
    
    print("\n[5/5] Testing complete!")
    print(f"\nResults saved to: {out_dir}")
    print("\n✅ Single sample inference test PASSED!")
else:
    print("❌ No semantic IDs in sample")
    exit(1)

PYTEST

END_TIME=$(date +%s)
echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo "Finished at: $(date)"
echo "Results: /no_backups/s1492/Ctrl-V/outputs/single_sample_test/"
echo "========================================="
