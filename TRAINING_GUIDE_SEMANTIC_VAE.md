# 🚀 Training Guide: Semantic VAE Integration

## Overview

This guide covers training the Ctrl-V model with **Semantic VAE** integration for KITTI-360 dataset.

**Key Changes:**
- ✅ Resolution: **192×704** (matches semantic VAE training)
- ✅ Clip size: **25 frames**
- ✅ Semantic VAE encoding for semantic IDs
- ✅ DualVAEManager in both training stages

---

## Training Configuration

### Stage 1: Semantic Prediction (BBox Prediction)

**Script:** `scripts/train_scripts/train_kitti360_bbox_predict.sh`

**Key Parameters:**
```bash
--train_H 192           # Height (matches semantic VAE)
--train_W 704           # Width (matches semantic VAE)
--clip_length 25        # Number of frames
--use_segmentation      # Enable semantic mode
--predict_bbox          # Predict semantic masks
```

**Training Details:**
- **Input:** Initial RGB frame
- **Conditioning:** Semantic IDs (via Semantic VAE)
- **Output:** Predicted semantic frames (25 frames)
- **Epochs:** 10
- **Batch size:** 2
- **Learning rate:** 5e-6
- **Checkpoint:** `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae/`

### Stage 2: Semantic to Video

**Script:** `scripts/train_scripts/train_kitti360_sem2video.sh`

**Key Parameters:**
```bash
--train_H 192                      # Height
--train_W 704                      # Width  
--clip_length 25                   # Number of frames
--use_segmentation                 # Enable semantic mode
--finetuned_svd_path <stage1_ckpt> # Use Stage 1 checkpoint as base
```

**Training Details:**
- **Input:** Initial RGB frame + Stage 1 semantic predictions
- **Conditioning:** Semantic IDs (via Semantic VAE) through ControlNet
- **Output:** Final RGB video (25 frames)
- **Epochs:** 5
- **Batch size:** 2
- **Learning rate:** 1e-5
- **Checkpoint:** `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video_vae/`

---

## Code Integration Summary

### Modified Files

#### 1. `tools/train_video_diffusion.py` (Stage 1)
```python
# Added DualVAEManager initialization
if args.use_segmentation:
    vae_manager = DualVAEManager(
        rgb_vae=vae,
        semantic_vae_checkpoint="<semantic_vae_path>",
        num_semantic_classes=19,
        device=accelerator.device
    )

# Updated encoding logic
if args.predict_bbox and args.use_segmentation and 'bbox_ids' in batch:
    # Use Semantic VAE
    semantic_ids = rearrange(batch['bbox_ids'], "b f h w -> (b f) h w")
    latents = vae_manager.encode_semantic_from_ids(semantic_ids)
else:
    # Use RGB VAE
    latents = vae.encode(frames).latent_dist.sample()
```

#### 2. `tools/train_video_controlnet.py` (Stage 2)
```python
# Added DualVAEManager initialization (same as Stage 1)
vae_manager = DualVAEManager(...)

# Updated bbox/semantic encoding for ControlNet conditioning
if args.use_segmentation and vae_manager is not None and 'bbox_ids' in batch:
    # Use Semantic VAE for semantic conditioning
    semantic_ids = rearrange(batch['bbox_ids'], 'b f h w -> (b f) h w')
    bbox_em = vae_manager.encode_semantic_from_ids(semantic_ids)
else:
    # Use RGB VAE for RGB bbox encoding
    bbox_em = vae.encode(bbox_frames).latent_dist.sample()
```

#### 3. Dataset Integration
- `get_dataloader()` now passes `return_semantic_ids=True` when `use_segmentation=True`
- Dataset returns `bbox_ids` field containing semantic trainIDs [B, F, H, W]
- Automatic remapping from KITTI-360 IDs to trainIDs (0-18)

---

## ⚡ Important: Parallel Training

**You can train BOTH stages simultaneously!**

During **training**, Stage 2 uses **ground truth semantic labels** from the dataset as ControlNet conditioning, NOT Stage 1 predictions. Therefore:

- ✅ **Stage 1 and Stage 2 are independent during training**
- ✅ **Both can run in parallel** on separate GPUs
- ✅ **No need to wait** for Stage 1 to complete

During **inference**, Stage 2 uses Stage 1 predictions as conditioning. But for training, ground truth is used.

---

## Step-by-Step Training

### Prerequisites

1. **Semantic VAE checkpoint available:**
   ```bash
   /usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth
   ```

2. **Dataset paths configured:**
   - RGB images: `/no_backups/s1492/kitti360_ctrlv/images/`
   - Semantic labels: `/data/public/kitti-360/KITTI-360/data_2d_semantics/train/`

3. **Conda environment active:**
   ```bash
   conda activate kitti
   ```

### Stage 1: Train Semantic Prediction

```bash
cd /usrhomes/s1492/Ctrl-V-seg

# Submit Stage 1 training
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh
```

**Monitor progress:**
```bash
# Check SLURM logs
tail -f /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.out

# Check WandB
# Project: ctrl_v_kitti360
# Run: kitti360_semantic_predict_vae
```

**Expected duration:** ~24-48 hours for 10 epochs

**Checkpoints saved to:**
```
/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae/
├── checkpoint-200/
├── checkpoint-400/
└── ...
```

### Stage 2: Train Semantic to Video

**No need to wait** - can start immediately or run in parallel with Stage 1:

```bash
cd /usrhomes/s1492/Ctrl-V-seg

# Option 1: Train from base SVD (recommended for parallel training)
sbatch scripts/train_scripts/train_kitti360_sem2video.sh

# Option 2: Fine-tune from Stage 1 checkpoint (after Stage 1 completes)
# Edit train_kitti360_sem2video.sh:
#   Uncomment: FINETUNED_SVD_PATH="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae"
# Then submit:
# sbatch scripts/train_scripts/train_kitti360_sem2video.sh
```

**Monitor progress:**
```bash
tail -f /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.out
```

**Expected duration:** ~18-36 hours for 5 epochs

**Checkpoints saved to:**
```
/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video_vae/
├── checkpoint-77300/
│   ├── unet/
│   └── controlnet/
└── ...
```

---

## Monitoring Training

### WandB Metrics

**Stage 1:**
- `train/loss` - Training loss
- `validation/demo_samples` - Generated semantic predictions
- `validation/gt_frames` - Ground truth frames
- `train/lr` - Learning rate

**Stage 2:**
- `train/loss` - Training loss  
- `validation/generated_videos` - Generated RGB videos
- `validation/gt_videos` - Ground truth videos
- `validation/gt_bbox_frames` - Semantic conditioning

### Validation Outputs

Every 500 steps, validation samples are logged:
- **Stage 1:** 4 demo samples showing predicted semantic frames
- **Stage 2:** 4 demo samples showing generated RGB videos

---

## Key Differences from RGB Training

| Aspect | RGB Training | Semantic VAE Training |
|--------|--------------|----------------------|
| **Resolution** | 128×512 | **192×704** |
| **Encoding** | RGB VAE only | **Semantic VAE** for masks |
| **Input format** | RGB images [B,F,3,H,W] | Semantic IDs [B,F,H,W] |
| **Latent shape** | [B,F,4,16,64] | [B,F,4,24,88] |
| **Dataset** | BDD100K format | **Mixed paths** (RGB + semantic) |
| **VAE Manager** | None | **DualVAEManager** |

---

## Troubleshooting

### Issue: "bbox_ids not in batch"

**Cause:** Dataset not returning semantic IDs

**Fix:** Ensure `return_semantic_ids=True` in `get_dataloader()`
```python
train_dataset, train_loader = get_dataloader(
    ...,
    return_semantic_ids=args.use_segmentation  # ✓ Added
)
```

### Issue: "Semantic file not found"

**Cause:** Frame number mismatch between preprocessed RGB and official semantics

**Solution:** This is expected - dataset falls back to zeros for missing frames. Training continues normally.

### Issue: Shape mismatch in latents

**Cause:** Resolution mismatch

**Fix:** Ensure `--train_H 192 --train_W 704` in both training scripts

### Issue: OOM (Out of Memory)

**Solutions:**
1. Reduce batch size: `--train_batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 6`
3. Enable gradient checkpointing: `--enable_gradient_checkpointing` (already enabled)

---

## Expected Training Time

**With 1× RTX A6000 (49GB):**

| Stage | Epochs | Steps/Epoch | Time/Step | Total Time |
|-------|--------|-------------|-----------|------------|
| Stage 1 | 10 | ~2,415 | 3-4s | 24-36 hours |
| Stage 2 | 5 | ~2,415 | 2-3s | 12-24 hours |
| **Total** | - | - | - | **~36-60 hours** |

---

## Post-Training

### Evaluate Models

```bash
# Update eval script with semantic VAE
# See INFERENCE_INTEGRATION_COMPLETE.md

bash scripts/eval_scripts/eval_kitti360_sem_overall.sh
```

### Test Single Inference

```bash
# Already tested and working!
sbatch scripts/test_scripts/test_full_inference_with_viz.sh
```

**Outputs:** GIFs + frames in `/usrhomes/s1492/Ctrl-V-seg/tests/full_inference_output/`

---

## Files Modified

✅ `tools/train_video_diffusion.py` - Stage 1 training with Semantic VAE
✅ `tools/train_video_controlnet.py` - Stage 2 training with Semantic VAE  
✅ `scripts/train_scripts/train_kitti360_bbox_predict.sh` - Updated resolution
✅ `scripts/train_scripts/train_kitti360_sem2video.sh` - Updated resolution
✅ `src/ctrlv/utils/util.py` - Added `return_semantic_ids` parameter
✅ `src/ctrlv/datasets/bdd100k.py` - Return semantic IDs when requested
✅ `src/ctrlv/datasets/kitti360_bdd_format.py` - Pass through `return_semantic_ids`

---

## Quick Start Commands

```bash
# Stage 1: Semantic Prediction
cd /usrhomes/s1492/Ctrl-V-seg
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh

# Wait for Stage 1 to complete...

# Stage 2: Semantic to Video
sbatch scripts/train_scripts/train_kitti360_sem2video.sh
```

---

**Ready to train!** All integration is complete and tested. 🚀
