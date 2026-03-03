# Stage 1 Semantic Diffusion: Root Cause Analysis & Fix

## Executive Summary

Stage 1 (RGB → Semantic prediction via diffusion) produced ~4% mIoU on generated frames (1-23), while conditioned frames (0, 24) achieved 32-65% mIoU. The root cause is a **latent space mismatch** introduced when adapting the original Ctrl-V bbox pipeline for semantic prediction. The conditioning for intermediate frames (1-23) was encoded via the RGB VAE, while target latents and anchor frames (0, 24) were encoded via the Semantic VAE — two completely different latent spaces. The fix ensures all conditioning uses the Semantic VAE consistently.

---

## 1. Background: How Ctrl-V Conditioning Works

The Ctrl-V diffusion model uses SVD (Stable Video Diffusion) architecture with 8-channel UNet input:
- **4 channels**: noisy target latents (to be denoised)
- **4 channels**: conditioning latents (concatenated)

The conditioning latent tensor `[B, 25, 4, H, W]` is constructed as:
- **Frame 0**: GT encoded latent of first frame (anchor)
- **Frames 1-23**: Encoded initial/first frame repeated (background fill)
- **Frame 24**: GT encoded latent of last frame (anchor)

The UNet learns to denoise the target by interpolating between the two anchors, using the background fill as a default signal for the middle frames.

## 2. Original Ctrl-V (bbox, working) — Single Latent Space

In `~/Ctrl-V/tools/train_video_diffusion.py`:

```python
# ALL frames encoded with the SAME RGB VAE → latent space A
frames = rearrange(batch['bbox_images'], ...)
latents = vae.encode(frames).latent_dist.sample()           # RGB VAE → space A

initial_frame_latent = vae.encode(initial_images).latent_dist.sample()  # RGB VAE → space A

conditional_latents = latents.clone()                                    # space A
conditional_latents[:, 1:-1, :] = initial_frame_latent.repeat(...)       # space A
```

**Result**: All conditioning is in latent space A. Consistent. The model learns effectively.

## 3. Ctrl-V-seg (semantic, broken) — Mixed Latent Spaces

In `~/Ctrl-V-seg/tools/train_video_diffusion.py` (before fix):

```python
# Target frames encoded with SEMANTIC VAE → latent space B
semantic_ids = rearrange(batch['semantic_ids'], ...)
latents = vae_manager.encode_semantic_from_ids(semantic_ids)  # Semantic VAE → space B

# Initial frame encoded with RGB VAE → latent space A (BUG!)
initial_frame_latent = vae.encode(initial_images).latent_dist.sample()  # RGB VAE → space A

conditional_latents = latents.clone()                                    # space B (frames 0, 24)
conditional_latents[:, 1:-1, :] = initial_frame_latent.repeat(...)       # space A (frames 1-23)!
```

**Result**: Conditioning mixes two incompatible latent spaces:
- Frames 0, 24: Semantic VAE latents (space B) — correct anchors
- Frames 1-23: RGB VAE latents (space A) — wrong space, meaningless to the UNet

The RGB VAE encodes RGB pixel distributions (ImageNet/LAION trained). The Semantic VAE encodes one-hot semantic logits through a learned stem → frozen encoder core. These produce completely different distributions in the 4D latent space.

### Why the Loss Still Dropped

Training loss went from 0.35 → 0.00005, which appears normal but is misleading:
- Frames 0 and 24 have GT information in conditioning (trivially easy to predict)
- The loss is averaged over all 25 frames and all timesteps
- High-noise timesteps contribute less meaningful signal
- The model optimized for the easy frames, masking poor learning on frames 1-23

## 4. Same Issue in Inference Pipeline

In `pipeline_video_diffusion.py` (before fix):

```python
# Base image_latents for ALL 25 frames: RGB VAE → space A
image_latents = self._encode_vae_image(image, ...)  # RGB VAE → space A
image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

# Overwrite frames 0 and 24 with Semantic VAE → space B
cond_latents = self._encode_vae_condition(..., use_semantic_vae=True)  # space B
image_latents[:, 0:1, :] = cond_latents[:, 0:1, :]   # space B
image_latents[:, -1, :]  = cond_latents[:, -1, :]     # space B
```

Same mismatch: frames 1-23 in space A, frames 0/24 in space B.

## 5. Other Investigated Issues

### Scaling Factor (`vae.config.scaling_factor = 0.18215`)

Both projects apply `latents * 0.18215` to target latents. In the original, this is the correct RGB VAE scaling. In ours, it's applied to Semantic VAE outputs — conceptually wrong (Semantic VAE has no built-in scaling), but numerically it's just a constant multiplier that doesn't cause the main failure. The model can adapt to any consistent scale.

**Verdict**: Not the root cause. No change needed — the constant multiplier is absorbed by model weights during training.

### Classifier-Free Guidance (`conditioning_dropout_prob`)

Both projects use `conditioning_dropout_prob=0.0` (no dropout during training). Both use CFG at inference (`guidance_scale 3-7`). The original project works with this configuration because SVD was pre-trained with CFG support. The base model's unconditional behavior is inherited.

**Verdict**: Not the root cause. Identical between projects. Could add 0.1 dropout for marginal improvement, but not required.

### image_latents Overwriting (Bug 5)

The overwriting pattern in the pipeline is correct and matches training exactly. Frames 0 and 24 are replaced with per-frame semantic conditioning; frames 1-23 keep the repeated initial frame encoding. This is the intended Ctrl-V design.

**Verdict**: Not a bug. Same pattern in both projects.

## 6. The Fix

### Training (`train_video_diffusion.py`)

```python
# BEFORE (broken): RGB VAE for initial_frame_latent
initial_frame_latent = vae.encode(initial_images.to(weight_dtype)).latent_dist.sample()

# AFTER (fixed): Semantic VAE for initial_frame_latent when using semantic mode
if args.predict_bbox and args.use_segmentation and vae_manager is not None:
    first_frame_sem_ids = batch['semantic_ids'][:, 0, :, :]
    initial_frame_latent = vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
else:
    initial_frame_latent = vae.encode(initial_images.to(weight_dtype)).latent_dist.sample()
```

### Inference Pipeline (`pipeline_video_diffusion.py`)

```python
# BEFORE (broken): RGB VAE for base image_latents
image_latents = self._encode_vae_image(image, ...)

# AFTER (fixed): Semantic VAE for base image_latents when using semantic mode
if use_semantic_vae and semantic_ids is not None and hasattr(self, 'vae_manager'):
    first_frame_sem_ids = semantic_ids[:, 0, :, :].to(device=device)
    image_latents = self.vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
    # Handle CFG batch doubling
    ...
else:
    image_latents = self._encode_vae_image(image, ...)
```

### Training Validation Inference

Also fixed to pass `semantic_ids` and `use_semantic_vae=True` to the pipeline during training validation, ensuring consistency.

## 7. Files Modified

| File | Change |
|------|--------|
| `tools/train_video_diffusion.py` | Use Semantic VAE for `initial_frame_latent` when in semantic mode; pass `semantic_ids` to pipeline during validation |
| `src/ctrlv/pipelines/pipeline_video_diffusion.py` | Use Semantic VAE for base `image_latents` (frames 1-23) when `use_semantic_vae=True` |

## 8. Expected Impact

After retraining with this fix, all conditioning latents will be in the same Semantic VAE latent space:
- **Frames 0, 24**: Per-frame semantic latents (anchors)
- **Frames 1-23**: First frame's semantic latent repeated (consistent background)
- **Target**: All 25 frames' semantic latents

This mirrors exactly how the original Ctrl-V works with bbox images (all RGB VAE). The UNet should now be able to learn meaningful temporal interpolation between the semantic anchors.

## 9. Evaluation Results (Pre-Fix)

| Metric | Value |
|--------|-------|
| Overall mIoU | 4.14% |
| Overall Pixel Accuracy | 15.01% |
| Mean Class Accuracy | 9.91% |
| Checkpoint Step | 49200 |
| Samples Evaluated | 15 |

Top per-class IoU (all very low):
- road: 12.98%
- car: 10.77%
- building: 7.28%
- vegetation: 7.01%

Post-fix results will be obtained after retraining.
