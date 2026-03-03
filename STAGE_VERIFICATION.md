# Two-Stage Training Verification

## Summary
✓ **Stage 1**: Generates **25 semantic frames** from 1 RGB frame  
✓ **Stage 2**: Generates **25 RGB frames** from 25 semantic frames  

Both stages correctly use `clip_length=25` and process full video clips.

---

## Stage 1: train_video_diffusion.py (RGB → Semantic)

### Purpose
Train diffusion model to generate semantic segmentation video from a single RGB image.

### Input (Conditioning)
```python
# Line 447-448
batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]  # video_length = 25
initial_images = batch['clips'][:,0,:,:,:]  # First RGB frame [B, 3, H, W]
```
- Uses **first RGB frame** as conditioning image
- `video_length = 25` frames per clip

### Target (What to Generate)
```python
# Lines 458-462
if args.predict_bbox and args.use_segmentation and vae_manager is not None and batch.get('semantic_ids') is not None:
    # Use Semantic VAE for semantic ID encoding (grayscale trainIDs)
    semantic_ids = rearrange(batch['semantic_ids'], "b f h w -> (b f) h w")  # [B*25, H, W]
    latents = vae_manager.encode_semantic_from_ids(semantic_ids)  # Encode 25 semantic frames
    latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size)  # [B, 25, C, H, W]
```

**Key point**: `batch['semantic_ids']` has shape `[B, 25, H, W]` - **25 semantic frames**

```python
# Line 486
target_latents = latents = latents * vae.config.scaling_factor
```

### Verification
- ✓ Input: 1 RGB frame `[B, 3, H, W]`
- ✓ Target: **25 semantic frames** `[B, 25, H, W]` encoded by Semantic VAE
- ✓ Output: Model learns to generate **25 semantic frames** from single RGB image

---

## Stage 2: train_video_controlnet.py (Semantic → RGB)

### Purpose
Train ControlNet to generate RGB video from semantic segmentation video.

### Input (ControlNet Conditioning)
```python
# Lines 385, 403-407
batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]  # video_length = 25

if args.use_segmentation and vae_manager is not None and batch.get('semantic_ids') is not None:
    # Use Semantic VAE for semantic ID encoding (grayscale trainIDs)
    semantic_ids = rearrange(batch['semantic_ids'], 'b f h w -> (b f) h w')  # [B*25, H, W]
    bbox_em = vae_manager.encode_semantic_from_ids(semantic_ids)  # Encode 25 semantic frames
    bbox_em = rearrange(bbox_em, '(b f) c h w -> b f c h w', f=video_length)  # [B, 25, C, H, W]
```

**Key point**: `batch['semantic_ids']` has shape `[B, 25, H, W]` - **25 semantic frames** used for ControlNet conditioning

### Target (What to Generate)
```python
# Lines 416-419
frames = rearrange(batch['clips'] if not args.generate_bbox else batch['bbox_images'], 'b f c h w -> (b f) c h w')  # [B*25, 3, H, W]
latents = vae.encode(frames).latent_dist.sample()  # Encode 25 RGB frames
latents = rearrange(latents, '(b f) c h w -> b f c h w', f=video_length)  # [B, 25, C, H, W]
target_latents = latents = latents * vae.config.scaling_factor
```

**Key point**: `batch['clips']` has shape `[B, 25, 3, H, W]` - **25 RGB frames**

### Verification
- ✓ Input (ControlNet): **25 semantic frames** `[B, 25, H, W]` encoded by Semantic VAE
- ✓ Target: **25 RGB frames** `[B, 25, 3, H, W]` encoded by RGB VAE
- ✓ Output: Model learns to generate **25 RGB frames** from 25 semantic frames

---

## Dataloader Confirmation

Both stages use the **same dataloader** with `clip_length=25`:

```python
# From test output:
✓ Loaded 49004 frame pairs from 2013_05_28_drive_train_frames.txt
✓ Created 48788 clips from 9 sequences
  Clip length: 25
```

Each batch contains:
- `batch['clips']`: RGB frames `[B, 25, 3, H, W]`
- `batch['semantic_ids']`: Semantic IDs `[B, 25, H, W]`

The **training scripts decide** which is input vs target:
- **Stage 1**: RGB (first frame) → Semantic (all 25 frames)
- **Stage 2**: Semantic (all 25 frames) → RGB (all 25 frames)

---

## Critical Flags

### Stage 1 (train_video_diffusion.py)
Must set:
- `--predict_bbox` (enables semantic target)
- `--use_segmentation` (enables Semantic VAE)
- `--clip_length 25`

### Stage 2 (train_video_controlnet.py)
Must set:
- `--use_segmentation` (enables Semantic VAE for ControlNet)
- `--clip_length 25`
- `--generate_bbox False` (default, to use RGB clips as target)

---

## Conclusion

✅ **VERIFIED**: Both stages correctly process **25-frame clips**:
- Stage 1 generates 25 semantic frames
- Stage 2 generates 25 RGB frames
- Dataloader provides both RGB and semantic for all 25 frames
- VAE encoders (RGB and Semantic) handle the full temporal dimension

**No issues found. Implementation is correct.**
