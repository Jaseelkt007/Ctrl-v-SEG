# KITTI-360 Semantic Evaluation Pipeline - Complete Code Map

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ eval_kitti360_sem_overall.sh                                     │
│ └─> accelerate launch tools/eval_overall.py                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ DATASET LOADING (util.py:57-63)                                 │
│ ├─> get_dataloader()                                            │
│ │   └─> KITTI360BDDDataset                                      │
│ │       ├─> Reads from: /no_backups/s1492/kitti360_ctrlv/      │
│ │       ├─> Returns: RGB images + semantic RGB images          │
│ │       └─> Clip length: 25 frames                             │
│ └─> eval_samples_generator()                                    │
│     └─> Yields samples one by one (batch_size=1)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: SEMANTIC PREDICTION (eval_overall.py:77-115)          │
│                                                                  │
│ FOR EACH GUIDANCE SCALE COMBINATION (5 attempts):               │
│ ┌──────────────────────────────────────────────────────────────┤
│ │ 1. Load Models (lines 263-276)                               │
│ │    ├─> bbox_pipeline = VideoDiffusionPipeline               │
│ │    │   └─> UNet from: SEM_PRED_DIR_PARENT (auto-detect)     │
│ │    └─> Model trained to predict semantic segmentation        │
│ │                                                               │
│ │ 2. Run Inference (lines 86-96)                               │
│ │    bbox_pipeline(                                            │
│ │      image_init,              # First frame (1x3xHxW)        │
│ │      bbox_images,             # 25 semantic GT frames        │
│ │      height=128, width=512,                                  │
│ │      num_frames=25,                                          │
│ │      num_inference_steps=30,                                 │
│ │      min_guidance_scale,      # [1,1,2,2,3]                 │
│ │      max_guidance_scale,      # [2,3,4,5,5]                 │
│ │      noise_aug_strength=0.01, # Gaussian noise              │
│ │      num_cond_bbox_frames=1   # Condition on frame 0        │
│ │    )                                                         │
│ │         ↓                                                    │
│ │    ┌─────────────────────────────────────────────────────┐ │
│ │    │ INSIDE pipeline_video_diffusion.py:56-315          │ │
│ │    │                                                      │ │
│ │    │ A. Encode First Frame (lines 172-193)              │ │
│ │    │    ├─> Add Gaussian noise to init image            │ │
│ │    │    │   noise = randn_tensor(...)                   │ │
│ │    │    │   image = image + 0.01 * noise                │ │
│ │    │    ├─> Encode through VAE                          │ │
│ │    │    └─> image_embeddings (CLIP encoding)            │ │
│ │    │                                                      │ │
│ │    │ B. Prepare Conditioning (lines 200-207)            │ │
│ │    │    ├─> Encode bbox_images through VAE              │ │
│ │    │    │   cond_latents = _encode_vae_condition()      │ │
│ │    │    ├─> Replace frame 0 with GT semantic:           │ │
│ │    │    │   image_latents[:,0,:,:,:] = cond_latents[:,0]│ │
│ │    │    └─> Replace frame 24 with GT semantic:          │ │
│ │    │        image_latents[:,-1,:,:,:] = cond_latents[:,-1]│
│ │    │                                                      │ │
│ │    │ C. Prepare Latents (lines 234-245)                 │ │
│ │    │    └─> Random noise: randn_tensor(shape)           │ │
│ │    │                                                      │ │
│ │    │ D. Denoising Loop (lines 258-293)                  │ │
│ │    │    FOR each timestep t in [T, T-1, ..., 0]:       │ │
│ │    │      ├─> Concat: [latents, image_latents]          │ │
│ │    │      ├─> UNet forward:                             │ │
│ │    │      │   noise_pred = unet(                        │ │
│ │    │      │     latent_model_input,                     │ │
│ │    │      │     t,                                      │ │
│ │    │      │     encoder_hidden_states,                  │ │
│ │    │      │     added_time_ids                          │ │
│ │    │      │   )                                         │ │
│ │    │      ├─> Classifier-Free Guidance:                 │ │
│ │    │      │   noise_pred = uncond + guidance_scale *    │ │
│ │    │      │                (cond - uncond)              │ │
│ │    │      └─> Scheduler step:                           │ │
│ │    │          latents = scheduler.step(noise_pred, t, latents)│
│ │    │                                                      │ │
│ │    │ E. Decode Latents (lines 295-303)                  │ │
│ │    │    ├─> frames = decode_latents(latents, 25, 8)    │ │
│ │    │    ├─> Clamp to [-1, 1]                            │ │
│ │    │    └─> Convert to PIL/np/pt format                 │ │
│ │    └─────────────────────────────────────────────────────┘ │
│ │         ↓                                                    │
│ │    Output: bbox_im (torch tensor, values in [0,1], shape: 25x3xHxW)│
│ │                                                               │
│ │ 3. Post-Process (lines 97-105) ⚠️ BUG HERE                  │
│ │    bbox_frames = (bbox_im * 255).numpy().astype(uint8)      │
│ │                                                               │
│ │    # Filter 1: Remove completely empty frames                │
│ │    tmp = bbox_frames.sum(axis=1) < 50                        │
│ │    bbox_frames[tmp] = 0                                      │
│ │                                                               │
│ │    # Filter 2: ❌ BUGGY - REMOVES VALID FRAMES              │
│ │    for frame_i in range(1, 24):  # Skip first & last        │
│ │        if bbox_frames[frame_i].sum(axis=0).min() > 50:      │
│ │            # This sets frames with content to BLACK!         │
│ │            bbox_frames[frame_i] = np.zeros_like(...)        │
│ │                                                               │
│ │ 4. Evaluate Metrics (line 107)                               │
│ │    clip_miou, clip_ap, clip_ar = binary_mask_iou(           │
│ │        sample['bbox_img_np'][:25],  # GT                    │
│ │        bbox_frames                   # Corrupted predictions!│
│ │    )                                                          │
│ └──────────────────────────────────────────────────────────────┤
│                                                                  │
│ 5. Select Best (lines 108-115)                                 │
│    best_generation_bbox = bbox_im with highest mIoU            │
│    best_generation_np = corresponding bbox_frames (numpy)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: RGB VIDEO GENERATION (eval_overall.py:164-178)        │
│                                                                  │
│ 1. Load Models (lines 278-284)                                 │
│    ├─> ctrlnet = ControlNetModel (semantic2video)             │
│    ├─> unet = UNetSpatioTemporalConditionModel                │
│    └─> pipeline = StableVideoControlPipeline                   │
│                                                                  │
│ 2. Run Inference (lines 165-176)                               │
│    frames = ctrl_pipeline(                                      │
│      image_init,                                                │
│      cond_images=2*(best_generation_bbox-0.5),  # Normalize    │
│      height=128, width=512,                                     │
│      num_inference_steps=50,                                    │
│      num_frames=25,                                             │
│      control_condition_scale,                                   │
│      min_guidance_scale=1.0,                                    │
│      max_guidance_scale=5.0,                                    │
│      noise_aug_strength=0.01                                    │
│    )                                                            │
│         ↓                                                       │
│    (Similar diffusion process with ControlNet conditioning)    │
│         ↓                                                       │
│    Output: frames (torch tensor, RGB video)                    │
│                                                                  │
│ 3. Post-Process (lines 181-182)                                │
│    frames = (frames * 255).numpy().astype(uint8)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LOGGING TO WANDB (eval_overall.py:146-255)                     │
│                                                                  │
│ For each frame i in [0, 24]:                                   │
│   ├─> Stage 1 Prediction:                                      │
│   │   {orig_name}_pred_sem.png = best_generation_np[i]        │
│   │                                                             │
│   ├─> Stage 1 Ground Truth:                                    │
│   │   {orig_name}_gt_sem.png = sample['segmentation_np'][i]   │
│   │                                                             │
│   ├─> Stage 2 Generated RGB:                                   │
│   │   {orig_name}_generated.png = frames[i]                    │
│   │                                                             │
│   └─> Stage 2 Ground Truth RGB:                                │
│       {orig_name}_gt.png = sample['gt_clip_np'][i]            │
│                                                                  │
│ Also log videos (lines 234-249):                               │
│   ├─> {scene_id}_{i}_stage1_pred_sem_video.mp4               │
│   ├─> {scene_id}_{i}_generated_video.mp4                      │
│   ├─> {scene_id}_{i}_gt_video.mp4                             │
│   └─> {scene_id}_{i}_gt_sem_video.mp4                         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

```python
# Sample from dataset
sample = {
    'image_init': PIL.Image,           # First frame (conditioning)
    'gt_clip': torch.Tensor,           # [25, 3, H, W] RGB ground truth
    'gt_clip_np': np.ndarray,          # [25, 3, H, W] uint8 numpy
    'bbox_img': torch.Tensor,          # [25, 3, H, W] semantic RGB
    'bbox_img_np': np.ndarray,         # [25, 3, H, W] uint8 semantic
    'segmentation_np': np.ndarray,     # Same as bbox_img_np
    'image_paths': List[str],          # 25 frame paths
    'objects_tensors': Dict,           # Bounding box annotations
}

# Stage 1 output
bbox_im = torch.Tensor           # [25, 3, H, W], values in [0, 1]
bbox_frames = np.ndarray         # [25, 3, H, W], uint8 [0, 255]

# Stage 2 output  
frames = torch.Tensor            # [25, 3, H, W], values in [0, 1]
```

## Noise Specification

### Conditioning Image Noise (Stage 1 & 2)
- **Type**: Simple Gaussian addition
- **Strength**: 0.01 (very small)
- **Location**: `pipeline_video_diffusion.py:180-181`
```python
noise = torch.randn_like(image)  # N(0, 1)
image = image + 0.01 * noise     # Slight perturbation
```

### Latent Noise (Initial state for diffusion)
- **Type**: Random Gaussian
- **Location**: `pipeline_video_diffusion.py:235-245`
```python
latents = torch.randn(
    (batch_size, num_frames, num_channels, H//8, W//8)
)
```

### Denoising Process
- **Scheduler**: DDPM or DDIM (from pretrained SVD model)
- **Steps**: 30 (Stage 1), 50 (Stage 2)
- **Process**: Iteratively removes noise:
  ```
  for t in timesteps:
      latents = scheduler.step(noise_pred, t, latents)
  ```

## Frame Conditioning Strategy

### Stage 1 (Semantic Prediction)
```python
# Line 103: num_cond_bbox_frames=1
image_latents[:, 0, :, :, :] = cond_latents[:, 0, :, :, :]   # Frame 0 = GT
image_latents[:, -1, :, :, :] = cond_latents[:, -1, :, :, :]  # Frame 24 = GT
# Frames 1-23: Model predicts
```

### Stage 2 (RGB Generation)
Uses predicted semantics from Stage 1 as ControlNet conditioning:
```python
cond_images = 2 * (best_generation_bbox - 0.5)  # Normalize to [-1, 1]
```

## Critical Issue Summary

**Line 101-105 in eval_overall.py:**
```python
for frame_i in range(1, bbox_frames.shape[0]-1):
    if bbox_frames[frame_i].sum(axis=0).min() > 50:
        bbox_frames[frame_i] = np.zeros_like(bbox_frames[frame_i])
```

**Problem**: Sets frames to BLACK when they have content (inverted logic)

**Impact**:
1. Valid semantic predictions are zeroed out
2. Metrics (mIoU, AP, AR) computed on corrupted data
3. Stage 2 receives black frames as conditioning → worse RGB output
4. Significant underestimation of model quality

**Fix**: Remove lines 101-105 or invert the condition.

## File References

| File | Purpose |
|------|---------|
| `eval_kitti360_sem_overall.sh` | Bash script entry point |
| `tools/eval_overall.py` | Main evaluation orchestration |
| `src/ctrlv/utils/util.py` | Dataset loading utilities |
| `src/ctrlv/datasets/kitti360_bdd_format.py` | KITTI-360 dataset class |
| `src/ctrlv/datasets/bdd100k.py` | Base dataset class |
| `src/ctrlv/pipelines/pipeline_video_diffusion.py` | Stage 1 pipeline |
| `src/ctrlv/pipelines/pipeline_video_control.py` | Stage 2 pipeline |
| `src/ctrlv/models/` | UNet and ControlNet models |
| `src/ctrlv/metrics/` | Evaluation metrics |
