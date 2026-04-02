# Stage 1 Pipeline: RGB → Semantic Segmentation Prediction

**Task**: Given the first RGB frame of a driving sequence, predict the full 25-frame semantic
segmentation sequence (including all intermediate and future frames).

**Architecture base**: Stable Video Diffusion XT (SVD-XT) — 25-frame UNet
(`UNetSpatioTemporalConditionModel`) with a modified 8-channel input (instead of the original
4-channel) to accept both noisy target latents and conditioning latents concatenated on the
channel dimension.

---

## 0. High-Level Overview

```
KITTI-360 Disk
  RGB frames (.png)          →  [B, T, 3, H, W]     clip of T=25 frames
  Semantic label PNGs        →  [B, T, H, W]         grayscale raw IDs

  ─── Dataset & Collate ──────────────────────────────────────────────────────

  Remap raw IDs → trainIds   →  [B, T, H, W] int64   trainIds 0–18
  Resize (nearest)           →  [B, T, 192, 704]     training resolution

  ─── DualVAEManager ─────────────────────────────────────────────────────────

  Full clip encoding         →  [B, T, 4, 24, 88]    unscaled semantic latents
  First-frame encoding       →  [B, 4, 24, 88]        first-frame semantic latents
  Build conditional_latents  →  [B, T, 4, 24, 88]    spatial-padding conditioning

  ─── Diffusion Setup ────────────────────────────────────────────────────────

  Scale latents × 0.18215   →  [B, T, 4, 24, 88]    target_latents
  Add noise (forward diff.)  →  [B, T, 4, 24, 88]    noisy_latents
  σ-scale for UNet input     →  [B, T, 4, 24, 88]    inp_noisy_latents

  ─── Conditioning Assembly ──────────────────────────────────────────────────

  Channel concat (8 ch)      →  [B, T, 8, 24, 88]    concatenated_noisy_latents
  CLIP image embeddings      →  [B, 1, 1024]          cross-attention conditioning
  Added-time-ids             →  [B, 3]                fps / motion_bucket / noise_aug

  ─── UNet Forward ───────────────────────────────────────────────────────────

  Conv-in (8→320 ch)         →  [B*T, 320, 24, 88]
  4× Down-blocks (cross-attn)→  multi-resolution feature maps
  Mid-block (cross-attn)     →  [B*T, 1280, 3, 11]   (after 3× downsampling)
  4× Up-blocks   (cross-attn)→  multi-resolution feature maps
  Conv-out (320→4 ch)        →  [B*T, 4, 24, 88]

  ─── Loss ───────────────────────────────────────────────────────────────────

  Denoise: c_skip + c_out    →  [B, T, 4, 24, 88]    denoised_latents
  Sigma-weighted MSE loss    →  scalar

  ─── Inference Decoding ─────────────────────────────────────────────────────

  Semantic VAE decoder       →  [B*T, 19, 192, 704]  class logits
  Argmax                     →  [B, T, 192, 704]      predicted trainIds 0–18
```

---

## 1. Dataset Loading and Preprocessing

### 1.1 KITTI-360 Official Dataset

**File**: `src/ctrlv/datasets/kitti360_official.py`
**Class**: `KITTI360OfficialDataset` (inherits `KittiAbstract`)

**Dataset root**: `/misc/data/public/kitti-360/KITTI-360`

**Directory layout used**:
```
KITTI-360/
├── data_2d_raw/{seq}/image_00/data_rect/{frame:010d}.png     ← RGB frames
└── data_2d_semantics/train/
    ├── {seq}/image_00/semantic/{frame:010d}.png              ← semantic labels (grayscale)
    ├── 2013_05_28_drive_train_frames.txt                     ← train split (pairs of paths)
    └── 2013_05_28_drive_val_frames.txt                       ← val split
```

**Split file format** (one clip per line):
```
data_2d_raw/{seq}/{frame}.png  data_2d_semantics/train/{seq}/semantic/{frame}.png
```

### 1.2 Semantic ID Remapping

**File**: `src/ctrlv/utils/semantic_preprocessing.py`
**Function**: `load_and_remap_semantic(path, ignore_index=255)`

KITTI-360 stores raw label IDs (e.g. 7=road, 11=building, ...) that are non-contiguous.
These are remapped to 19 contiguous trainIds (0–18) via `KITTI360_LABEL_MAPPING`:

| Raw ID | trainId | Class      |
|--------|---------|------------|
| 7      | 0       | road       |
| 8      | 1       | sidewalk   |
| 11     | 2       | building   |
| 12     | 3       | wall       |
| 13     | 4       | fence      |
| 17     | 5       | pole       |
| 19     | 6       | traffic light |
| 20     | 7       | traffic sign |
| 21     | 8       | vegetation |
| 22     | 9       | terrain    |
| 23     | 10      | sky        |
| 24     | 11      | person     |
| 25     | 12      | rider      |
| 26     | 13      | car        |
| 27     | 14      | truck      |
| 28     | 15      | bus        |
| 31     | 16      | train      |
| 32     | 17      | motorcycle |
| 33     | 18      | bicycle    |
| all others | 255 | ignored   |

**Output**: `[H, W]` numpy array, dtype int64, values in {0–18, 255}

### 1.3 Clip Assembly

**Function**: `KITTI360OfficialDataset._getclipitem`

For each sample index, T=25 consecutive frames are loaded:

```
clips        = [T, 3, H, W]   float32, range [-1, 1]   (RGB)
semantic_ids = [T, H, W]      int64,   values 0–18      (trainIds)
```

**Resize**: If training resolution (192×704) differs from native resolution, semantic IDs
are resized with `mode='nearest'` to preserve label values (no interpolation artifacts).

### 1.4 Collate Function

**File**: `src/ctrlv/datasets/__init__.py`
**Function**: `kitti_clip_with_bbox_collate_fn`

Batches N clips into:
```python
batch = {
    'clips':        [B, T, 3, H, W]    # RGB frames, float32
    'semantic_ids': [B, T, H, W]        # trainIds, int64
    'bbox_images':  [B, T, 3, H, W]    # semantic visualization (RGB colormap)
    'prompts':      list of strings
    'indices':      list of ints
}
```

With B=1, T=25, H=192, W=704:
- `clips`:        `[1, 25, 3, 192, 704]`
- `semantic_ids`: `[1, 25, 192, 704]`

---

## 2. Semantic VAE Encoding (DualVAEManager)

**File**: `src/ctrlv/models/dual_vae_manager.py`
**Class**: `DualVAEManager`

Both VAEs are **frozen** throughout Stage 1 training. The UNet is the only trainable component.

### 2.1 VAE Initialization

```python
vae_manager = DualVAEManager(
    rgb_vae=vae,   # AutoencoderKLTemporalDecoder from SVD (frozen)
    semantic_vae_checkpoint='/usrhomes/s1492/vae_semantic/checkpoints/
                              semantic_vae_native/best_model_with_dice_boundaryweight.pth',
    num_semantic_classes=19,
    device='cuda',
    clip_size=25,   # must match --clip_length
)
```

- **RGB VAE**: `AutoencoderKLTemporalDecoder` — standard SVD VAE, 8× spatial downsampling,
  4 latent channels. Used only for its `scaling_factor = 0.18215`.
- **Semantic VAE**: `SemanticVAEInference` wrapping `SemanticVAENative` — pretrained on
  KITTI-360. Same 8× spatial downsampling, same 4 latent channels as RGB VAE.

### 2.2 Full Clip Encoding: `encode_semantic_from_ids`

**Input**: `semantic_ids` — `[B*T, H, W]` int64 trainIds

**Step 1 — Temporal grouping**:
```
[B*T, H, W]  →  view as [B, T, H, W]   (requires B*T divisible by clip_size=25)
```
If not divisible, last frame is repeated to pad.

**Step 2 — One-hot encoding**:
```
[B, T, H, W]  →  flatten to [B*T, H, W]
               →  clamp to [0, 18]
               →  F.one_hot(·, num_classes=19)
               →  permute to [B*T, 19, H, W]   float32
```

**Step 3 — Semantic stem** (learned projection, frozen):
```
[B*T, 19, H, W]  →  Conv2d 19→128 + activation  →  [B*T, 128, H, W]
```

**Step 4 — VAE encoder core** `_encode_semantic_features`:
```
[B*T, 128, H, W]  →  3× ResBlock+Downsample  →  [B*T, 4, H/8, W/8]
```
Output is the **mean** of the VAE latent distribution (no reparameterisation at inference/training).

**Step 5 — Reshape and return**:
```
[B*T, 4, H/8, W/8]  (flat, no scaling applied)
```

With B=1, T=25, H=192, W=704 → latent shape: `[25, 4, 24, 88]`

> **Note**: These latents are **unscaled**. The diffusion training loop applies
> `scaling_factor = 0.18215` before adding noise (see Section 4.2).

### 2.3 First-Frame Encoding

The same `encode_semantic_from_ids` is called on the first frame only:
```python
first_frame_sem_ids = batch['semantic_ids'][:, 0, :, :]  # [B, H, W]
initial_frame_latent = vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
# → [B, 4, H/8, W/8]  =  [1, 4, 24, 88]
```

---

## 3. Conditioning Latent Construction (Spatial Padding)

**File**: `tools/train_video_diffusion.py`, lines 647–657

The conditioning tensor `conditional_latents` encodes the "what is known" at each frame
position. It is later concatenated with the noisy target latents on the **channel dimension**
before entering the UNet.

### 3.1 Frame-Slot Assignment (with `num_cond_bbox_frames=1`)

```
Frame index:  0        1  2  3 … 23       24
              ─────────────────────────────────────────
Target:       GT[0]   GT[1] … GT[23]    GT[24]    ← what the model learns to denoise
Conditioning: GT[0]   F0   F0 … F0       GT[24]   ← what the UNet sees as spatial hint
```

Where `F0` = first-frame semantic latent (repeated across all intermediate positions).

**Code**:
```python
conditional_latents = latents.clone()                          # [B, T, 4, H/8, W/8]
# Overwrite frames 1..23 with the first-frame latent
conditional_latents[:, num_cond_bbox_frames:-1, :, :, :] = \
    initial_frame_latent.unsqueeze(1).repeat(
        1, video_length - num_cond_bbox_frames - 1, 1, 1, 1
    )
# Result:
# Frame 0   : ground-truth semantic latent  (actual condition)
# Frames 1-23: first-frame semantic latent  (repeated placeholder)
# Frame 24  : ground-truth semantic latent  (boundary anchor)
```

**Shape**: `conditional_latents` → `[B, T, 4, H/8, W/8]` = `[1, 25, 4, 24, 88]`

### 3.2 Conditioning Dropout (CFG Training)

When `--conditioning_dropout_prob` is set, conditioning is randomly zeroed to train
classifier-free guidance:

```python
# CLIP embedding dropout: fully zeroed with probability = 2 × dropout_prob
null_conditioning = torch.zeros_like(encoder_hidden_states)
encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

# Image conditioning dropout: zeroed for dropout_prob ≤ random_p < 3 × dropout_prob
image_mask = 1 - ((random_p >= dropout_prob) * (random_p < 3 * dropout_prob))
conditional_latents = image_mask * conditional_latents  # [B, 1, 1, 1, 1] broadcast
```

This enables classifier-free guidance at inference with:
- `min_guidance_scale = 1.0` (frame 0: unconditioned)
- `max_guidance_scale = 3.0` (frame 24: fully conditioned)
- Linear schedule between frames.

---

## 4. Diffusion Setup

### 4.1 Noise Scheduler

**Type**: `EulerDiscreteScheduler` (from diffusers)
- `num_train_timesteps = 1000`
- Timesteps sampled uniformly from `[0, 999]` per batch
- Variance schedule: v-prediction / Euler continuous

### 4.2 Target Latent Scaling

```python
target_latents = latents * vae.config.scaling_factor   # 0.18215
# [B, T, 4, H/8, W/8]  unscaled → scaled
```

The `scaling_factor = 0.18215` from the RGB VAE is reused for the semantic latents.
This normalises the latent distribution to approximately unit variance, matching the
noise schedule's assumptions.

### 4.3 Forward Diffusion (Adding Noise)

```python
noise = torch.randn_like(target_latents)               # [B, T, 4, H/8, W/8]
indices = torch.randint(0, 1000, (B,))                 # uniform timestep sampling
timesteps = noise_scheduler.timesteps[indices]         # [B]

noisy_latents = noise_scheduler.add_noise(             # [B, T, 4, H/8, W/8]
    target_latents, noise, timesteps
)
```

### 4.4 Sigma Scaling for UNet Input

SVD uses EDM-style sigma scaling (not the standard DDPM normalisation):

```python
sigmas = get_sigmas(timesteps, ndim=5, dtype)          # [B, 1, 1, 1, 1]
inp_noisy_latents = noisy_latents / sqrt(sigmas² + 1)  # [B, T, 4, H/8, W/8]
```

This places the noisy latent on the unit-variance input hypersphere expected by the UNet.

---

## 5. Channel-Concatenation Conditioning

```python
concatenated_noisy_latents = torch.cat(
    [inp_noisy_latents, conditional_latents], dim=2
)
# [B, T, 4, H/8, W/8]  +  [B, T, 4, H/8, W/8]
# = [B, T, 8, H/8, W/8]   ← 8-channel UNet input
```

With B=1, T=25: shape `[1, 25, 8, 24, 88]`.

The UNet `conv_in` layer has been extended from 4→320 channels to **8→320 channels** to
accommodate this doubled channel count. All other UNet weights are loaded from the pretrained
SVD-XT checkpoint; only the temporal transformer blocks (`temporal_transformer_block.*`) have
their gradients enabled for training.

---

## 6. CLIP Image Conditioning (Cross-Attention)

**Source**: First RGB frame of the clip (not the semantic maps)

### 6.1 Encoding

**File**: `src/ctrlv/utils/util.py`, function `encode_video_image`

```python
# Step 1: Resize to CLIP input size
pixel_values = _resize_with_antialiasing(initial_images, (224, 224))
#              [B, 3, H, W]  →  [B, 3, 224, 224]

# Step 2: Rescale from [-1,1] to [0,1]
pixel_values = (pixel_values + 1.0) * 0.5
pixel_values = torch.clamp(pixel_values, 0., 1.)

# Step 3: Normalize for CLIP (ImageNet mean/std)
pixel_values = feature_extractor(
    images=pixel_values,
    do_normalize=True,
    do_center_crop=False, do_resize=False, do_rescale=False,
    return_tensors="pt"
).pixel_values                                          # [B, 3, 224, 224]

# Step 4: CLIP vision encoder (CLIPVisionModelWithProjection)
image_embeddings = image_encoder(pixel_values).image_embeds
#                  → [B, 1024]  (CLIP projection dimension)

# Step 5: Add sequence dimension for cross-attention
encoder_hidden_states = image_embeddings.unsqueeze(1)  # [B, 1, 1024]
```

**Model**: `CLIPVisionModelWithProjection` from `stabilityai/stable-video-diffusion-img2vid-xt`
- Visual backbone: ViT
- Output: 1024-dimensional CLIP embedding per image (single token per frame)
- Frozen throughout training

### 6.2 Injection into UNet

Before the UNet forward pass, the batch dimension is repeated for all frames:

```python
encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
# [B, 1, 1024]  →  [B*T, 1, 1024]
```

This single CLIP embedding (from the first RGB frame) is used as the cross-attention key/value
for **every frame** in the clip. It provides global scene context: lighting, scene type,
object appearance. The UNet attends to this embedding in all cross-attention layers of the
down-blocks, mid-block, and up-blocks.

> **Semantic vs. RGB conditioning split**:
> - **Channel concat** (spatial conditioning): carries per-frame semantic layout information
> - **Cross-attention** (CLIP): carries scene-level appearance context from the first RGB frame

---

## 7. Added-Time-IDs (Noise Augmentation Embedding)

**File**: `src/ctrlv/utils/util.py`, function `get_add_time_ids`

SVD conditions the UNet on three scalar values that describe the generation context:

| Field              | Training value  | Type    | Meaning |
|--------------------|----------------|---------|---------|
| `fps`              | `args.fps - 1` | int     | Frames per second minus 1 (e.g. 7 → 6). Controls temporal speed. |
| `motion_bucket_id` | `127`          | int     | Motion intensity bucket (0=still, 255=very dynamic). |
| `noise_aug_strength` | `args.noise_aug_strength` | float | Noise added to the conditioning image. Higher = more motion freedom. |

**Encoding**:
```python
add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
# → tensor [3]

# Each value projected via sinusoidal embedding of dim addition_time_embed_dim=256
time_embeds = add_time_proj(add_time_ids.flatten())  # [3 × 256] = [768]
time_embeds = time_embeds.reshape(batch_size, -1)    # [B, 768]

aug_emb = add_embedding(time_embeds)  # MLP: 768 → 1280
```

**Integration into UNet time embedding**:
```python
t_emb = time_proj(timesteps)      # sinusoidal: [B, 320]
emb = time_embedding(t_emb)       # MLP: 320 → 1280
emb = emb + aug_emb               # [B, 1280]  ← combined time + motion context
```

This combined embedding is then broadcast to all frames:
```python
emb = emb.repeat_interleave(num_frames, dim=0)  # [B*T, 1280]
```

and added to every ResNet block's hidden state via a linear projection.

---

## 8. UNet Forward Pass

**File**: `src/ctrlv/models/unet_spatio_temporal_condition.py`
**Class**: `UNetSpatioTemporalConditionModel`
**Config**: `in_channels=8`, `out_channels=4`, `num_frames=25`

### 8.1 Input Reshaping

```python
sample = sample.flatten(0, 1)   # [B, T, 8, H/8, W/8] → [B*T, 8, 24, 88]
emb    = emb.repeat_interleave(T, dim=0)              # [B, 1280] → [B*T, 1280]
encoder_hidden_states = encoder_hidden_states.repeat_interleave(T, dim=0)
#                       [B, 1, 1024] → [B*T, 1, 1024]
```

### 8.2 Temporal Indicator

```python
image_only_indicator = torch.zeros(B, T)  # [B, T], all zeros
```

All-zeros means every frame is treated as a video frame (not a static image), enabling
full temporal attention in the spatio-temporal blocks.

### 8.3 Conv-In (Channel Expansion)

```python
sample = conv_in(sample)  # Conv2d(8 → 320, kernel=3, pad=1)
# [B*T, 8, 24, 88] → [B*T, 320, 24, 88]
```

The 8-channel input (4 noisy + 4 conditioning) is projected to the 320-channel UNet feature space.

### 8.4 Down-Blocks

Four blocks with progressively halved spatial resolution:

| Block | Type                            | Input ch | Output ch | Spatial out |
|-------|---------------------------------|----------|-----------|-------------|
| 0     | CrossAttnDownBlockSpatioTemporal | 320     | 320       | 24×88       |
| 1     | CrossAttnDownBlockSpatioTemporal | 320     | 640       | 12×44       |
| 2     | CrossAttnDownBlockSpatioTemporal | 640     | 1280      | 6×22        |
| 3     | DownBlockSpatioTemporal (no attn)| 1280    | 1280      | 3×11        |

Each `CrossAttnDownBlockSpatioTemporal` contains:
1. **Spatial ResNet** — `(hidden_states, emb)` → residual blocks with time conditioning
2. **Temporal ResNet** — processes along the T dimension with `image_only_indicator`
3. **Cross-Attention** — `query` from spatial features, `key/value` from `encoder_hidden_states`
   (CLIP embedding, shape `[B*T, 1, 1024]`)
4. **Temporal Attention** — self-attention across the T dimension (within each spatial position)
5. **Downsampler** — Conv2d stride-2 (except block 3)

Residual samples from each block are stored in `down_block_res_samples` for skip connections.

### 8.5 Mid-Block

```
CrossAttnMidBlockSpatioTemporal
Input:  [B*T, 1280, 3, 11]
Output: [B*T, 1280, 3, 11]
```

Same structure as a down-block (ResNet + cross-attention + temporal attention) but no
spatial downsampling.

### 8.6 Up-Blocks

Four blocks with progressively doubled spatial resolution:

| Block | Type                          | Input ch (with skip) | Output ch | Spatial out |
|-------|-------------------------------|----------------------|-----------|-------------|
| 0     | UpBlockSpatioTemporal (no attn) | 1280+1280          | 1280      | 6×22        |
| 1     | CrossAttnUpBlockSpatioTemporal | 1280+1280           | 1280      | 12×44       |
| 2     | CrossAttnUpBlockSpatioTemporal | 1280+640            | 640       | 24×88       |
| 3     | CrossAttnUpBlockSpatioTemporal | 640+320             | 320       | 24×88       |

Each up-block concatenates the skip connection from the corresponding down-block before
processing. This follows the standard U-Net skip connection pattern.

### 8.7 Output Projection

```python
sample = conv_norm_out(sample)  # GroupNorm(32, 320)
sample = conv_act(sample)       # SiLU
sample = conv_out(sample)       # Conv2d(320 → 4, kernel=3, pad=1)
# [B*T, 320, 24, 88] → [B*T, 4, 24, 88]
```

### 8.8 Reshape to Temporal Form

```python
sample = sample.reshape(B, T, 4, H/8, W/8)
# [B*T, 4, 24, 88] → [B, T, 4, 24, 88]
```

**Output** (`model_pred`): `[B, T, 4, H/8, W/8]` = `[1, 25, 4, 24, 88]`

---

## 9. Trainable Parameters

Only the **temporal transformer blocks** are trained; all spatial weights are frozen from SVD-XT:

```python
unet.enable_grad(temporal_transformer_block=True)
# Trains only: layers with 'temporal_transformer_block' in their name
# All spatial ResNets, spatial attention, Conv-in/out → frozen
```

This parameter-efficient approach:
- Leverages SVD's pretrained spatial priors (scene understanding, edges, textures)
- Learns to predict future temporal dynamics via the temporal attention layers
- Reduces GPU memory and training time

---

## 10. Loss Computation

### 10.1 Denoising Target

SVD uses the **v-prediction** denoising parameterisation (EDM formulation):

```python
c_out  = -sigmas / sqrt(sigmas² + 1)          # [B, 1, 1, 1, 1]
c_skip =  1      / (sigmas² + 1)              # [B, 1, 1, 1, 1]
denoised_latents = model_pred * c_out + c_skip * noisy_latents
# [B, T, 4, 24, 88]
```

This maps the UNet's raw output to a denoised estimate of the clean latent.

### 10.2 Sigma-Weighted MSE Loss

```python
weighting = (1 + sigmas²) * sigmas^{-2}       # [B, 1, 1, 1, 1]
loss = mean(
    weighting × (denoised_latents - target_latents)²
)   # scalar
```

The weighting term up-weights high-noise timesteps (large σ), following the EDM loss
formulation. This makes training more stable by balancing the loss across the noise schedule.

---

## 11. Inference Pipeline

**File**: `src/ctrlv/pipelines/pipeline_video_diffusion.py`
**Class**: `VideoDiffusionPipeline`

### 11.1 Inputs at Inference

```python
pipeline(
    image=initial_rgb_frame,            # PIL.Image or [1, 3, H, W] — first RGB frame
    semantic_ids=semantic_ids_cond,     # [1, T, H, W] — semantic IDs (only frame 0 used)
    use_semantic_vae=True,
    num_cond_bbox_frames=1,
    num_frames=25,
    num_inference_steps=30,
    min_guidance_scale=1.0,
    max_guidance_scale=3.0,
    noise_aug_strength=0.02,
    motion_bucket_id=127,
    fps=8,
    output_type='latent',               # returns latents, not RGB
)
```

### 11.2 CLIP Image Embedding

```python
image_embeddings = _encode_image(image, ...)  # CLIPVisionModelWithProjection
# → [1, 1, 1024]   (with CFG: [2, 1, 1024] = uncond||cond)
```

The initial RGB frame is also VAE-encoded with a small noise perturbation for the image
latents (standard SVD procedure), but in semantic mode this path is bypassed for the
spatial conditioning (see 11.3).

### 11.3 Spatial Conditioning Latents at Inference

```python
# Encode only frame 0's semantic IDs
first_frame_sem_ids = semantic_ids[:, 0, :, :]    # [1, H, W]
image_latents = vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
# [1, 4, 24, 88]

# Duplicate for CFG
if do_cfg:
    negative_image_latents = torch.zeros_like(image_latents)
    image_latents = cat([negative_image_latents, image_latents])
    # [2, 4, 24, 88]

# Broadcast across all T frames
image_latents = image_latents.unsqueeze(1).repeat(1, T, 1, 1, 1)
# [2, T, 4, 24, 88]

# Apply boundary conditions (frame 0 and frame 24 from ground truth)
cond_latents = _encode_vae_condition(semantic_ids, ...)   # [1, T, 4, 24, 88]
image_latents[:, 0:1]  = cond_latents[:, 0:1]            # frame 0: exact GT
image_latents[:, -1:]  = cond_latents[:, -1:]             # frame 24: exact GT
```

### 11.4 Denoising Loop

```python
latents = randn(1, T, 4, 24, 88)   # random Gaussian noise as starting point
guidance_scale = linspace(min_gs=1.0, max_gs=3.0, T)   # per-frame CFG scale

for t in timesteps:  # 30 steps (DDIM/Euler)
    # Concatenate noisy latents + conditioning (channel dim)
    latent_model_input = cat([latents, latents], dim=0) if CFG  # [2, T, 4, ...]
    latent_model_input = cat([latent_model_input, image_latents], dim=2)
    # [2, T, 8, 24, 88]

    # UNet prediction
    noise_pred = unet(latent_model_input, t,
                      encoder_hidden_states=image_embeddings,
                      added_time_ids=added_time_ids)[0]
    # [2, T, 4, 24, 88]

    # Classifier-free guidance (per-frame scale)
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    # [1, T, 4, 24, 88]

    # Scheduler step (Euler)
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Output: [1, T, 4, 24, 88] semantic latents
```

### 11.5 Semantic VAE Decoding

```python
# Flatten temporal
latents_flat = latents.reshape(B*T, 4, H/8, W/8)    # [25, 4, 24, 88]

# Unscale from diffusion space
latents_unscaled = latents_flat / 0.18215             # [25, 4, 24, 88]

# VAE decoder: latents → semantic features
decoded_features = semantic_vae._decode_to_semantic_features(latents_unscaled)
# [25, 128, 192, 704]   (8× upsampling)

# Temporal dim for semantic head
decoded_features = decoded_features.unsqueeze(1)      # [25, 1, 128, 192, 704]

# Semantic head: features → class logits
logits = semantic_vae.semantic_head(decoded_features) # [25, 1, 19, 192, 704]
logits = logits[:, 0]                                  # [25, 19, 192, 704]

# Argmax to get predicted class
semantic_ids_pred = argmax(logits, dim=1)              # [25, 192, 704]

# Reshape to clip form
semantic_ids_pred = semantic_ids_pred.reshape(B, T, H, W)  # [1, 25, 192, 704]
```

**Final output**: `[1, 25, 192, 704]` int64 tensor with trainIds 0–18.

---

## 12. Complete Tensor Shape Reference

All shapes for the standard training configuration:
`B=1, T=25, H=192, W=704, H_lat=24 (H/8), W_lat=88 (W/8)`

| Stage | Tensor | Shape | Dtype | Notes |
|-------|--------|-------|-------|-------|
| Dataset | `semantic_ids` | `[1, 25, 192, 704]` | int64 | trainIds 0–18 |
| Dataset | `clips` | `[1, 25, 3, 192, 704]` | float32 | RGB, range [-1,1] |
| VAE encode (full clip) | `latents` (unscaled) | `[1, 25, 4, 24, 88]` | float16 | from Semantic VAE |
| VAE encode (frame 0) | `initial_frame_latent` | `[1, 4, 24, 88]` | float16 | first-frame only |
| Conditioning | `conditional_latents` | `[1, 25, 4, 24, 88]` | float16 | frame 0=GT, 1-23=F0, 24=GT |
| Diffusion prep | `target_latents` | `[1, 25, 4, 24, 88]` | float16 | scaled × 0.18215 |
| Diffusion prep | `noise` | `[1, 25, 4, 24, 88]` | float16 | N(0,1) |
| Diffusion prep | `timesteps` | `[1]` | int64 | sampled from [0,999] |
| Diffusion prep | `sigmas` | `[1, 1, 1, 1, 1]` | float16 | σ(t) |
| Diffusion prep | `noisy_latents` | `[1, 25, 4, 24, 88]` | float16 | after add_noise |
| Diffusion prep | `inp_noisy_latents` | `[1, 25, 4, 24, 88]` | float16 | ÷ √(σ²+1) |
| CLIP conditioning | `encoder_hidden_states` | `[1, 1, 1024]` | float16 | per batch |
| Added time IDs | `added_time_ids` | `[1, 3]` | float16 | [fps, mbid, noise_aug] |
| UNet input | `concatenated_noisy_latents` | `[1, 25, 8, 24, 88]` | float16 | 4+4 channels |
| UNet (after flatten) | `sample` | `[25, 8, 24, 88]` | float16 | temporal flattened |
| UNet (after conv_in) | `sample` | `[25, 320, 24, 88]` | float16 | after 8→320 projection |
| UNet (after block 0) | `sample` | `[25, 320, 24, 88]` | float16 | cross-attn applied |
| UNet (after block 1) | `sample` | `[25, 640, 12, 44]` | float16 | 2× downsampled |
| UNet (after block 2) | `sample` | `[25, 1280, 6, 22]` | float16 | 4× downsampled |
| UNet (after block 3) | `sample` | `[25, 1280, 3, 11]` | float16 | 8× downsampled |
| UNet (mid) | `sample` | `[25, 1280, 3, 11]` | float16 | same spatial res |
| UNet (after up 0) | `sample` | `[25, 1280, 6, 22]` | float16 | 2× upsampled |
| UNet (after up 1) | `sample` | `[25, 1280, 12, 44]` | float16 | 4× upsampled |
| UNet (after up 2) | `sample` | `[25, 640, 24, 88]` | float16 | 8× upsampled |
| UNet (after up 3) | `sample` | `[25, 320, 24, 88]` | float16 | |
| UNet (after conv_out) | `sample` | `[25, 4, 24, 88]` | float16 | 320→4 projection |
| UNet output | `model_pred` | `[1, 25, 4, 24, 88]` | float16 | after reshape |
| Loss | `denoised_latents` | `[1, 25, 4, 24, 88]` | float32 | c_skip×noisy + c_out×pred |
| Loss | scalar | — | float32 | sigma-weighted MSE |
| Decode (unscale) | `latents_unscaled` | `[25, 4, 24, 88]` | float16 | ÷ 0.18215 |
| Decode (features) | `decoded_features` | `[25, 128, 192, 704]` | float16 | after 8× upsample |
| Decode (logits) | `logits` | `[25, 19, 192, 704]` | float32 | 19-class softmax |
| Decode (argmax) | `semantic_ids_pred` | `[1, 25, 192, 704]` | int64 | final prediction |

---

## 13. Conditioning Mechanisms Summary

The Stage 1 UNet receives conditioning from four distinct sources, applied at different
levels of the architecture:

### 13.1 Spatial Conditioning via Channel Concatenation
- **What**: First-frame semantic latent, broadcast to all frame positions (or zeros for
  unobserved frames in the boundary condition pattern)
- **How**: Concatenated on channel dim before `conv_in` → doubles input channels (4→8)
- **Where**: Entire UNet (enters at the very first layer)
- **Effect**: Tells the model the initial scene layout; anchors prediction to a known state

### 13.2 Cross-Attention (CLIP Image Embedding)
- **What**: 1024-dim CLIP embedding of the first RGB frame
- **How**: Single token `[B*T, 1, 1024]` used as K/V in cross-attention of all cross-attn blocks
- **Where**: Down-blocks 0, 1, 2; Mid-block; Up-blocks 1, 2, 3
- **Effect**: Injects scene-level appearance context (object classes, lighting, road texture)

### 13.3 Time / Diffusion Step Embedding
- **What**: Sinusoidal embedding of the current diffusion timestep `t ∈ [0, 999]`
- **How**: Projected via MLP to `[B*T, 1280]`, added to every ResNet block's hidden state
- **Where**: Every ResNet block in all down/mid/up-blocks
- **Effect**: Conditions all residual transformations on the current noise level

### 13.4 Added-Time-IDs (Motion Context Embedding)
- **What**: Three scalars: `fps-1`, `motion_bucket_id=127`, `noise_aug_strength`
- **How**: Sinusoidal embedding → MLP → `aug_emb [B, 1280]`, added to the timestep embedding
- **Where**: Propagates to every ResNet block via the combined `emb = t_emb + aug_emb`
- **Effect**: Controls motion dynamics; `motion_bucket_id=127` sets a moderate motion level;
  `noise_aug_strength` encodes how much the conditioning image was perturbed

---

## 14. Key Hyperparameters

| Parameter | Value | Where set |
|-----------|-------|-----------|
| `clip_length` / `num_frames` | 25 | `--clip_length 25` |
| Training resolution | 192×704 | `--train_H 192 --train_W 704` |
| Latent resolution | 24×88 | H/8 × W/8 |
| Latent channels | 4 | SemanticVAE output channels |
| UNet input channels | 8 | 4 noisy + 4 conditioning |
| Scaling factor | 0.18215 | `vae.config.scaling_factor` |
| `num_cond_bbox_frames` | 1 | `--num_cond_bbox_frames 1` |
| Semantic classes | 19 | KITTI-360 trainIds |
| `fps` (added_time_ids) | `args.fps - 1` | e.g. 8-1=7 |
| `motion_bucket_id` | 127 | hardcoded |
| Learning rate | 5e-6 | `--learning_rate 5e-6` |
| Batch size | 1 (effective: 6 with grad accum) | `--train_batch_size 1 --gradient_accumulation_steps 6` |
| Mixed precision | fp16 | `--mixed_precision fp16` |
| Trainable params | Temporal transformer blocks | `unet.enable_grad(temporal_transformer_block=True)` |

---

## 15. File Map

| File | Role in Stage 1 |
|------|----------------|
| `tools/train_video_diffusion.py` | Main training loop: data loading, VAE encoding, conditioning construction, diffusion, loss |
| `src/ctrlv/pipelines/pipeline_video_diffusion.py` | Inference pipeline: denoising loop, CFG, latent decoding |
| `src/ctrlv/models/dual_vae_manager.py` | Semantic VAE encoder/decoder; RGB VAE wrapper |
| `src/ctrlv/models/unet_spatio_temporal_condition.py` | Modified UNet: 8-channel input, temporal attention enabled |
| `src/ctrlv/utils/util.py` | `encode_video_image` (CLIP), `get_add_time_ids` |
| `src/ctrlv/utils/semantic_preprocessing.py` | `load_and_remap_semantic`, `semantic_ids_to_onehot` |
| `src/ctrlv/datasets/kitti360_official.py` | Dataset: loads RGB + semantic PNGs, assembles clips |
| `src/ctrlv/datasets/kitti_abstract.py` | Base dataset class: clip sampling logic |
| `src/ctrlv/datasets/__init__.py` | `kitti_clip_with_bbox_collate_fn`: batch assembly |
| `scripts/train_scripts/train_kitti360_bbox_predict.sh` | SLURM launch script for Stage 1 |
| `scripts/eval_scripts/eval_stage1_semantic.py` | Evaluation: mIoU, pixel accuracy |
