# Stage 2 Pipeline: Sem2Video ControlNet — Detailed Documentation

**Task**: Semantic segmentation maps → Photorealistic RGB video
**Architecture**: SVD-XT UNet + ControlNet (semantic conditioning) + Semantic VAE (DualVAEManager)
**Input**: Semantic map sequence (T=25 frames, KITTI-360 trainIds 0–18) + initial RGB frame
**Output**: Photorealistic RGB video (25 frames, 192×704)

---

## Overview of Conditioning Streams

The Stage 2 pipeline has **four parallel conditioning streams** that all converge inside the UNet:

| Stream | Input | Encoding | Injection Point |
|--------|-------|----------|-----------------|
| Semantic ControlNet | Semantic maps (T frames) | Semantic VAE → 4-ch latents | ControlNet residuals → UNet encoder skip connections + mid block |
| RGB frame concat | Initial RGB frame | RGB VAE → 4-ch latents, repeated T times | Channel-concatenated with noisy latents (8-ch input) |
| CLIP visual embed | Initial RGB frame | CLIP ViT → 1024-d | Cross-attention in each down block |
| Time embeddings | Timestep + FPS + motion_bucket + noise_aug | Sinusoidal + linear projection | Added to every ResNet block as shift/scale (AdaGN) |

---

## 1. Data Ingestion: KITTI-360 Dataset

**Files**: `src/ctrlv/datasets/kitti360_official.py`, `src/ctrlv/utils/semantic_preprocessing.py`

### What is loaded
- **RGB frames**: `data_2d_raw/{sequence}/image_00/data_rect/{frame}.png` — [T, 3, H, W] uint8
- **Semantic maps**: `data_2d_semantics/train/{sequence}/image_00/semantic/{frame}.png` — [T, H, W] uint8 grayscale

### Semantic remapping pipeline (inside dataset `__getitem__`)
1. Load grayscale PNG → numpy [H, W] with raw KITTI-360 label IDs (e.g., 7=road, 11=building, 26=car)
2. Apply `KITTI360_LABEL_MAPPING` dict → remap raw IDs to continuous trainIds 0–18 (19 classes)
3. Unmapped pixels → 255 (ignore_index, treated as void)
4. Resize with nearest-neighbor interpolation to training resolution (192×704)
5. Convert to `torch.long` tensor [H, W]
6. Stack across clip to get `semantic_ids` [T, H, W]

### Batch collation
`kitti_clip_with_bbox_collate_fn` stacks per-sample tensors:
- `clips`: [B, T, 3, H, W] float32 RGB, normalized to [-1, 1]
- `semantic_ids`: [B, T, H, W] int64, values in {0–18, 255}

---

## 2. Semantic Encoding via DualVAEManager

**Files**: `src/ctrlv/models/dual_vae_manager.py`, `src/ctrlv/utils/semantic_preprocessing.py`

This step converts raw integer semantic IDs into 4-channel latent vectors compatible with the SVD latent space.

### DualVAEManager manages two VAEs
- **RGB VAE**: Frozen `AutoencoderKLTemporalDecoder` from SVD (encodes/decodes RGB frames)
- **Semantic VAE**: Pretrained `SemanticVAENative` loaded from `/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth`

### `encode_semantic_from_ids(semantic_ids)` — step by step

**Input**: `semantic_ids` [B×T, H, W] — integer trainIds (after flattening batch and temporal dims)

**Step 1 — Temporal padding**:
- Semantic VAE expects clips of size 4 (clip_size=4)
- If B×T is not divisible by 4, pad by repeating the last frame
- Pad shape: [B×T_padded, H, W]

**Step 2 — One-hot encoding**:
```
semantic_clamped = clamp(semantic_ids, 0, 18)   # Treat 255 (ignore) → 0 (safe clamp)
x_onehot = F.one_hot(semantic_clamped, num_classes=19)
# [B*T, H, W, 19] → permute → [B*T, 19, H, W] float32
```

**Step 3 — Semantic stem (input projection)**:
```
h0 = semantic_vae.model.semantic_stem(x_onehot)
# Conv2d: 19ch → 128ch, [B*T, 128, H, W]
```

**Step 4 — VAE encoder core**:
```
latents_flat = semantic_vae.model._encode_semantic_features(h0)
# 3× downsampling + residual blocks → [B*T, 4, H/8, W/8]
```

**Step 5 — Remove padding**:
- Slice to original [B×T, 4, H/8, W/8], drop padded frames

**Step 6 — Scale to diffusion latent space**:
```
semantic_latents = latents_flat * vae.config.scaling_factor   # × 0.18215
```

**Step 7 — Reshape back to temporal**:
```
bbox_em = rearrange(semantic_latents, '(b f) c h w -> b f c h w', f=T)
# Shape: [B, T, 4, H/8, W/8]
```

The semantic latents are now in the **same latent space** as the RGB VAE outputs (both 4-channel, same spatial resolution H/8 × W/8 = 24 × 88 for 192×704 input).

---

## 3. RGB Initial Frame Encoding (Two Paths)

**File**: `tools/train_video_controlnet.py` lines 513–560

The initial RGB frame (first frame of the clip) is encoded **two separate ways** for two different conditioning mechanisms.

### Path A: CLIP visual embedding (for cross-attention)
```
initial_image [B, 3, H, W]
    → feature_extractor (CLIP preprocessor) → normalized patches
    → image_encoder (CLIP ViT-H/14) → penultimate layer features
    → [B, 1, 1024]  (pooled spatial tokens)
```
This embedding conditions the UNet via **cross-attention** in each spatio-temporal down block. It encodes "what scene is in the initial frame" as a global context.

### Path B: RGB VAE latent (for channel concatenation)
```
initial_image [B, 3, H, W]
    → rgb_vae.encode() → latent_dist.sample() → [B, 4, H/8, W/8]
    → × scaling_factor (0.18215)
    → repeat along temporal dim → [B, T, 4, H/8, W/8]
```
This is the **frame conditioning latent**: it tells the UNet "generate a video that starts from this frame." Every frame in the video shares the same initial-frame latent as an additional channel input.

> **Note on noise augmentation** (inference only): During inference, Gaussian noise is added to the initial frame before VAE encoding (`image + noise_aug_strength × ε`). This prevents over-conditioning on a single frame and is controlled by `noise_aug_strength` (typically 0.02). The same `noise_aug_strength` value is passed as an embedding (see Section 5).

---

## 4. Noisy Latents Preparation (Forward Diffusion)

**File**: `tools/train_video_controlnet.py` lines 590–616

### Encode RGB video targets
```
rgb_clips [B, T, 3, H, W]
    → flatten → [B*T, 3, H, W]
    → rgb_vae.encode().latent_dist.sample()
    → [B*T, 4, H/8, W/8] × scaling_factor
    → rearrange → [B, T, 4, H/8, W/8]   ← target_latents
```

### Add diffusion noise
```
noise = randn_like(target_latents)                         # [B, T, 4, H/8, W/8]
timesteps = sample_from_logit_normal(T_max=1000)           # Random per sample
noisy_latents = noise_scheduler.add_noise(target, noise, t) # [B, T, 4, H/8, W/8]

# Karras et al. input scaling:
sigmas = get_sigmas(timesteps)                             # Noise level magnitudes
inp_noisy_latents = noisy_latents / sqrt(sigmas² + 1)      # Scale for UNet input
```

### Concatenate with RGB frame conditioning (8-channel input)
```
inp_noisy_latents    [B, T, 4, H/8, W/8]   ← partially-denoised latents
conditional_latents  [B, T, 4, H/8, W/8]   ← initial RGB frame repeated
    → torch.cat(dim=2)
    → concatenated_noisy_latents [B, T, 8, H/8, W/8]
```

This 8-channel tensor is the **primary input** to both the ControlNet and the UNet. The channel split means the UNet always has access to both "what noise to remove" and "what the first frame looks like."

---

## 5. Time and Augmentation Embeddings

**File**: `src/ctrlv/models/controlnet.py`, `tools/train_video_controlnet.py`

### Diffusion timestep embedding
```
timestep t (scalar per sample)
    → Timesteps (sinusoidal): [B] → [B, 320]
    → TimestepEmbedding (MLP): [B, 320] → [B, 1280]   ← t_emb
```

### Augmentation time embedding (`added_time_ids`)
Three extra conditioning scalars are packed into a [B, 3] tensor:

| Index | Field | Value | Meaning |
|-------|-------|-------|---------|
| 0 | `fps_id` | 7 (training) | Target frame rate |
| 1 | `motion_bucket_id` | 127 | Amount of motion expected |
| 2 | `noise_aug_strength` | 0.0 (train) / 0.02 (infer) | How much noise was on the initial frame |

```
added_time_ids [B, 3]
    → flatten → [B*3,]
    → Timesteps (sinusoidal, freq_shift=0): [B*3, 256]
    → reshape: [B, 768]
    → AddedTimeEmbedding (MLP): [B, 768] → [B, 1280]   ← aug_emb
```

### Combined time conditioning
```
emb = t_emb + aug_emb   # [B, 1280]
emb = emb.repeat_interleave(num_frames, dim=0)   # [B*T, 1280]
```

This is **added (AdaGN)** to every ResNet block inside both the ControlNet and the UNet, providing noise-level and motion-level guidance at every layer.

---

## 6. ControlNet Forward Pass

**File**: `src/ctrlv/models/controlnet.py`

The ControlNet is a **frozen copy of the UNet encoder + mid block**, with extra pathways for semantic conditioning. It processes the same 8-channel input as the UNet but produces **residual corrections** that are added to the UNet's encoder outputs.

### Input layout
```
sample           [B, T, 8, H/8, W/8]   ← noisy latents + RGB frame concat
control_cond     [B, T, 4, H/8, W/8]   ← semantic latents from DualVAEManager
encoder_hidden   [B, T, 1024]          ← CLIP visual embeddings
emb              [B, 1280]             ← timestep + aug embeddings
```

### Step 1 — Flatten temporal dimension
```
sample        → [B*T, 8, H/8, W/8]
control_cond  → [B*T, 4, H/8, W/8]   # Save as control_cond_original (not downsampled further)
emb           → repeat_interleave(T) → [B*T, 1280]
encoder_hidden → repeat_interleave(T) → [B*T, 1, 1024]
```

### Step 2 — Input projection and initial semantic injection
```
# UNet-side conv_in (same as UNet's conv_in, initialized from it)
sample = conv_in(sample)          # [B*T, 8ch] → [B*T, 320ch]

# Semantic side: control_conv_in is a separate Conv2d(4 → 320)
control_cond = control_conv_in(control_cond)   # [B*T, 4ch] → [B*T, 320ch]

# INITIAL ADDITIVE INJECTION — semantics folded directly into feature map
sample = sample + control_cond    # [B*T, 320, H/8, W/8]
```

At this point, semantic information is already "baked into" the feature maps before any block processes them.

### Step 3 — Down blocks with multi-scale semantic re-injection

The ControlNet has **4 down blocks** (mirrors the UNet encoder). After blocks 0, 1, and 2 (not 3), the **original semantic latents** are re-injected at the current spatial resolution via zero-initialized projectors.

#### Down Block 0 (spatial: H/8 × W/8 = 24×88, channels: 320)
```
# Block forward (CrossAttnDownBlockSpatioTemporal)
sample, res_samples_0 = down_block_0(
    hidden_states=sample,          # [B*T, 320, 24, 88]
    temb=emb,                      # [B*T, 1280]
    encoder_hidden_states=clip,    # [B*T, 1, 1024]
    image_only_indicator=zeros,    # [B, T] zeros (enables temporal attention)
)
# res_samples_0: tuple of residual tensors saved for skip connections

# MULTI-SCALE RE-INJECTION 0
semantic_rescaled = F.interpolate(control_cond_original, size=(24, 88), mode='bilinear')
# [B*T, 4, 24, 88]
sample = sample + semantic_scale_projectors[0](semantic_rescaled)
# semantic_scale_projectors[0] = zero_conv(4 → 320), zero-initialized
```

#### Down Block 1 (spatial: H/16 × W/16 = 12×44, channels: 640)
```
sample, res_samples_1 = down_block_1(sample, emb, clip, ...)

# MULTI-SCALE RE-INJECTION 1
semantic_rescaled = F.interpolate(control_cond_original, size=(12, 44), mode='bilinear')
# [B*T, 4, 12, 44]
sample = sample + semantic_scale_projectors[1](semantic_rescaled)
# semantic_scale_projectors[1] = zero_conv(4 → 640), zero-initialized
```

#### Down Block 2 (spatial: H/32 × W/32 = 6×22, channels: 1280)
```
sample, res_samples_2 = down_block_2(sample, emb, clip, ...)

# MULTI-SCALE RE-INJECTION 2
semantic_rescaled = F.interpolate(control_cond_original, size=(6, 22), mode='bilinear')
# [B*T, 4, 6, 22]
sample = sample + semantic_scale_projectors[2](semantic_rescaled)
# semantic_scale_projectors[2] = zero_conv(4 → 1280), zero-initialized
```

#### Down Block 3 (spatial: H/32 × W/32 = 6×22, channels: 1280 — no downsampling)
```
sample, res_samples_3 = down_block_3(sample, emb, clip, ...)
# No re-injection at this final down block
```

#### Why zero-initialization?
The projectors are initialized with zero weights. This means at the start of training, their output is zero — the ControlNet behaves identically to the pretrained SVD checkpoint. As training progresses, the projectors gradually learn to inject meaningful semantic corrections. This ensures a smooth warm-up from the pretrained initialization.

#### Why interpolate from the original rather than progressively downsample?
Using `F.interpolate(control_cond_original, ...)` at each scale preserves the full semantic information at every resolution. Progressive downsampling would accumulate spatial information loss. Bilinear interpolation provides smooth semantic boundaries at coarser scales.

### Step 4 — Mid block
```
sample = mid_block(
    hidden_states=sample,          # [B*T, 1280, 6, 22]
    temb=emb,                      # [B*T, 1280]
    encoder_hidden_states=clip,    # [B*T, 1, 1024]
    image_only_indicator=zeros,
)
# Output: [B*T, 1280, 6, 22]
```

### Step 5 — Zero-conv residual projections

All accumulated `res_samples` (from down blocks) plus the mid block output are passed through **zero-initialized 1×1 convolutions** (the ControlNet's standard residual projection mechanism):

```
controlnet_down_block_res_samples = ()
for res_sample, controlnet_block in zip(all_res_samples, self.controlnet_down_blocks):
    res_sample = controlnet_block(res_sample)   # zero-conv 1×1
    controlnet_down_block_res_samples += (res_sample,)

mid_block_res = self.controlnet_mid_block(sample)   # zero-conv 1×1

# Scale all by conditioning_scale (1.0 by default)
down_block_residuals = [r * conditioning_scale for r in controlnet_down_block_res_samples]
mid_block_residual   = mid_block_res * conditioning_scale
```

### ControlNet output
```
down_block_residuals: list of tensors (12 total, 3 per down block)
    - Block 0 residuals: 3 × [B*T, 320, 24, 88]
    - Block 1 residuals: 3 × [B*T, 640, 12, 44]  (or mixed sizes)
    - Block 2 residuals: 3 × [B*T, 1280, 6, 22]
    - Block 3 residuals: 3 × [B*T, 1280, 6, 22]
mid_block_residual:   [B*T, 1280, 6, 22]
```

---

## 7. UNet Forward Pass with ControlNet Residuals

**File**: `src/ctrlv/models/unet_spatio_temporal_condition.py`

The UNet takes the same 8-channel input and produces a noise prediction. ControlNet residuals are **added to the skip connections** that connect the encoder to the decoder.

### Input
```
sample           [B, T, 8, H/8, W/8]   ← same concatenated noisy+conditioning latents
encoder_hidden   [B, T, 1024]          ← CLIP visual embeddings
emb              [B, 1280]             ← timestep + aug embeddings
down_block_residuals (list)            ← from ControlNet
mid_block_residual                     ← from ControlNet
```

### Conv-in
```
sample = conv_in(sample)   # [B*T, 8ch] → [B*T, 320ch]
```
Note: The UNet's `conv_in` also accepts 8 channels (standard SVD configuration). Noisy latents and RGB conditioning latents are simply concatenated in the channel dimension.

### Down blocks (encoder)
```
down_block_res_samples = (sample,)   # Initial sample saved as first skip

for i, down_block in enumerate(self.down_blocks):
    sample, res_samples = down_block(
        hidden_states=sample,
        temb=emb,
        encoder_hidden_states=clip,
        image_only_indicator=zeros,
    )
    down_block_res_samples += res_samples

# ADD CONTROLNET RESIDUALS TO ALL DOWN-BLOCK OUTPUTS
new_down_block_res_samples = ()
for unet_res, ctrl_res in zip(down_block_res_samples, down_block_residuals):
    new_down_block_res_samples += (unet_res + ctrl_res,)
down_block_res_samples = new_down_block_res_samples
```

This is the **primary ControlNet injection point**: every skip connection from the encoder is corrected by the corresponding ControlNet residual before being passed to the decoder.

### Mid block
```
sample = mid_block(sample, emb, clip, ...)

# ADD CONTROLNET MID-BLOCK RESIDUAL
sample = sample + mid_block_residual
```

### Up blocks (decoder)
```
for up_block in self.up_blocks:
    # Pop the last N skip connections (N = len(up_block.resnets))
    res_samples = down_block_res_samples[-len(up_block.resnets):]
    down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

    sample = up_block(
        hidden_states=sample,
        res_hidden_states_tuple=res_samples,   # Now contain ControlNet corrections
        temb=emb,
        encoder_hidden_states=clip,
        image_only_indicator=zeros,
    )
```

The decoder never directly sees semantic information — it only receives semantically-corrected skip connections and the corrected bottleneck. This is the standard ControlNet design principle.

### Output projection
```
sample = conv_norm_out(sample)   # GroupNorm
sample = conv_act(sample)        # SiLU
sample = conv_out(sample)        # [B*T, 320ch] → [B*T, 4ch]
noise_pred = rearrange(sample, '(b f) c h w -> b f c h w', f=T)
# [B, T, 4, H/8, W/8]
```

---

## 8. Denoising Loop (Inference)

**File**: `src/ctrlv/pipelines/pipeline_video_control.py`

### Initialization
```
latents = randn([B, T, 4, H/8, W/8])   # Pure noise at t=T_max
scheduler = EulerDiscreteScheduler (or DDIM)
timesteps = scheduler.timesteps         # Decreasing from T_max to 0
```

### Per-step
```python
for t in timesteps:
    # Classifier-free guidance: duplicate for unconditional + conditional
    latent_input = cat([latents, latents])           # [B*2, T, 4, H/8, W/8]
    latent_input = scheduler.scale_model_input(latent_input, t)

    # Concatenate initial RGB frame conditioning
    latent_input = cat([latent_input, image_latents_cfg], dim=2)   # [B*2, T, 8, H/8, W/8]

    # ControlNet pass
    down_res, mid_res = controlnet(
        latent_input,                        # [B*2, T, 8]
        timestep=t,
        encoder_hidden_states=clip_cfg,      # [B*2, T, 1024]
        added_time_ids=time_ids_cfg,         # [B*2, 3]
        control_cond=semantic_latents_cfg,   # [B*2, T, 4] (zeros for uncond, semantics for cond)
        conditioning_scale=1.0,
    )

    # UNet pass
    noise_pred = unet(
        sample=latent_input,
        timestep=t,
        encoder_hidden_states=clip_cfg,
        added_time_ids=time_ids_cfg,
        down_block_additional_residuals=down_res,
        mid_block_additional_residuals=mid_res,
    )

    # Classifier-free guidance
    noise_uncond, noise_cond = noise_pred.chunk(2)
    noise_pred = noise_uncond + guidance_scale × (noise_cond - noise_uncond)

    # Scheduler step
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

**CFG zero unconditional**: For the unconditional branch, `control_cond` is set to all-zeros (zero semantic latents). This allows the guidance scale to control how strongly semantic maps steer the generation.

---

## 9. Post-Processing: Decode Latents to RGB Frames

**File**: `src/ctrlv/pipelines/pipeline_video_control.py` — `decode_latents()`

```
latents [B, T, 4, H/8, W/8]
    → ÷ vae.config.scaling_factor (0.18215)
    → flatten → [B*T, 4, H/8, W/8]
    → rgb_vae.decode(latents) → [B*T, 3, H, W]   # AutoencoderKLTemporalDecoder
    → clamp to [-1, 1] → rescale to [0, 1]
    → rearrange → [B, T, 3, H, W]
    → tensor2vid() → List[List[PIL.Image]] or numpy [B, T, H, W, 3]
```

The RGB VAE decoder is the **same frozen decoder from SVD** — it has no knowledge of semantics. All semantic conditioning was handled upstream.

---

## 10. Multi-Scale Semantic Re-Injection: Architectural Detail

**File**: `src/ctrlv/models/controlnet.py` lines 136–152, 326–356

This is the novel contribution on top of standard ControlNet. Here is the complete picture:

### Standard ControlNet (baseline)
- Semantic latents → `control_conv_in` → add to `conv_in` output once at the start
- All subsequent processing is semantic-unaware

### Our augmented ControlNet (implemented)
- Semantic latents → `control_conv_in` → add at input (1/8 resolution) **AND**
- Re-injected at 1/16 resolution (after block 0 downsampling) via `semantic_scale_projectors[0]`
- Re-injected at 1/32 resolution (after block 1 downsampling) via `semantic_scale_projectors[1]`
- Re-injected at 1/32 resolution (after block 2, no-downsample) via `semantic_scale_projectors[2]`

### Projector specifications

| Projector | Input ch | Output ch | Spatial scale | Spatial size (192×704) | Init |
|-----------|----------|-----------|---------------|------------------------|------|
| `control_conv_in` | 4 | 320 | 1/8 | 24×88 | Random |
| `semantic_scale_projectors[0]` | 4 | 320 | 1/8 → interp to 1/8 | 24×88 | Zero |
| `semantic_scale_projectors[1]` | 4 | 640 | 1/8 → interp to 1/16 | 12×44 | Zero |
| `semantic_scale_projectors[2]` | 4 | 1280 | 1/8 → interp to 1/32 | 6×22 | Zero |

The projectors are 3×3 Conv2d layers (not 1×1), allowing spatial context to influence the re-injection.

### Semantic information flow through ControlNet
```
semantic_latents [B*T, 4, 24, 88]
    ├─→ control_conv_in (4→320, 3×3) → [B*T, 320, 24, 88]
    │       ↓ add to conv_in(noisy+rgb_latents)
    │       ↓ [B*T, 320, 24, 88]  ← combined features enter blocks
    │
    ├─→ interp(24×88) → scale_proj[0](4→320, 3×3) → [B*T, 320, 24, 88]
    │       ↓ add after down_block_0
    │
    ├─→ interp(12×44) → scale_proj[1](4→640, 3×3) → [B*T, 640, 12, 44]
    │       ↓ add after down_block_1
    │
    └─→ interp(6×22) → scale_proj[2](4→1280, 3×3) → [B*T, 1280, 6, 22]
            ↓ add after down_block_2
```

---

## 11. Complete Data Flow Summary

```
STAGE 2 PIPELINE: Semantic → RGB Video
═══════════════════════════════════════════════════════════════════════

INPUT
├── semantic_ids [B, T, H, W]     (KITTI-360 trainIds 0-18)
├── initial_rgb  [B, 3, H, W]     (first RGB frame of target clip)
└── target_rgb   [B, T, 3, H, W]  (full RGB clip, training only)

SEMANTIC ENCODING PATH
  semantic_ids [B, T, H, W]
    → flatten → [B*T, H, W]
    → clamp(0,18) → one-hot → [B*T, 19, H, W]
    → semantic_stem (19→128) → [B*T, 128, H, W]
    → semantic_vae_encoder → [B*T, 4, H/8, W/8]
    → × 0.18215
    → reshape → bbox_em [B, T, 4, H/8, W/8]

RGB INITIAL FRAME PATHS
  initial_rgb [B, 3, H, W]
    ├─→ CLIP ViT-H/14 → image_embeddings [B, 1, 1024]   (for cross-attention)
    └─→ RGB VAE encode → [B, 4, H/8, W/8]
            → repeat(T) → image_latents [B, T, 4, H/8, W/8]  (for channel concat)

NOISY LATENTS (training)
  target_rgb [B, T, 3, H, W]
    → RGB VAE encode → target_latents [B, T, 4, H/8, W/8]
    → add_noise(t) → noisy_latents [B, T, 4, H/8, W/8]
    → Karras scale → inp_noisy_latents [B, T, 4, H/8, W/8]

8-CHANNEL INPUT CONSTRUCTION
  inp_noisy_latents [B, T, 4, H/8, W/8]
  + image_latents   [B, T, 4, H/8, W/8]
  ─────────────────────────────────────
  = concat_input    [B, T, 8, H/8, W/8]   ← primary UNet/ControlNet input

TIME EMBEDDINGS
  timestep t  → sinusoidal → MLP → t_emb  [B, 1280]
  (fps, motion_bucket, noise_aug) → sinusoidal → MLP → aug_emb [B, 1280]
  emb = t_emb + aug_emb                         [B, 1280]
  → repeat(T) →                                 [B*T, 1280]

CONTROLNET FORWARD
  concat_input [B*T, 8, H/8, W/8]   semantic_latents [B*T, 4, H/8, W/8]
       ↓                                     ↓
  conv_in (8→320)              control_conv_in (4→320)
       ↓          ADD ←──────────────────────┘
  [B*T, 320, H/8, W/8]  ← initial semantic injection (1/8 scale)
       ↓
  down_block_0 (320ch, 24×88)  +  temporal attention  +  cross-attn(CLIP)
       ↓         ADD ←── interp(semantic_orig to 24×88) → scale_proj[0](4→320)
  [B*T, 320, 24×88]  ← re-injection 0
       ↓
  down_block_1 (640ch, 12×44)  +  temporal attention  +  cross-attn(CLIP)
       ↓         ADD ←── interp(semantic_orig to 12×44) → scale_proj[1](4→640)
  [B*T, 640, 12×44]  ← re-injection 1
       ↓
  down_block_2 (1280ch, 6×22)  +  temporal attention  +  cross-attn(CLIP)
       ↓         ADD ←── interp(semantic_orig to 6×22) → scale_proj[2](4→1280)
  [B*T, 1280, 6×22]  ← re-injection 2
       ↓
  down_block_3 (1280ch, 6×22)  +  temporal attention  +  cross-attn(CLIP)
       ↓
  mid_block (1280ch, 6×22)     +  temporal attention  +  cross-attn(CLIP)
       ↓
  zero-conv projections on all skip connections + mid output
       ↓
  down_residuals (12 tensors) + mid_residual

UNET FORWARD
  concat_input [B*T, 8, H/8, W/8]
       ↓ conv_in (8→320)
  down_block_0 → skip_0
  down_block_1 → skip_1
  down_block_2 → skip_2
  down_block_3 → skip_3
       ↓ (each skip_i += ctrl_residual_i)  ← CONTROLNET INJECTION TO ENCODER
  mid_block
       ↓ (mid_out += ctrl_mid_residual)    ← CONTROLNET INJECTION TO BOTTLENECK
  up_block_0 (uses skip_3 + skip_2)
  up_block_1 (uses skip_1 + skip_0)
  up_block_2 (uses skip_0 extras)
  up_block_3 (uses initial sample)
       ↓ conv_out (320→4)
  noise_pred [B, T, 4, H/8, W/8]

DENOISING STEP
  noise_pred → CFG: noise_uncond + scale × (noise_cond - noise_uncond)
  latents_{t-1} = scheduler.step(noise_pred, t, latents_t)

FINAL DECODE (after all timesteps)
  latents_0 [B, T, 4, H/8, W/8]
    → ÷ 0.18215
    → RGB VAE decode
    → [B, T, 3, H, W] float32 in [0, 1]
    → output video frames
```

---

## 12. Key Files Reference

| File | Role |
|------|------|
| `src/ctrlv/pipelines/pipeline_video_control.py` | Full inference pipeline: encoding, denoising loop, CFG, decoding |
| `src/ctrlv/models/controlnet.py` | ControlNet: semantic injection, multi-scale re-injection, residual projection |
| `src/ctrlv/models/unet_spatio_temporal_condition.py` | UNet: adds ControlNet residuals to encoder skip connections and bottleneck |
| `src/ctrlv/models/dual_vae_manager.py` | Dual VAE management: one-hot → semantic VAE → 4-ch latents |
| `src/ctrlv/utils/semantic_preprocessing.py` | KITTI-360 label remapping, one-hot encoding utilities |
| `src/ctrlv/datasets/kitti360_official.py` | Dataset: loads RGB + semantic map pairs, produces `semantic_ids` tensors |
| `src/ctrlv/datasets/__init__.py` | Collate fn: stacks `semantic_ids` int64 tensors in batch dimension |
| `tools/train_video_controlnet.py` | Training loop: loss computation, CFG dropout, checkpoint saving |
| `scripts/eval_scripts/eval_stage2_rgb.sh` | Evaluation: DRN mIoU on generated RGB, FID/FVD |

---

## 13. Training-Specific Details

### Trainable parameters
- **ControlNet**: Fully trained (all weights)
- **UNet**: Frozen by default; optionally partially unfrozen — only `mid_block`, `conv_norm_out`, `conv_act`, `conv_out` (via `--unet_learning_rate`)
- **Semantic VAE**: Frozen
- **RGB VAE**: Frozen
- **CLIP image encoder**: Frozen

### Conditioning dropout (for CFG during inference)
```
random_p ~ Uniform[0, 1]   per sample

if random_p < 2 × dropout_prob:
    encoder_hidden_states ← null_conditioning (zeros)   # Drop CLIP embedding

if random_p in [dropout_prob, 3 × dropout_prob):
    conditional_latents ← 0                             # Drop initial RGB frame

# Semantic conditioning dropout is handled by CFG zero-uncond at inference
```

### Loss function (Karras et al. parameterization)
```
c_out  = -σ / √(σ² + 1)
c_skip = 1 / (σ² + 1)
denoised = noise_pred × c_out + c_skip × noisy_latents

weighting = (1 + σ²) × σ⁻²      # Up-weights high-noise timesteps
loss = mean(weighting × (denoised - target_latents)²)
```

---

## 14. Extra Clarification: How Semantic Reinjection Is Actually Added Inside ControlNet

**Code path**: `src/ctrlv/models/controlnet.py` lines 136–152 and 297–345

One subtle but important detail is that Stage 2 uses **two different kinds of "zero-conv" logic**, and they should not be confused:

1. **Semantic reinjection inside ControlNet feature extraction**
   - The original semantic latent `control_cond` is first flattened to `[B*T, 4, H/8, W/8]` and saved as `control_cond_original`.
   - It is injected once at the input through `control_conv_in`, which is a **normal learned 3×3 Conv2d(4→320)**, not zero-initialized:
   ```
   sample = self.conv_in(sample)                  # 8 → 320
   control_cond = self.control_conv_in(control_cond)  # 4 → 320
   sample = sample + control_cond
   ```
   - After that, the model keeps re-injecting **fresh semantic latents** after each non-final down block. This is done by resizing `control_cond_original` to the current feature-map size, projecting it with a **zero-initialized 3×3 conv**, and then adding it to the current ControlNet feature tensor:
   ```
   semantic_rescaled = F.interpolate(
       control_cond_original,
       size=sample.shape[-2:],
       mode="bilinear",
       align_corners=False,
   )
   sample = sample + self.semantic_scale_projectors[inject_idx](semantic_rescaled)
   ```
   - `semantic_scale_projectors` contains three zero-initialized layers:
     - `semantic_scale_projectors[0]`: `Conv2d(4→320, 3×3)`
     - `semantic_scale_projectors[1]`: `Conv2d(4→640, 3×3)`
     - `semantic_scale_projectors[2]`: `Conv2d(4→1280, 3×3)`
   - At training resolution 192×704, the actual reinjection scales are:
     - after down block 0: `24×88 → 12×44`, reinject semantics resized to `12×44`
     - after down block 1: `12×44 → 6×22`, reinject semantics resized to `6×22`
     - after down block 2: `6×22 → 3×11`, reinject semantics resized to `3×11`
   - So the architecture is **not** "extract features first, then only pass them through the standard ControlNet zero-conv outputs." Instead, semantic information is refreshed **inside the encoder itself** at multiple depths by direct additive fusion into the running ControlNet hidden state.

2. **Standard ControlNet zero-conv outputs to the UNet**
   - Separately, after all ControlNet down-block features are extracted, those feature tensors are passed through the usual zero-initialized **1×1** `controlnet_down_blocks` and `controlnet_mid_block`.
   - These 1×1 zero-convs do **not** perform semantic reinjection. Their job is to convert ControlNet features into residuals that are added to the matching UNet skip connections and mid block.

So the full semantic path is:
`semantic latent → control_conv_in initial add → multi-scale resized semantic add via zero-initialized 3×3 projectors inside ControlNet → extracted ControlNet features → zero-initialized 1×1 residual projections → UNet skip/mid residual injection`.
