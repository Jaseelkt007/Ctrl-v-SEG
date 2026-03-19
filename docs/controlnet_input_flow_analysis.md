# ControlNet Input Formulation, Forward Pass, and Output Extraction
## Complete Flow Analysis with Shapes

All shapes use concrete values from KITTI-360 semantic training:
- `B` = batch size (typically 1)
- `F` = 25 (clip_length)
- `H, W` = 192, 704 (training resolution)
- `lH, lW` = 24, 88 (latent resolution = H/8, W/8)

---

## 1. What Is Fed Into the ControlNet

The ControlNet call in both training and inference receives five inputs:

```python
# Training  (train_video_controlnet.py, lines 605–613)
ctrlnet(
    concatenated_noisy_latents,   # ← "sample"      [B, F, 8, lH, lW]
    timestep=timesteps,           # ← "timestep"    [B]
    encoder_hidden_states=...,    # ← CLIP embeds   [B, 1, 1024]
    added_time_ids=...,           # ← time metadata [B, 3]
    control_cond=bbox_em,         # ← semantic cond [B, F, 4, lH, lW]
    conditioning_scale=1.0,
)

# Inference  (pipeline_video_control.py, lines 338–346)
self.controlnet(
    latent_model_input,           # ← "sample"      [2B, F, 8, lH, lW]  (doubled for CFG)
    timestep=t,
    encoder_hidden_states=...,    # ← CLIP embeds   [2B, 1, 1024]
    added_time_ids=...,           # ← time metadata [2B, 3]
    control_cond=cond_em,         # ← semantic cond [2B, F, 4, lH, lW]  (doubled for CFG)
    conditioning_scale=1.0,
)
```

The two key inputs — the `sample` and `control_cond` — are constructed differently in training vs inference and deserve detailed explanation.

---

## 2. Input A: `sample` (the Noisy Latent Input)

This is called `concatenated_noisy_latents` in training and `latent_model_input` in the pipeline. It is an **8-channel tensor** formed by concatenating the noisy RGB latents with the first-frame conditioning latents along the channel dimension.

### 2.1 Training Construction (`train_video_controlnet.py`, lines 526–602)

**Step 1 — Encode RGB video clips through the VAE:**
```python
# batch['clips']: [B, 25, 3, 192, 704]  — full RGB clip
frames = rearrange(batch['clips'], 'b f c h w -> (b f) c h w')
# frames: [B*25, 3, 192, 704]

latents = vae.encode(frames).latent_dist.sample()
# latents: [B*25, 4, 24, 88]

latents = rearrange(latents, '(b f) c h w -> b f c h w', f=25)
# latents: [B, 25, 4, 24, 88]

target_latents = latents = latents * vae.config.scaling_factor   # scale by 0.18215
# latents: [B, 25, 4, 24, 88]
```

**Step 2 — Add diffusion noise to produce noisy latents:**
```python
noise = torch.randn_like(latents)                         # [B, 25, 4, 24, 88]
timesteps = noise_scheduler.timesteps[random_indices]      # [B]  — random timestep per sample
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
# noisy_latents: [B, 25, 4, 24, 88]

sigmas = get_sigmas(timesteps, ...)                        # [B, 1, 1, 1, 1]
inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
# inp_noisy_latents: [B, 25, 4, 24, 88]  — sigma-scaled noisy latents
```

**Step 3 — Encode and repeat the first frame as conditioning:**
```python
initial_images = batch['clips'][:, 0, :, :, :]            # [B, 3, 192, 704]  — first frame only
conditional_latents = vae.encode(initial_images).latent_dist.sample()
# conditional_latents: [B, 4, 24, 88]

conditional_latents = unet.encode_bbox_frame(conditional_latents, None)
# encode_bbox_frame does: frame_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
# conditional_latents: [B, 25, 4, 24, 88]  — first frame repeated across all 25 frames
```

**Step 4 — Concatenate noisy + first-frame on the channel dimension:**
```python
concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
# Shape: [B, 25, 8, 24, 88]
#         ↑     ↑   ↑
#         batch frames  8ch = 4 (noisy) + 4 (first-frame conditioning)
```

This is the **`sample`** that the ControlNet receives. The 8-channel design mirrors the SVD UNet's `in_channels=8` config — the UNet was pretrained to receive noisy+conditioning concatenated.

### 2.2 Inference Construction (`pipeline_video_control.py`, lines 291–337)

Inference adds **Classifier-Free Guidance (CFG)**, which doubles the batch:

```python
# latents: [B, 25, 4, 24, 88]  — pure Gaussian noise (initialized)

# CFG: duplicate for unconditional + conditional pass
latent_model_input = torch.cat([latents] * 2)             # [2B, 25, 4, 24, 88]
latent_model_input = scheduler.scale_model_input(latent_model_input, t)
# scale_model_input: latent / sqrt(sigma^2 + 1)           # [2B, 25, 4, 24, 88]

# First-frame image latent (encoded separately):
image_latents = vae.encode(image + noise_aug).latent_dist.mode()
# [B, 4, 24, 88] → for CFG: [2B, 4, 24, 88] (doubled in _encode_vae_image)
image_latents = image_latents.unsqueeze(1).repeat(1, 25, 1, 1, 1)
# [2B, 25, 4, 24, 88]

# Concatenate on channel dimension:
latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
# [2B, 25, 8, 24, 88]  ← same 8-channel structure as training
```

---

## 3. Input B: `control_cond` (the Semantic Conditioning)

This is the semantic signal that tells the ControlNet what layout to enforce.

### 3.1 Training Construction (`train_video_controlnet.py`, lines 531–537)

```python
# batch['semantic_ids']: [B, 25, 192, 704]  — integer grayscale trainIDs (0–18)

semantic_ids = rearrange(batch['semantic_ids'], 'b f h w -> (b f) h w')
# semantic_ids: [B*25, 192, 704]

bbox_em = vae_manager.encode_semantic_from_ids(semantic_ids)
# encode_semantic_from_ids:
#   IDs → one-hot [B*25, 19, 192, 704]
#   → Semantic VAE encoder → [B*25, 4, 24, 88]  (latent space, UNSCALED)

bbox_em = bbox_em * vae.config.scaling_factor   # multiply by 0.18215
# bbox_em: [B*25, 4, 24, 88]  — scaled to match RGB latent magnitude

bbox_em = rearrange(bbox_em, '(b f) c h w -> b f c h w', f=25)
# bbox_em: [B, 25, 4, 24, 88]  ← this is control_cond
```

**Why scale by 0.18215?** The RGB target latents are also multiplied by `vae.config.scaling_factor` (line 549). Scaling the semantic latents consistently puts them in the same magnitude range as the noisy RGB latents the ControlNet processes, making optimization easier.

### 3.2 Inference Construction (`pipeline_video_control.py`, lines 91–128)

For inference, `_encode_vae_condition()` handles encoding with CFG:

```python
# semantic_ids: [B, 25, 192, 704]  — provided externally

semantic_ids_flat = rearrange(semantic_ids, "b f h w -> (b f) h w")
# [B*25, 192, 704]

cond_em = self.vae_manager.encode_semantic_from_ids(semantic_ids_flat)
# [B*25, 4, 24, 88]  — unscaled

cond_em = cond_em * self.vae.config.scaling_factor
# [B*25, 4, 24, 88]  — scaled (consistent with training)

cond_em = rearrange(cond_em, "(b f) c h w -> b f c h w", f=25)
# [B, 25, 4, 24, 88]

# CFG: prepend zeros as unconditional conditioning
negative_cond_em = torch.zeros_like(cond_em)             # [B, 25, 4, 24, 88]
cond_em = torch.cat([negative_cond_em, cond_em])         # [2B, 25, 4, 24, 88]
```

At inference, the CFG unconditional pass uses **zero semantic conditioning** — the model generates without any semantic guidance for that half of the batch. The guidance formula then amplifies the difference.

---

## 4. Input C: `encoder_hidden_states` (CLIP Image Embedding)

```python
# Training (lines 522–523):
initial_images = batch['clips'][:, 0, :, :, :]           # [B, 3, 192, 704]  — first frame

# Resize to 224×224 for CLIP, normalize:
pixel_values = feature_extractor(resize(initial_images)).pixel_values
# [B, 3, 224, 224]

image_embeddings = image_encoder(pixel_values).image_embeds
# [B, 1024]  — CLIP ViT-H/14 global image embedding

encoder_hidden_states = image_embeddings.unsqueeze(1)
# [B, 1, 1024]
```

This is the CLIP embedding of the **first RGB frame** — it encodes the global scene appearance and is used in the cross-attention layers of each `CrossAttnDownBlockSpatioTemporal` inside the ControlNet. It tells the model what the scene "looks like" visually, complementing the semantic layout from `control_cond`.

---

## 5. Input D: `added_time_ids` (Auxiliary Time Metadata)

```python
# get_add_time_ids (util.py, lines 170–193):
add_time_ids = [fps-1, motion_bucket_id, noise_aug_strength]
# concrete values: [6, 127, 0.02]

add_time_ids = torch.tensor([add_time_ids]).repeat(batch_size, 1)
# [B, 3]
```

These three scalars encode video generation metadata:
- `fps - 1 = 6`: frame rate conditioning (SVD was trained on fps-1)
- `motion_bucket_id = 127`: motion amount (0=static, 255=high motion)
- `noise_aug_strength = 0.02`: how much noise was added to the conditioning image

Inside the ControlNet, these are Fourier-embedded per scalar (each projects to `addition_time_embed_dim=256` dims), concatenated to `[B, 768]`, then projected through `add_embedding` MLP to `[B, 1280]` and added to the main timestep embedding.

---

## 6. ControlNet Forward Pass: Step by Step

The forward method in `controlnet.py` processes all inputs through six stages.

### Stage 1 — Timestep Embedding (`forward`, lines 262–283)

```python
# timestep: [B]  e.g. [800] for a high-noise step
t_emb = self.time_proj(timesteps)           # Fourier embedding: [B, 320]
emb   = self.time_embedding(t_emb)          # MLP 320→1280:       [B, 1280]

time_embeds = self.add_time_proj(added_time_ids.flatten())
# added_time_ids [B, 3] → flatten → [B*3]
# add_time_proj: Fourier per-scalar → [B*3, 256]
time_embeds = time_embeds.reshape(B, -1)    # [B, 768]  (3 × 256)
aug_emb = self.add_embedding(time_embeds)   # MLP 768→1280: [B, 1280]

emb = emb + aug_emb                         # [B, 1280]  — combined time embedding
```

### Stage 2 — Flatten Batch×Frames (`forward`, lines 287–294)

```python
# sample: [B, 25, 8, 24, 88] → [B*25, 8, 24, 88]
sample = sample.flatten(0, 1) # noise rgb latents + first rgb latent ( thats how 8 channel)

# control_cond: [B, 25, 4, 24, 88] → [B*25, 4, 24, 88]
control_cond = control_cond.flatten(0, 1)  # semantic vae latent ( 4 channel)

# emb: [B, 1280] → [B*25, 1280]  — repeat for each frame
emb = emb.repeat_interleave(num_frames, dim=0)

# encoder_hidden_states: [B, 1, 1024] → [B*25, 1, 1024]
encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
```

From this point on all spatial processing happens in the `[B*F, C, H, W]` format — the temporal structure is only handled inside the `SpatioTemporal` blocks via the `image_only_indicator` flag.

### Stage 3 — Pre-Process and Single-Point Semantic Injection (`forward`, lines 297–299)

```python
sample       = self.conv_in(sample)           # Conv2d(8→320, k=3, p=1)
                                              # [B*25, 8, 24, 88] → [B*25, 320, 24, 88]

control_cond = self.control_conv_in(control_cond)  # Conv2d(4→320, k=3, p=1)
                                                   # [B*25, 4, 24, 88] → [B*25, 320, 24, 88]

sample = sample + control_cond                # ADD elementwise
                                              # [B*25, 320, 24, 88]  ← semantic injected ONCE
```

`control_conv_in` is a standard 3×3 convolution — **not** zero-initialized. It is trained from scratch to project the 4-channel semantic latents into the 320-channel feature space that matches the ControlNet's input.

### Stage 4 — Down Blocks: Feature Extraction (`forward`, lines 301–319)

An `image_only_indicator` of all zeros is passed to each block to activate video (spatiotemporal) mode:

```python
image_only_indicator = torch.zeros(B, F, dtype=sample.dtype, device=sample.device)
# [B, 25]  — zeros = full video mode (temporal attention active)
```

The loop collects all intermediate features for later zero-conv projection:

```python
down_block_res_samples = (sample,)            # start with initial [B*25, 320, 24, 88]

for downsample_block in self.down_blocks:
    sample, res_samples = downsample_block(
        hidden_states=sample,
        temb=emb,
        encoder_hidden_states=encoder_hidden_states,   # used in CrossAttn blocks
        image_only_indicator=image_only_indicator,
    )
    down_block_res_samples += res_samples
```

Each `CrossAttnDownBlockSpatioTemporal` contains:
- `num_layers=2` ResNet+TemporalAttention pairs → each outputs a residual
- 1 Downsampler (stride-2 Conv2d) at the end → outputs 1 residual

The final `DownBlockSpatioTemporal` has `num_layers=2` ResNets but **no Downsampler** (it is the deepest block, spatial resolution stops here).

**Detailed residual collection across all 4 blocks:**

```
Initial sample (before block 0):
  Index  0: [B*25,  320, 24, 88]   ← full latent resolution

Block 0 (CrossAttnDownBlock, 320→320, has downsample):
  Index  1: [B*25,  320, 24, 88]   ← ResNet+Attn layer 1 output
  Index  2: [B*25,  320, 24, 88]   ← ResNet+Attn layer 2 output
  Index  3: [B*25,  320, 12, 44]   ← Downsampler output (stride-2)
  → sample after block 0: [B*25, 320, 12, 44]

Block 1 (CrossAttnDownBlock, 320→640, has downsample):
  Index  4: [B*25,  640, 12, 44]   ← ResNet+Attn layer 1 output
  Index  5: [B*25,  640, 12, 44]   ← ResNet+Attn layer 2 output
  Index  6: [B*25,  640,  6, 22]   ← Downsampler output (stride-2)
  → sample after block 1: [B*25, 640, 6, 22]

Block 2 (CrossAttnDownBlock, 640→1280, has downsample):
  Index  7: [B*25, 1280,  6, 22]   ← ResNet+Attn layer 1 output
  Index  8: [B*25, 1280,  6, 22]   ← ResNet+Attn layer 2 output
  Index  9: [B*25, 1280,  3, 11]   ← Downsampler output (stride-2)
  → sample after block 2: [B*25, 1280, 3, 11]

Block 3 (DownBlock, 1280→1280, NO downsample — final):
  Index 10: [B*25, 1280,  3, 11]   ← ResNet layer 1 output
  Index 11: [B*25, 1280,  3, 11]   ← ResNet layer 2 output
  → sample after block 3: [B*25, 1280, 3, 11]

Total down_block_res_samples: 12 tensors
```

The deepest spatial resolution reached is **3×11** (= 24/8 × 88/8). At this resolution, a traffic light that occupied 3–4 pixels in the original image occupies a sub-pixel area — its semantic signal has been entirely lost.

### Stage 5 — Mid Block (`forward`, lines 322–327)

```python
sample = self.mid_block(
    hidden_states=sample,         # [B*25, 1280, 3, 11]
    temb=emb,
    encoder_hidden_states=encoder_hidden_states,
    image_only_indicator=image_only_indicator,
)
# sample after mid_block: [B*25, 1280, 3, 11]
```

`UNetMidBlockSpatioTemporal` contains: ResNet → SpatioTemporalAttn → ResNet. No spatial dimension change.

### Stage 6 — Zero-Conv Projection of All Features (`forward`, lines 331–344)

This is where the ControlNet's collected features become the residuals that will guide the UNet.

**`controlnet_down_blocks` (12 zero-initialized 1×1 convolutions):**

```python
# __init__ builds these:
# First one (before loop): Conv2d(320→320, k=1) — for index 0
# Per-block in loop:
#   Block 0 (2 layers + 1 downsample) → 3 Conv2d(320→320, k=1)
#   Block 1 (2 layers + 1 downsample) → 3 Conv2d(640→640, k=1)
#   Block 2 (2 layers + 1 downsample) → 3 Conv2d(1280→1280, k=1)
#   Block 3 (2 layers, no downsample) → 2 Conv2d(1280→1280, k=1)
# Total: 1 + 3 + 3 + 3 + 2 = 12 ✓

controlnet_down_block_res_samples = ()
for feature, zero_conv in zip(down_block_res_samples, self.controlnet_down_blocks):
    feature = zero_conv(feature)               # C→C, kernel=1×1, weights=0
    controlnet_down_block_res_samples += (feature,)
```

**`controlnet_mid_block` (1 zero-initialized 1×1 convolution):**

```python
# __init__: Conv2d(1280→1280, k=1), zero-initialized
mid_block_res_sample = self.controlnet_mid_block(sample)
# [B*25, 1280, 3, 11] → [B*25, 1280, 3, 11]
```

**Why zero-initialization?**
At the start of training, all 13 zero-conv layers output exactly zero. This means the UNet starts in its fully pretrained SVD state — ControlNet adds nothing initially. As training progresses, gradients flow back through the zero-convs and they learn non-zero weights, gradually steering the UNet toward semantic-conditioned outputs. This ensures training stability and prevents the random initialization of new layers from destabilizing the pretrained UNet.

**Conditioning scale multiplication:**
```python
down_block_res_samples = [s * conditioning_scale for s in down_block_res_samples]
# each of 12 residuals scaled by conditioning_scale (default 1.0)

mid_block_res_sample = mid_block_res_sample * conditioning_scale
# [B*25, 1280, 3, 11]
```

---

## 7. ControlNet Output: The 13 Residuals

The ControlNet returns two collections:

```python
return (down_block_res_samples, mid_block_res_sample)
# down_block_res_samples: list of 12 tensors — matching down_block skip connection positions
# mid_block_res_sample:   1 tensor            — matching UNet mid_block output position
```

**Complete output tensor inventory:**

| Output | Shape | Maps to UNet position |
|---|---|---|
| `down_block_res_samples[0]` | `[B*25, 320, 24, 88]` | Before UNet down_block[0] |
| `down_block_res_samples[1]` | `[B*25, 320, 24, 88]` | UNet down_block[0], layer 1 skip |
| `down_block_res_samples[2]` | `[B*25, 320, 24, 88]` | UNet down_block[0], layer 2 skip |
| `down_block_res_samples[3]` | `[B*25, 320, 12, 44]` | UNet down_block[0] downsampler skip |
| `down_block_res_samples[4]` | `[B*25, 640, 12, 44]` | UNet down_block[1], layer 1 skip |
| `down_block_res_samples[5]` | `[B*25, 640, 12, 44]` | UNet down_block[1], layer 2 skip |
| `down_block_res_samples[6]` | `[B*25, 640, 6, 22]` | UNet down_block[1] downsampler skip |
| `down_block_res_samples[7]` | `[B*25, 1280, 6, 22]` | UNet down_block[2], layer 1 skip |
| `down_block_res_samples[8]` | `[B*25, 1280, 6, 22]` | UNet down_block[2], layer 2 skip |
| `down_block_res_samples[9]` | `[B*25, 1280, 3, 11]` | UNet down_block[2] downsampler skip |
| `down_block_res_samples[10]` | `[B*25, 1280, 3, 11]` | UNet down_block[3], layer 1 skip |
| `down_block_res_samples[11]` | `[B*25, 1280, 3, 11]` | UNet down_block[3], layer 2 skip |
| `mid_block_res_sample` | `[B*25, 1280, 3, 11]` | UNet mid_block output |

---

## 8. How UNet Receives and Uses the Residuals

The UNet (`unet_spatio_temporal_condition.py`, forward method) receives the residuals as optional arguments:

```python
model_pred = unet(
    sample=concatenated_noisy_latents,              # [B, 25, 8, 24, 88]
    timestep=timesteps,
    encoder_hidden_states=encoder_hidden_states,
    added_time_ids=added_time_ids,
    down_block_additional_residuals=down_block_additional_residuals,   # 12 tensors
    mid_block_additional_residuals=mid_block_additional_residuals,     # 1 tensor
)
```

Inside the UNet's encoder, at each point where the ControlNet collected a residual, the UNet **adds** it to its own feature:

```python
# Pseudocode of UNet skip connection injection
for i, (unet_block, ctrl_residual) in enumerate(zip(down_blocks, ctrl_residuals)):
    hidden_states = unet_block(hidden_states)
    hidden_states = hidden_states + ctrl_residual       # additive residual from ControlNet
```

Similarly for the mid_block:
```python
hidden_states = self.mid_block(hidden_states)
hidden_states = hidden_states + mid_block_additional_residual
```

The UNet decoder (`up_blocks`) then uses these modified skip connections from the encoder to reconstruct the denoised output, guided by the semantic conditioning that entered via the ControlNet.

---

## 9. Complete Flow Diagram

```
DATASET
 batch['clips']:         [B, 25, 3, 192, 704]   RGB frames
 batch['semantic_ids']:  [B, 25, 192, 704]       Grayscale trainIDs 0–18
 batch['clips'][:,0]:    [B, 3, 192, 704]        First frame

         │                          │                          │
         ▼ VAE encode               ▼ Semantic VAE encode      ▼ CLIP encode
         │                          │                          │
 [B*25, 4, 24, 88]         [B*25, 4, 24, 88]          [B, 1024]
 × 0.18215 = target_latents × 0.18215 = bbox_em        unsqueeze(1)
         │                          │                    [B, 1, 1024]
         ▼ add_noise                │                          │
 [B, 25, 4, 24, 88]                │                          │
 inp_noisy_latents                  │ rearrange                │
         │                  [B, 25, 4, 24, 88]                │
         │  cat(dim=2) with                                    │
         │  first-frame repeated                               │
 [B, 25, 8, 24, 88]                │                          │
  = concatenated_noisy_latents      │ = bbox_em                │ = encoder_hidden_states
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  ControlNetModel │
                           │                 │
                           │ [Stage 1] Time  │  timesteps [B]
                           │   Embedding     │  added_time_ids [B,3]
                           │   → emb [B,1280]│
                           │                 │
                           │ [Stage 2] Flat  │
                           │  B×F→B*F        │
                           │                 │
                           │ [Stage 3]       │
                           │  conv_in(noise) │  [B*25, 320, 24, 88]
                           │  + control_conv │  [B*25, 320, 24, 88]
                           │  _in(semantic)  │  ← ONLY injection point
                           │  ────────────── │
                           │  [B*25,320,24,88]
                           │                 │
                           │ [Stage 4]       │ cross-attn uses encoder_hidden_states
                           │  down_block[0]  │  → 3 residuals (idx 1,2,3)
                           │  down_block[1]  │  → 3 residuals (idx 4,5,6)
                           │  down_block[2]  │  → 3 residuals (idx 7,8,9)
                           │  down_block[3]  │  → 2 residuals (idx 10,11)
                           │  + initial      │  → 1 residual  (idx 0)
                           │  = 12 features  │
                           │                 │
                           │ [Stage 5]       │
                           │  mid_block      │  → 1 feature
                           │  [B*25,1280,3,11]
                           │                 │
                           │ [Stage 6]       │
                           │  12× zero_conv  │  1×1 Conv, C→C, weights=0
                           │  1× zero_conv   │
                           │  × cond_scale   │
                           └────────┬────────┘
                                    │
                      ┌─────────────┴──────────────┐
                      │  12 down residuals          │  1 mid residual
                      │  + 1 mid residual           │
                      ▼                             ▼
                           ┌─────────────────┐
                           │  UNetSpatio     │
                           │  Temporal       │
                           │                 │
                           │  each residual  │
                           │  ADDED to UNet  │
                           │  skip connection│
                           │  at matching    │
                           │  position       │
                           │                 │
                           │  up_blocks      │
                           │  reconstruct    │
                           └────────┬────────┘
                                    │
                                    ▼
                           model_pred [B, 25, 4, 24, 88]
                           → denoised RGB latents
                           → MSE loss vs target_latents
```

---

## 10. Summary: Zero-Conv Layers in the ControlNet

| Layer name | Location | Count | Shape | Purpose |
|---|---|---|---|---|
| `controlnet_down_blocks[0]` | Before loop, initial | 1 | `Conv2d(320→320, k=1)` | Project initial sample feature |
| `controlnet_down_blocks[1–3]` | Block 0 (3 residuals) | 3 | `Conv2d(320→320, k=1)` | Project block 0 features |
| `controlnet_down_blocks[4–6]` | Block 1 (3 residuals) | 3 | `Conv2d(640→640, k=1)` | Project block 1 features |
| `controlnet_down_blocks[7–9]` | Block 2 (3 residuals) | 3 | `Conv2d(1280→1280, k=1)` | Project block 2 features |
| `controlnet_down_blocks[10–11]` | Block 3 (2 residuals) | 2 | `Conv2d(1280→1280, k=1)` | Project block 3 features |
| `controlnet_mid_block` | After mid_block | 1 | `Conv2d(1280→1280, k=1)` | Project mid_block feature |
| **Total** | | **13** | | |

All 13 are zero-initialized (`zero_module` wraps them with `weight.data.zero_()` and `bias.data.zero_()`). They start as identity-zero operations and learn to pass semantically meaningful residuals to the UNet.

The `control_conv_in` (Conv2d 4→320, k=3) is the **only non-zero-initialized new layer** — it is trained from scratch to project the semantic conditioning into the feature space. It is not zero-initialized because it needs to learn a meaningful projection immediately for the conditioning to have any effect.
