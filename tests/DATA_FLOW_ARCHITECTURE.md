# 🔄 Ctrl-V with Semantic VAE: Complete Data Flow Architecture

## Overview

This document visualizes the complete data flow from input to output for the **Ctrl-V** video generation model with **Semantic VAE** integration.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  RGB Clip (25 frames)              Semantic IDs (25 frames)            │
│  [B, 25, 3, 128, 512]              [B, 25, 128, 512]                   │
│  ↓ Preprocessed KITTI-360          ↓ Official KITTI-360                │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: BBOX/SEMANTIC PREDICTION                    │
│                      (VideoDiffusionPipeline)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Initial Frame Conditioning + Semantic VAE Encoding                     │
│  → Noise Addition → UNet3D Denoising → VAE Decoding                    │
│  Output: Predicted Semantic/BBox Frames [25, 3, 128, 512]              │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: VIDEO GENERATION                            │
│                   (StableVideoControlPipeline)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Initial Frame + Stage 1 Output (ControlNet) + Semantic VAE            │
│  → Noise Addition → UNetSpatioTemporal + ControlNet → VAE Decoding    │
│  Output: Final Video Frames [25, 3, 128, 512]                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Detailed Stage 1: BBox/Semantic Prediction

### Input Processing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1 INPUTS                                 │
└─────────────────────────────────────────────────────────────────────────┘

1. Initial Frame (RGB)                     2. Semantic IDs (Ground Truth)
   [1, 3, 128, 512]                          [25, 128, 512]
         ↓                                           ↓
   CLIP Image Encoder                        Semantic VAE Encoder
   (stabilityai/svd)                         (Custom trained)
         ↓                                           ↓
   Image Embeddings                          One-Hot Encoding
   [1, 1024]                                 [25, 19, 128, 512]
         ↓                                           ↓
   Broadcast to [25, ...]                    Semantic Stem (Conv layers)
         ↓                                           ↓
         │                                   RGB VAE Encoder (frozen core)
         │                                           ↓
         │                                   Latent Space
         │                                   [(25), 4, 16, 64]
         │                                           ↓
         └──────────────┬────────────────────────────┘
                        ↓
```

### Latent Space Conditioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC VAE LATENT ENCODING                         │
└─────────────────────────────────────────────────────────────────────────┘

Semantic IDs [25, 128, 512]
      ↓
Remap KITTI-360 IDs → TrainIDs (0-18)
      ↓
One-Hot Encode [25, 19, 128, 512]
      ↓
┌────────────────────────────────────┐
│  Semantic VAE Encoder              │
│  ┌──────────────────────────────┐  │
│  │ Semantic Stem (Trainable)    │  │
│  │  Conv2d(19 → 3)              │  │
│  │  InstanceNorm2d              │  │
│  │  SiLU activation             │  │
│  └──────────────────────────────┘  │
│              ↓                     │
│  ┌──────────────────────────────┐  │
│  │ RGB VAE Encoder (Frozen)     │  │
│  │  From: stabilityai/svd       │  │
│  │  Conv layers + Downsampling  │  │
│  │  Attention layers            │  │
│  └──────────────────────────────┘  │
│              ↓                     │
│  Latent Space [25, 4, 16, 64]     │
│  ↓                                 │
│  Scale by 0.18215                  │
└────────────────────────────────────┘
      ↓
Semantic Latents [25, 4, 16, 64]
```

### Conditional Latents Construction

```
┌─────────────────────────────────────────────────────────────────────────┐
│              CONDITIONAL LATENT CONSTRUCTION                            │
└─────────────────────────────────────────────────────────────────────────┘

Initial Frame [1, 3, 128, 512]
      ↓
RGB VAE Encode
      ↓
Initial Latent [1, 4, 16, 64]
      ↓
Repeat for temporal dimension
      ↓
Initial Latents [25, 4, 16, 64]

                    +
                    
Semantic Latents [25, 4, 16, 64]
(From Semantic VAE)

      ↓
┌────────────────────────────────────┐
│  Conditional Latents               │
│  [25, 4, 16, 64]                   │
│                                    │
│  These guide the denoising process │
│  during diffusion                  │
└────────────────────────────────────┘
```

### Diffusion Process (Stage 1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DIFFUSION DENOISING LOOP                            │
└─────────────────────────────────────────────────────────────────────────┘

Random Noise Latents [25, 4, 16, 64]
      ↓
      │  ┌─────────────────────────────────────┐
      │  │  Timestep t = T, T-1, ..., 1        │
      │  └─────────────────────────────────────┘
      │                 ↓
      ├───> Concatenate with Conditional Latents
      │              ↓
      │     ┌─────────────────────────────────┐
      │     │  UNet3DConditionModel            │
      │     │  ┌──────────────────────────┐    │
      │     │  │ Down Blocks (Conv3D)     │    │
      │     │  │  + Temporal Attention    │    │
      │     │  └──────────────────────────┘    │
      │     │            ↓                     │
      │     │  ┌──────────────────────────┐    │
      │     │  │ Middle Block             │    │
      │     │  │  Spatial + Temporal Attn │    │
      │     │  │  Cross-Attention with    │    │
      │     │  │  Image Embeddings        │    │
      │     │  └──────────────────────────┘    │
      │     │            ↓                     │
      │     │  ┌──────────────────────────┐    │
      │     │  │ Up Blocks (Conv3D)       │    │
      │     │  │  + Temporal Attention    │    │
      │     │  │  + Skip Connections      │    │
      │     │  └──────────────────────────┘    │
      │     └─────────────────────────────────┘
      │                 ↓
      │     Predicted Noise ε_θ(z_t, t, cond)
      │                 ↓
      │     DDPM Denoising Step:
      │     z_{t-1} = (z_t - ε_θ)/√α - √(1-α)·ε_θ
      │                 ↓
      └─────────────────┘
             (Loop until t=0)
                    ↓
      Denoised Latents [25, 4, 16, 64]
```

### VAE Decoding (Stage 1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RGB VAE DECODER                                  │
└─────────────────────────────────────────────────────────────────────────┘

Denoised Latents [25, 4, 16, 64]
      ↓
Unscale by 0.18215
      ↓
┌────────────────────────────────────┐
│  RGB VAE Decoder                   │
│  (AutoencoderKLTemporalDecoder)    │
│  ┌──────────────────────────────┐  │
│  │ Temporal Up-sample           │  │
│  │ Conv layers                  │  │
│  └──────────────────────────────┘  │
│              ↓                     │
│  ┌──────────────────────────────┐  │
│  │ Spatial Up-sample            │  │
│  │ Conv + Attention layers      │  │
│  │ Upsample 16×64 → 128×512     │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
      ↓
Stage 1 Output: Predicted Semantic/BBox Frames
[25, 3, 128, 512]
```

---

## 🎬 Detailed Stage 2: Video Generation

### Input Processing (Stage 2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 2 INPUTS                                 │
└─────────────────────────────────────────────────────────────────────────┘

1. Initial Frame (Same as Stage 1)   2. Stage 1 Output (ControlNet Input)
   [1, 3, 128, 512]                     [25, 3, 128, 512]
         ↓                                      ↓
   CLIP Image Encoder                   RGB VAE Encode (if RGB)
         ↓                               OR
   Image Embeddings                     Semantic VAE Encode (if Semantic IDs)
   [1, 1024]                                    ↓
         ↓                               Control Latents
   Broadcast to [25, ...]               [25, 4, 16, 64]
         ↓                                      ↓
         │                              ┌──────────────────────┐
         │                              │  ControlNet Model    │
         │                              │  (Trained adapter)   │
         │                              │  Processes control   │
         │                              │  conditions          │
         │                              └──────────────────────┘
         │                                      ↓
         │                              Control Features
         │                              (Multiple scales)
         │                                      ↓
         └──────────────┬────────────────────────┘
                        ↓
```

### Diffusion Process with ControlNet (Stage 2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              CONTROLNET-GUIDED DIFFUSION DENOISING                      │
└─────────────────────────────────────────────────────────────────────────┘

Random Noise Latents [25, 4, 16, 64]
      ↓
      │  ┌─────────────────────────────────────┐
      │  │  Timestep t = T, T-1, ..., 1        │
      │  └─────────────────────────────────────┘
      │                 ↓
      ├───> ┌───────────────────────────────────┐
      │     │  ControlNet                       │
      │     │  Input: Control Latents           │
      │     │  [25, 4, 16, 64]                  │
      │     │         ↓                         │
      │     │  Encoder blocks (same as UNet)    │
      │     │         ↓                         │
      │     │  Down-sample features             │
      │     │  at multiple scales               │
      │     │         ↓                         │
      │     │  Zero-conv layers                 │
      │     │  (Initialized to 0)               │
      │     └───────────────────────────────────┘
      │                 ↓
      │        Control Features (multi-scale)
      │                 ↓
      ├───> ┌───────────────────────────────────┐
      │     │  UNetSpatioTemporalConditionModel │
      │     │  ┌──────────────────────────┐     │
      │     │  │ Down Blocks              │     │
      │     │  │  + Control Features      │ ← Added here
      │     │  │  + Temporal Attention    │     │
      │     │  │  + Spatial Attention     │     │
      │     │  └──────────────────────────┘     │
      │     │            ↓                      │
      │     │  ┌──────────────────────────┐     │
      │     │  │ Middle Block             │     │
      │     │  │  + Control Features      │ ← Added here
      │     │  │  + Cross-Attention       │     │
      │     │  │    (Image Embeddings)    │     │
      │     │  └──────────────────────────┘     │
      │     │            ↓                      │
      │     │  ┌──────────────────────────┐     │
      │     │  │ Up Blocks                │     │
      │     │  │  + Control Features      │ ← Added here
      │     │  │  + Skip Connections      │     │
      │     │  └──────────────────────────┘     │
      │     └───────────────────────────────────┘
      │                 ↓
      │     Predicted Noise ε_θ(z_t, t, cond, ctrl)
      │                 ↓
      │     DDPM Denoising Step with CFG:
      │     z_{t-1} = denoise(z_t) + guidance_scale * (cond - uncond)
      │                 ↓
      └─────────────────┘
             (Loop until t=0)
                    ↓
      Denoised Latents [25, 4, 16, 64]
```

### VAE Decoding (Stage 2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   FINAL VIDEO DECODING                                  │
└─────────────────────────────────────────────────────────────────────────┘

Denoised Latents [25, 4, 16, 64]
      ↓
Unscale by 0.18215
      ↓
┌────────────────────────────────────┐
│  RGB VAE Decoder                   │
│  (AutoencoderKLTemporalDecoder)    │
│  - Temporal upsampling             │
│  - Spatial upsampling              │
│  - Conv + Attention layers         │
└────────────────────────────────────┘
      ↓
Final Generated Video Frames
[25, 3, 128, 512]
```

---

## 🔧 Key Components Deep Dive

### 1. Semantic VAE Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC VAE (Native)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Semantic IDs [H, W] (values 0-18)                              │
│         ↓                                                               │
│  One-Hot Encode → [19, H, W]                                           │
│         ↓                                                               │
│  ┌─────────────────────────────────────────────────────┐               │
│  │ ENCODER                                              │               │
│  │  Semantic Stem (Trainable):                         │               │
│  │    Conv2d(19 → 3, kernel=3, padding=1)              │               │
│  │    InstanceNorm2d(3)                                 │               │
│  │    SiLU()                                            │               │
│  │         ↓                                            │               │
│  │  RGB VAE Encoder Core (Frozen):                     │               │
│  │    Down-sample blocks (Conv + Attention)            │               │
│  │    H×W → H/8×W/8 → H/16×W/16                        │               │
│  │    Channels: 3 → 128 → 256 → 512 → 512             │               │
│  │         ↓                                            │               │
│  │  Conv to mean/logvar                                 │               │
│  │    Conv2d(512 → 8)                                   │               │
│  │         ↓                                            │               │
│  │  Reparameterization:                                 │               │
│  │    z = μ + σ·ε, ε ~ N(0,1)                          │               │
│  │         ↓                                            │               │
│  │  Latent: [4, H/8, W/8]                              │               │
│  │    (For 128×512: [4, 16, 64])                       │               │
│  └─────────────────────────────────────────────────────┘               │
│         ↓                                                               │
│  ┌─────────────────────────────────────────────────────┐               │
│  │ DECODER                                              │               │
│  │  RGB VAE Decoder Core (Frozen):                     │               │
│  │    Up-sample blocks (ConvTranspose + Attention)     │               │
│  │    H/8×W/8 → H/4×W/4 → H/2×W/2 → H×W               │               │
│  │    Channels: 4 → 512 → 256 → 128 → 128             │               │
│  │         ↓                                            │               │
│  │  Semantic Head (Trainable):                         │               │
│  │    Conv2d(128 → 64)                                  │               │
│  │    InstanceNorm2d(64)                                │               │
│  │    SiLU()                                            │               │
│  │    Conv2d(64 → 19)                                   │               │
│  │         ↓                                            │               │
│  │  Output: Semantic Logits [19, H, W]                 │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Training Loss:                                                         │
│    L = L_CE + λ₁·L_dice + λ₂·L_boundary                                │
│    - Cross-Entropy for pixel classification                            │
│    - Dice Loss for class balance                                       │
│    - Boundary Loss for edge sharpness                                  │
│                                                                         │
│  Trained on: KITTI-360 grayscale semantic labels                       │
│  Performance: 81.51% IoU on validation set                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Initial Frame Conditioning (Padding Strategy)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   INITIAL FRAME CONDITIONING                            │
└─────────────────────────────────────────────────────────────────────────┘

Initial Frame [1, 3, H, W]
      ↓
CLIP Image Encoder
      ↓
Image Embeddings [1, 1024]
      ↓
Project to sequence length
      ↓
Broadcast/Repeat for temporal dimension
      ↓
Conditioning Embeddings [25, 1024]
      ↓
Used in Cross-Attention layers of UNet
      
     AND
      
Initial Frame [1, 3, H, W]
      ↓
RGB VAE Encode
      ↓
Initial Latent [1, 4, H/8, W/8]
      ↓
Repeat for temporal dimension
      ↓
Initial Latents [25, 4, H/8, W/8]
      ↓
Concatenated with noise latents in channel dimension
OR
Added to noise latents
      ↓
Fed to UNet input
```

### 3. Noise Scheduling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DDPM NOISE SCHEDULE                                │
└─────────────────────────────────────────────────────────────────────────┘

Forward Process (Data → Noise):
  q(z_t | z_0) = N(z_t; √(ᾱ_t)·z_0, (1-ᾱ_t)·I)
  
  where:
    β_t: noise schedule (e.g., linear from 0.0001 to 0.02)
    α_t = 1 - β_t
    ᾱ_t = ∏(α_s) for s=1 to t
    
  At t=0: Clean latent
  At t=T: Pure Gaussian noise

Reverse Process (Noise → Data):
  p_θ(z_{t-1} | z_t) = N(z_{t-1}; μ_θ(z_t, t), Σ_θ(z_t, t))
  
  where μ_θ is predicted by UNet

Denoising Step:
  z_{t-1} = (1/√α_t) · (z_t - (β_t/√(1-ᾱ_t))·ε_θ(z_t, t))
            + √β_t · ε,  ε ~ N(0, I)

Classifier-Free Guidance:
  ε_θ(z_t, t, cond) = ε_θ(z_t, t, ∅) + w·(ε_θ(z_t, t, cond) - ε_θ(z_t, t, ∅))
  
  where w is guidance scale (linearly interpolated from min to max)
```

---

## 📈 Complete Data Flow Summary

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END DATA FLOW                             │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  INPUTS                                                                   │
│  ┌────────────────────┐          ┌──────────────────────┐               │
│  │ RGB Frames         │          │ Semantic IDs         │               │
│  │ [25, 3, 128, 512]  │          │ [25, 128, 512]       │               │
│  │ From preprocessed  │          │ From official        │               │
│  │ KITTI-360          │          │ KITTI-360            │               │
│  └────────────────────┘          └──────────────────────┘               │
│           ↓                                   ↓                           │
│  ┌────────────────────┐          ┌──────────────────────┐               │
│  │ Extract Frame 0    │          │ One-Hot Encode       │               │
│  │ [3, 128, 512]      │          │ [25, 19, 128, 512]   │               │
│  └────────────────────┘          └──────────────────────┘               │
│           ↓                                   ↓                           │
│  ┌────────────────────┐          ┌──────────────────────┐               │
│  │ CLIP Encoder       │          │ Semantic VAE Encoder │               │
│  │ → [1024]           │          │ → [25, 4, 16, 64]    │               │
│  └────────────────────┘          └──────────────────────┘               │
│           ↓                                   ↓                           │
│           └───────────────┬───────────────────┘                           │
│                           ↓                                               │
│  ┌───────────────────────────────────────────────────────┐               │
│  │              STAGE 1: BBOX PREDICTION                 │               │
│  │  ┌─────────────────────────────────────────────────┐  │               │
│  │  │ UNet3D Denoising (30 steps)                     │  │               │
│  │  │ Input: Noise + Conditioning                     │  │               │
│  │  │ Guidance: Image embeddings + Semantic latents   │  │               │
│  │  └─────────────────────────────────────────────────┘  │               │
│  │                        ↓                              │               │
│  │  ┌─────────────────────────────────────────────────┐  │               │
│  │  │ RGB VAE Decoder                                 │  │               │
│  │  │ [25, 4, 16, 64] → [25, 3, 128, 512]             │  │               │
│  │  └─────────────────────────────────────────────────┘  │               │
│  └───────────────────────────────────────────────────────┘               │
│                           ↓                                               │
│           Stage 1 Output: Predicted Semantic Frames                       │
│                  [25, 3, 128, 512]                                        │
│                           ↓                                               │
│           ┌───────────────┴───────────────┐                               │
│           ↓                               ↓                               │
│  ┌────────────────────┐          ┌──────────────────────┐               │
│  │ Same Frame 0       │          │ Encode through       │               │
│  │ CLIP embeddings    │          │ Semantic VAE         │               │
│  │ [1024]             │          │ → [25, 4, 16, 64]    │               │
│  └────────────────────┘          └──────────────────────┘               │
│           ↓                                   ↓                           │
│           └───────────────┬───────────────────┘                           │
│                           ↓                                               │
│  ┌───────────────────────────────────────────────────────┐               │
│  │              STAGE 2: VIDEO GENERATION                │               │
│  │  ┌─────────────────────────────────────────────────┐  │               │
│  │  │ ControlNet processes Stage 1 latents            │  │               │
│  │  │ Produces multi-scale control features           │  │               │
│  │  └─────────────────────────────────────────────────┘  │               │
│  │                        ↓                              │               │
│  │  ┌─────────────────────────────────────────────────┐  │               │
│  │  │ UNetSpatioTemporal Denoising (25 steps)         │  │               │
│  │  │ Input: Noise + Image embeddings                 │  │               │
│  │  │ Control: ControlNet features from Stage 1       │  │               │
│  │  │ Guidance: Semantic VAE latents                  │  │               │
│  │  └─────────────────────────────────────────────────┘  │               │
│  │                        ↓                              │               │
│  │  ┌─────────────────────────────────────────────────┐  │               │
│  │  │ RGB VAE Decoder                                 │  │               │
│  │  │ [25, 4, 16, 64] → [25, 3, 128, 512]             │  │               │
│  │  └─────────────────────────────────────────────────┘  │               │
│  └───────────────────────────────────────────────────────┘               │
│                           ↓                                               │
│              FINAL OUTPUT: Generated Video                                │
│                  [25, 3, 128, 512]                                        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🎨 Latent Space Dimensions

| Component | Input | Latent | Output |
|-----------|-------|--------|--------|
| **RGB VAE** | [3, 128, 512] | [4, 16, 64] | [3, 128, 512] |
| **Semantic VAE** | [19, 128, 512] | [4, 16, 64] | [19, 128, 512] |
| **Spatial Compression** | 128×512 | 16×64 | - |
| **Compression Ratio** | 1x | 8x | - |

**Key Insight**: Both RGB and Semantic VAEs produce the **same latent shape** [4, 16, 64], enabling seamless integration in the diffusion model.

---

## ⚙️ Model Parameters

| Model | Trainable Params | Frozen Params | Total |
|-------|------------------|---------------|-------|
| **Semantic VAE Stem** | ~5K | - | ~5K |
| **Semantic VAE Head** | ~50K | - | ~50K |
| **RGB VAE (Encoder)** | - | ~34M | ~34M |
| **RGB VAE (Decoder)** | - | ~49M | ~49M |
| **UNet3D (Stage 1)** | ~1.2B | - | ~1.2B |
| **UNet Spatio-Temporal (Stage 2)** | ~1.5B | - | ~1.5B |
| **ControlNet** | ~400M | - | ~400M |
| **CLIP Image Encoder** | - | ~150M | ~150M |

---

## 📝 Notes

1. **Semantic VAE Training**: Only stem (19→3 conv) and head (128→19 conv) are trainable. The RGB VAE core remains frozen.

2. **Latent Alignment**: The key innovation is that semantic IDs are mapped to the **same latent space** as RGB images through the hybrid VAE architecture.

3. **Conditioning Strategy**: Initial frame provides both:
   - High-level semantic guidance (via CLIP embeddings)
   - Low-level pixel guidance (via latent concatenation/addition)

4. **ControlNet Zero-Init**: Control features start from zero (via zero-initialized convolutions), allowing gradual learning without disrupting pre-trained UNet.

5. **Guidance Scaling**: Linear interpolation from `min_guidance_scale` to `max_guidance_scale` across frames provides better temporal consistency.

---

Generated by: Ctrl-V Semantic VAE Integration  
Date: February 10, 2026
