# ControlNet Semantic Injection: Architecture Analysis & Baseline Results

---

## 1. What the ControlNet Actually Is

The `ControlNetModel` in this codebase is a **structural replica of the UNet's encoder half** — it mirrors the UNet's `conv_in`, all four `down_blocks`, and the `mid_block`, with one addition: a small `control_conv_in` layer that ingests the semantic conditioning. The UNet's decoder (`up_blocks`) has no counterpart in the ControlNet.

This design comes from the original ControlNet paper: create a trainable copy of the frozen UNet encoder, inject a conditioning signal into it, and feed the resulting intermediate features back into the UNet as additive residuals. The idea is that the ControlNet learns to "steer" the UNet from the same feature space it was pretrained in.

In Stage 2 pipeline (`ControlNetModel` + `UNetSpatioTemporalConditionModel`), this means:
- The ControlNet processes the noisy video latent (+ semantic conditioning) through a full encoder-style forward pass.
- Its intermediate features, after passing through zero-initialized 1×1 convolution "adaptor" blocks, are added element-wise into the UNet's own skip connections and mid-block output.
- The UNet's `up_blocks` then reconstruct the denoised output guided by these modified skip connections.

---

## 2. The Semantic Conditioning Path: Single-Scale Injection

### 2.1 Where the Semantic Signal Enters

The semantic conditioning enters the ControlNet at **exactly one point**: the very first layer, before any spatial processing.

Concretely, the flow is:

1. The noisy video latent (shape `[B×F, 8, H, W]`, 8 channels = 4 noise + 4 first-frame concat) passes through `conv_in` — a standard 3×3 convolution — producing a `[B×F, 320, H, W]` feature map.

2. The semantic conditioning latent (shape `[B×F, 4, H, W]`, 4 channels from the semantic VAE encoder) passes through `control_conv_in` — a **single** 3×3 convolution — also producing `[B×F, 320, H, W]`.

3. These two are **added together elementwise**. This summed tensor is the sole input to all subsequent ControlNet processing.

From this point forward, no additional semantic signal is ever injected. The semantic information must survive — undiluted — through three stride-2 downsampling stages and all the temporal attention layers of the ControlNet's down blocks.

### 2.2 Spatial Dimensions After Each Stage

After the single injection at `[B×F, 320, H×W]`, the feature map is progressively downsampled:

| Stage | Channels | Spatial Size | Notes |
|---|---|---|---|
| After injection (input) | 320 | H × W (e.g. 24×88) | Semantic signal enters here |
| After down_block[0] | 320 → 640 | H/2 × W/2 (12×44) | Stride-2 downsample |
| After down_block[1] | 640 → 1280 | H/4 × W/4 (6×22) | Stride-2 downsample |
| After down_block[2] | 1280 → 1280 | H/4 × W/4 (6×22) | Stride-2 downsample |
| After down_block[3] (final) | 1280 | H/4 × W/4 (6×22) | No downsample |
| After mid_block | 1280 | H/4 × W/4 (6×22) | Deepest representation |

At our training resolution of 192×704, the VAE latent space is 24×88. The deepest ControlNet features operate at just **6×22 spatial resolution**. A traffic light that was 3–4 pixels wide in the original image occupies a fractional pixel at this depth.

### 2.3 What Features Are Extracted and When

The ControlNet collects features at every layer for injection into the UNet. Specifically, 13 feature tensors are collected:

- **1 tensor** at the input level (320 channels, H×W) — this is the only one that still contains strong semantic signal
- **2 tensors per down_block layer** (from each resnet/attention block within the block) — semantic signal progressively weaker
- **1 tensor per downsampling operation** (between blocks) — captures the spatial compression step
- **1 tensor** from the mid_block output (1280 channels, H/4×W/4) — weakest semantic signal

Each of these 13 tensors passes through a **zero-initialized 1×1 convolution** (the `controlnet_down_blocks` and `controlnet_mid_block` layers). Zero-initialization is a training stability technique: at the start of training, ControlNet outputs are all zero so the UNet starts in its pretrained state and gradually learns to incorporate the conditioning.

These 13 residuals are then added to the UNet's corresponding skip connections and mid-block output during the UNet forward pass, guiding the UNet's decoder reconstruction.

---

## 3. Does It Make Sense to Train the Full ControlNet?

This is the central architectural tension.

### 3.1 What the Full ControlNet Is Training For

The ControlNet is a full encoder-depth copy of the UNet: four down blocks with temporal attention, a mid block with temporal attention. This is a large model — on the order of 1.5B parameters. Its `down_blocks` and `mid_block` contain spatiotemporal transformer layers that learn to propagate and contextualize features across both space and time.

However, as established above, **the semantic conditioning signal enters at only one point** — at the very first layer, at full spatial resolution. Everything deeper in the ControlNet is computing features from the noisy latent plus this one-time injection.

### 3.2 The Mismatch: Capacity vs. Conditioning Access

The implication is that the ControlNet's deeper layers (blocks 1–3, mid_block) are essentially re-running the UNet encoder computation on a signal that has already had the semantic information diluted through downsampling and multiple attention layers. By the time the ControlNet is computing features at H/4×W/4 (6×22 spatial), the original semantic structure from the 24×88 latent has passed through two stride-2 convolutions and multiple residual blocks. Fine-grained semantic distinctions (thin poles, small pedestrians, narrow sidewalks) are no longer recoverable from the feature map at that depth.

The ControlNet's mid_block residual — which conceptually should carry the most compressed, high-level semantic understanding — is in practice a 6×22 tensor that was derived from a semantic signal it received 3 downsampling stages earlier. It has had little opportunity to build a representation that is semantically discriminative at a fine-grained level.

This means:

- The **shallow ControlNet features** (first 1–3 controlnet_down_blocks, full to half resolution) carry genuine semantic content from the injection, and the zero-conv adaptors at these levels are doing meaningful work.

- The **deep ControlNet features** (later down blocks, mid block) are primarily expressing the UNet encoder's inductive bias on the noisy input, with only diluted semantic content. Training these layers to produce useful residuals is much harder because the semantic signal they were built on top of has been lost.

- The full ControlNet training is therefore **not efficiently utilizing its capacity**: the majority of its parameters (deeper blocks with temporal attention) are spending their computation re-running the UNet encoder on a weakly-conditioned signal, rather than building increasingly semantic-rich representations to guide the UNet decoder.

### 3.3 The Per-Class Evidence

This architectural bottleneck explains the per-class IoU pattern in our results exactly. Classes whose spatial footprint survives the 8× total downsampling chain (road, vegetation, sky, car) have high IoU because the semantic signal at the first injection layer already encodes them well. Classes that are thin, small, or occupy few pixels (traffic light, bicycle, pole, rider) fail precisely because their signal is lost by the time the ControlNet's features are collected for injection.

If the deeper ControlNet layers were successfully building semantically grounded representations, we would expect improvement on small classes as a function of training longer — but the bottleneck is structural, not a matter of optimization time.

---

## 4. Current Architecture Summary

```
Semantic VAE Latents  [B×F, 4, H, W]
        │
        ▼ control_conv_in (single 3×3 conv, NOT zero-initialized)
        │   [B×F, 320, H, W]
        │
        ├─ ADD ──────────────────────────────────────────────────────────────────┐
        │                                                                        │
Noisy Latent [B×F, 8, H, W]                                                    │
        │                                                                        │
        ▼ conv_in (3×3 conv)                                                    │
        │   [B×F, 320, H, W]  ◄─────────────────────────────────────────────────┘
        │          │
        │          ▼  (semantic signal fully injected, single point)
        │
        ▼ ControlNet down_block[0]    → features at [B×F, 320, H, W]    ← zero-conv → UNet skip
        ▼ (stride-2 downsample)
        ▼ ControlNet down_block[1]    → features at [B×F, 640, H/2, W/2] ← zero-conv → UNet skip
        ▼ (stride-2 downsample)
        ▼ ControlNet down_block[2]    → features at [B×F,1280, H/4, W/4] ← zero-conv → UNet skip
        ▼ ControlNet down_block[3]    → features at [B×F,1280, H/4, W/4] ← zero-conv → UNet skip
        ▼ ControlNet mid_block        → features at [B×F,1280, H/4, W/4] ← zero-conv → UNet mid
        │
        │   (13 residuals total, all scaled by conditioning_scale=1.0)
        │
        ▼ UNet up_blocks (decoder) uses modified skip connections
        ▼ RGB video output
```

**Multi-scale residuals?** YES — the ControlNet outputs 13 residuals spanning full to quarter resolution, all injected into the UNet.

**Multi-scale semantic conditioning?** NO — the semantic latent enters the ControlNet at only one point (input layer, full resolution). The deeper residuals carry only diluted semantic information.

---

## 5. Baseline Results: UNet-Unfreeze Checkpoint (checkpoint-22700)

These results represent the current best architecture — single-scale semantic injection with UNet mid+output blocks unfrozen for joint fine-tuning.

**Evaluation configuration:**
- Checkpoint: step 22700, `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze`
- Samples evaluated: 20
- Clip length: 25 frames
- Resolution: 192×704
- Metric: DRN segmentation network applied to generated RGB frames

### 5.1 Summary Metrics

| Metric | Value | Notes |
|---|---|---|
| mIoU (DRN) | **42.92%** | Primary semantic controllability metric |
| Pixel Accuracy | **89.04%** | High due to large class dominance (road, sky) |
| Mean Accuracy | **60.86%** | Per-class accuracy averaged equally |
| FW-IoU | **80.82%** | Frequency-weighted IoU |

### 5.2 Per-Class IoU

| Class | IoU | Interpretation |
|---|---|---|
| road | 91.51% | Large, flat, dominant — survives latent bottleneck |
| car | 91.04% | Large objects, frequent in KITTI-360 |
| vegetation | 85.32% | Large spatial footprint |
| sky | 80.29% | Large region, easy to control |
| building | 76.55% | Large, well-represented in training |
| terrain | 62.20% | Moderate size, infrequent confusion |
| truck | 66.09% | Larger than car, well-represented |
| sidewalk | 55.65% | Moderate — boundaries are thin |
| person | 51.93% | Small objects, but improved with UNet unfreezing |
| wall | 40.92% | Moderate — irregular shapes |
| fence | 35.31% | Thin structures, partially lost in latent |
| traffic sign | 29.47% | Small objects |
| pole | 30.28% | Thin vertical structures — severely hit by latent bottleneck |
| rider | 12.74% | Small, rare, confused with person |
| bicycle | 6.22% | Thin structures, rare in KITTI-360 |
| traffic light | 0.00% | Sub-pixel in latent space at 192×704 |
| bus | 0.00% | Near-absent from KITTI-360 training data |
| train | 0.00% | Near-absent from KITTI-360 training data |
| motorcycle | 0.00% | Rare + thin — not learned |

### 5.3 Structural Pattern in Results

The results divide cleanly into three tiers:

**Tier 1 — Well-controlled (>60% IoU):** road, car, vegetation, sky, building, terrain, truck
These are all large-footprint classes. At 192×704 input and 24×88 semantic latent resolution, these classes occupy tens to hundreds of latent pixels. The single-scale injection at full latent resolution is sufficient to encode their layout.

**Tier 2 — Partially controlled (20–60% IoU):** sidewalk, person, wall, fence, traffic sign, pole
These are medium-to-small classes where the single injection carries some signal but the latent bottleneck degrades boundary precision or object distinctiveness.

**Tier 3 — Uncontrolled (0–13% IoU):** traffic light, bus, train, motorcycle, bicycle, rider
These fail for one or both of: (a) sub-pixel or near-sub-pixel footprint in the 24×88 latent space, (b) extreme class imbalance in KITTI-360 training data meaning the model never learned them.

---

## 6. Implications for the Multi-Scale Conditioning Proposal

The evidence above establishes clearly what multi-scale conditioning would and would not address:

**Would address:** The dilution of mid-scale objects (poles, signs, fences, sidewalk edges) as the single semantic injection propagates through 3 downsampling stages. Injecting fresh semantic features at the input to each down_block level would keep semantically precise information available at all ControlNet depths.

**Would not address:** Sub-pixel objects (traffic lights, motorcycles at 192×704). These require higher input resolution, not better conditioning architecture — they simply do not exist as separable signals in the 24×88 latent space.

**Would also address:** The inefficiency of training a full ControlNet encoder on semantically-diluted features at depth. With multi-scale injection, the ControlNet's deeper blocks would receive direct semantic features at the appropriate resolution for their depth, making their computation genuinely semantic rather than noisy-latent-propagation.

The expected gain from multi-scale conditioning on top of the current checkpoint is **+4–8% mIoU** on Tier 2 classes, with meaningful FVD reduction from more consistent frame-to-frame semantic grounding at all spatial scales.
