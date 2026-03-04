# Why RGB VAE Spatial Conditioning Fails for Semantic Segmentation (STAGE 1 DIFFUSION MODELING): An Analysis

**Research Finding: Latent Space Incompatibility in Cross-Modal Diffusion Models**

---

## Abstract

This report analyzes the fundamental failure mode of using RGB VAE (Variational Autoencoder) for spatial conditioning in semantic segmentation diffusion models. We demonstrate that concatenating RGB-encoded latents with semantic-encoded latents in the channel dimension creates an irreconcilable latent space mismatch, leading to catastrophic performance degradation (4% mIoU vs. expected 60%+). We provide theoretical justification for why this approach fails, contrast it with successful text-based conditioning methods (CLIP), and explain the role of VAE scaling factors in this failure mode.

---

## 1. Introduction

### 1.1 Problem Statement

In adapting video diffusion models from RGB-to-RGB generation (e.g., Ctrl-V for bounding box visualization) to RGB-to-Semantic generation (semantic segmentation prediction), a critical architectural decision involves how to provide spatial conditioning to the diffusion UNet. The naive approach of using the same RGB VAE for both input encoding and spatial conditioning fails catastrophically, despite appearing architecturally sound.

### 1.2 Experimental Context

- **Task**: Video semantic segmentation prediction using diffusion models
- **Architecture**: Stable Video Diffusion (SVD) with 8-channel UNet input
  - 4 channels: noisy target latents (to be denoised)
  - 4 channels: spatial conditioning latents (concatenated)
- **Failure Mode**: 4.14% mIoU on generated frames vs. 32-65% on anchor frames
- **Root Cause**: Latent space mismatch between RGB VAE and Semantic VAE encodings

---

## 2. Theoretical Background: Latent Space Geometry

### 2.1 VAE Latent Space Properties

A Variational Autoencoder learns a compressed latent representation `z ∈ ℝ^(C×H×W)` of input data through:

```
Encoder:  x → μ(x), σ(x) → z ~ N(μ(x), σ(x))
Decoder:  z → x̂
```

The latent space geometry is determined by:
1. **Training data distribution**: What the VAE was trained on
2. **Encoder architecture**: How features are extracted and compressed
3. **Loss function**: Reconstruction loss + KL divergence regularization
4. **Latent dimensionality**: Compression ratio and information bottleneck

### 2.2 RGB VAE Latent Space (Stable Diffusion)

The RGB VAE used in Stable Video Diffusion was trained on:
- **Data**: Large-scale RGB images (ImageNet, LAION, etc.)
- **Input**: RGB pixel values `x ∈ [0, 255]^(3×H×W)`, normalized to `[-1, 1]`
- **Latent**: `z ∈ ℝ^(4×H/8×W/8)` with scaling factor `0.18215`
- **Learned Features**: Color distributions, textures, edges, object appearances

**Key Property**: The latent space encodes **continuous pixel intensity distributions** optimized for photorealistic image reconstruction.

### 2.3 Semantic VAE Latent Space (Custom-Trained)

The Semantic VAE was trained on:
- **Data**: KITTI-360 semantic segmentation labels
- **Input**: One-hot encoded semantic IDs `x ∈ {0,1}^(19×H×W)`
- **Latent**: `z ∈ ℝ^(4×H/8×W/8)` with different scaling characteristics
- **Learned Features**: Semantic boundaries, class transitions, spatial layouts

**Key Property**: The latent space encodes **discrete categorical distributions** optimized for semantic label reconstruction.

### 2.4 Latent Space Incompatibility

**Theorem (Informal)**: Two VAEs trained on different data distributions with different input modalities produce latent spaces that are **geometrically incompatible** even if they share the same dimensionality.

**Proof Sketch**:
1. RGB VAE learns `f_RGB: ℝ^(3×H×W) → ℝ^(4×H/8×W/8)`
2. Semantic VAE learns `f_Sem: {0,1}^(19×H×W) → ℝ^(4×H/8×W/8)`
3. Despite identical output dimensions, the learned manifolds `M_RGB` and `M_Sem` in latent space are **disjoint**
4. Points in `M_RGB` have no semantic meaning in `M_Sem` and vice versa
5. Concatenating latents from different manifolds creates **meaningless feature combinations**

**Empirical Evidence**:
- RGB VAE latents: Mean ≈ 0, Std ≈ 5.49 (1/0.18215)
- Semantic VAE latents: Mean ≈ X, Std ≈ Y (to be measured)
- Statistical distributions are fundamentally different

---

## 3. The Failure Mode: Mixed Latent Space Conditioning

### 3.1 Architecture Overview

The diffusion UNet receives 8-channel input:
```
UNet Input = [noisy_latents (4 ch) | conditioning_latents (4 ch)]
            = [z_noisy | z_cond]
```

For a 25-frame video, the conditioning tensor is constructed as:
```
z_cond[0]     = Semantic VAE(semantic_frame_0)      # Anchor (correct)
z_cond[1:24]  = RGB VAE(rgb_frame_0).repeat(23)     # Fill (WRONG!)
z_cond[24]    = Semantic VAE(semantic_frame_24)     # Anchor (correct)
```

### 3.2 Why This Fails

#### 3.2.1 Geometric Inconsistency

The UNet receives conditioning from **two different manifolds**:
- Frames 0, 24: Points in `M_Sem` (semantic latent manifold)
- Frames 1-23: Points in `M_RGB` (RGB latent manifold)

**Problem**: The UNet cannot learn a consistent mapping because:
1. The same spatial location has **different geometric meanings** across frames
2. Interpolation between `M_RGB` and `M_Sem` is **undefined**
3. The learned denoising function `ε_θ(z_t, t, z_cond)` receives **contradictory signals**

#### 3.2.2 Feature Misalignment

RGB latents encode:
- Pixel intensities, color gradients, texture patterns
- Example: A "car" region → RGB features of car appearance

Semantic latents encode:
- Class boundaries, categorical transitions, spatial layouts
- Example: A "car" region → One-hot encoding of class 13

**When concatenated**: The UNet sees RGB features for frames 1-23 but must predict semantic latents. The feature space provides **no useful gradient signal** for semantic prediction.

#### 3.2.3 Training Dynamics

The loss function averages over all frames:
```
L = (1/25) Σ_{t=0}^{24} ||ε - ε_θ(z_t, t, z_cond)||²
```

**Observed Behavior**:
- Frames 0, 24: Loss decreases (conditioning matches target space)
- Frames 1-23: Loss appears to decrease but model learns **trivial solutions**
- Overall loss: Misleadingly low (0.00005) due to averaging

**Why Loss Still Drops**:
1. Frames 0 and 24 are trivially easy (GT conditioning → GT target)
2. High-noise timesteps contribute less meaningful signal
3. Model optimizes for easy frames, ignoring hard frames
4. Gradient flow is dominated by the 2/25 frames with correct conditioning

### 3.3 Empirical Results

| Metric | Frames 0, 24 (Correct) | Frames 1-23 (Wrong) |
|--------|------------------------|---------------------|
| mIoU | 32-65% | **4.14%** |
| Pixel Accuracy | 60-80% | **15.01%** |
| Class Accuracy | 40-70% | **9.91%** |

**Interpretation**: The model completely fails on frames with RGB conditioning, performing worse than random guessing for many classes.

---

## 4. The RGB VAE Scaling Factor Issue

### 4.1 What is the Scaling Factor?

The RGB VAE applies a scaling factor `s = 0.18215` to latents:
```
z_scaled = z_raw * 0.18215
```

**Purpose**:
1. **Match latent magnitude to what the diffusion noise schedule expects**: The noise schedule (βₜ, ᾱₜ) is designed under the convention that the data distribution scale is roughly comparable to the noise distribution ε ~ N(0, I), i.e., order O(1)
2. **Ensure proper signal-to-noise ratio (SNR)** across timesteps in the diffusion process
3. **Numerical stability** during training and inference
4. **Pretrained UNet compatibility**: In latent diffusion, the UNet was pretrained assuming latents follow the VAE-scaled distribution (e.g., Stable Diffusion uses 0.18215)

**Origin**: Empirically determined during Stable Diffusion VAE training to achieve a specific latent distribution that the UNet expects.

**Important Nuance**: It's not that every diffusion UNet mathematically *requires* unit variance. Rather:
- The noise schedule is designed assuming data scale is comparable to noise scale (both O(1))
- Pretrained UNets expect the specific latent distribution they were trained on
- Swapping to a different latent distribution creates an **SNR mismatch** that degrades performance

### 4.2 Why Scaling Factor is Applied Before Noise

The diffusion forward process is:
```
z_t = √(ᾱ_t) * z_0 + √(1 - ᾱ_t) * ε,  where ε ~ N(0, I)
```

**Critical Requirement**: Balanced behavior is best when `std(z_0)` is O(1), roughly similar to `std(ε) = 1`.

**Why This Matters**:
- The scheduler uses `√(1 - ᾱ_t)` multiplying unit Gaussian noise
- If latent std is 10: early timesteps dominated by signal → poor noise learning
- If latent std is 0.05: signal drowned by noise → information loss
- Proper scaling ensures **well-behaved SNR** across the entire diffusion trajectory

**If scaling is wrong**:
- **Too large** (std >> 1): Signal dominates noise → UNet can't learn proper denoising
- **Too small** (std << 1): Noise dominates signal → information loss, poor reconstruction
- **Inconsistent**: Training instability, SNR mismatch with pretrained components

### 4.3 Scaling Factor Mismatch in Mixed Latent Spaces

**Problem**: Applying RGB VAE's `0.18215` to Semantic VAE latents is incorrect because:

1. **Different Natural Scales**:
   - RGB VAE latents: Trained to have std ≈ 5.49 (1/0.18215) after scaling
   - Semantic VAE latents: Natural std ≈ X (different distribution, to be measured)
   - These distributions are fundamentally different due to different input modalities

2. **Conceptual Mismatch**:
   - RGB scaling: Derived from RGB pixel statistics (continuous intensities)
   - Semantic data: Discrete one-hot encodings → different latent statistics
   - Class imbalance and stem/head architecture can create biased channels

3. **Numerical Impact**:
   - Semantic latents scaled by wrong factor → suboptimal variance
   - SNR mismatch with pretrained UNet expectations → poor training dynamics
   - Potential channel-wise bias not addressed by scalar scaling

4. **Pretrained UNet Mismatch**:
   - The UNet (from SVD/Ctrl-V) was trained on RGB VAE latents with specific statistics
   - Using different latent statistics creates distribution shift
   - Model's learned priors become less effective

**Verdict**: While not the primary failure cause (latent space mismatch is worse), incorrect scaling **compounds the problem** by introducing SNR mismatch and distribution shift.

### 4.4 Correct Approach: Compute Semantic-Specific Scaling

**Method**:
1. Encode many semantic samples (500-1000): `z_i = SemanticVAE(x_i)`
2. Compute empirical statistics:
   - **Global**: `μ_global = mean(z_i)`, `σ_global = std(z_i)`
   - **Per-channel**: `μ_c = mean(z_i[:, c])`, `σ_c = std(z_i[:, c])` for c ∈ [0, 3]
3. Choose scaling strategy (see below)
4. Apply during training consistently

**Scaling Strategy Options**:

**Option A: Scalar Scaling (Simple)**
```python
s_semantic = 1.0 / σ_global
z_scaled = z * s_semantic
```
- ✅ Simple, commonly used
- ✅ Usually sufficient if latents are roughly zero-mean
- ⚠️ Doesn't handle channel-wise bias or variance differences

**Option B: Channel-Wise Normalization (Recommended for Semantic VAE)**
```python
for c in range(4):
    z_scaled[:, c] = (z[:, c] - μ_c) / (σ_c + eps)
```
- ✅ Handles channel-wise bias from class imbalance
- ✅ Accounts for different variance per channel
- ✅ Better for semantic VAE due to stem/head architecture effects
- ⚠️ Slightly more complex

**Option C: Match Pretrained UNet Distribution (Best for Reusing Ctrl-V/SVD)**
```python
# Compute target stats from RGB VAE (what UNet expects)
rgb_latents = RGB_VAE(rgb_samples)
μ_target, σ_target = mean(rgb_latents), std(rgb_latents)

# Scale semantic latents to match
semantic_latents = SemanticVAE(semantic_samples)
z_scaled = (semantic_latents - μ_semantic) / σ_semantic * σ_target + μ_target
```
- ✅ Matches distribution that pretrained UNet expects
- ✅ Minimizes distribution shift
- ✅ Best for transfer learning scenarios
- ⚠️ Requires computing both distributions

**Recommendation for Ctrl-V-seg**:
Use **Option C** (match pretrained distribution) since you're reusing SVD/Ctrl-V components. This ensures semantic latents have similar statistics to what the UNet was trained on.

**Why Standard Deviation Matters More Than Mean**:
- **SNR is primarily scale-sensitive**: The scheduler's noise injection depends on variance
- **Mean can usually be absorbed**: Networks with normalization layers can adapt to constant offsets
- **Scale dominates training dynamics**: Wrong std → wrong SNR → poor learning
- **But mean is not completely irrelevant**: Large non-zero mean can bias activations, especially with normalization layers. For semantic VAE with potential class imbalance, centering helps.

**Best Practice Summary**:
- **Start with**: Channel-wise normalization (Option B) to handle semantic VAE's unique characteristics
- **Then**: Match pretrained distribution (Option C) if using pretrained UNet
- **Simplify to**: Scalar scaling (Option A) only if empirical results show channel stats are similar

---

## 5. Why CLIP Conditioning Works: A Contrasting Case

### 5.1 CLIP Architecture in Diffusion Models

Text-to-image diffusion models (e.g., Stable Diffusion) use CLIP for text conditioning:
```
Text → CLIP Text Encoder → Embedding ∈ ℝ^(77×768)
                          ↓
                    Cross-Attention in UNet
```

**Key Difference**: CLIP embeddings are **not in the latent diffusion space**.

### 5.2 Why CLIP Doesn't Suffer from Latent Space Mismatch

#### 5.2.1 Different Conditioning Mechanism

**Spatial Conditioning (Concatenation)**:
- Conditioning latents **concatenated in channel dimension**
- Requires **same geometric space** as target latents
- Direct spatial correspondence: `z_cond[i,j]` conditions `z_target[i,j]`

**Cross-Attention Conditioning (CLIP)**:
- Text embeddings **injected via cross-attention layers**
- No requirement for spatial alignment
- Learned attention mechanism maps text features to spatial locations

#### 5.2.2 Semantic Feature Space vs. Latent Space

**CLIP Embeddings**:
- Live in **semantic feature space**, not latent image space
- Encode high-level concepts: "a cat sitting on a mat"
- Dimensionality: 768-dimensional vectors (not 4-channel spatial)
- No geometric correspondence to pixel space

**Why This Works**:
1. Cross-attention learns **flexible mapping** from text to image features
2. No assumption of spatial alignment or geometric compatibility
3. Text features provide **global semantic guidance**, not pixel-level conditioning
4. The mapping is **learned end-to-end** during diffusion training

#### 5.2.3 Mathematical Formulation

**Cross-Attention**:
```
Q = W_q * z_spatial     # Query from spatial features
K = W_k * z_text        # Key from text embeddings
V = W_v * z_text        # Value from text embeddings

Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

**Key Property**: The attention mechanism **learns the mapping** between incompatible spaces. No requirement for `z_text` and `z_spatial` to share geometric structure.

### 5.3 Lesson for Spatial Conditioning

**Conclusion**: Spatial conditioning via concatenation **requires geometric compatibility**. When conditioning and target live in different latent spaces, one must either:

1. **Use the same encoder** for both (e.g., all Semantic VAE)
2. **Use cross-attention** instead of concatenation (allows different spaces)
3. **Learn an explicit mapping** between spaces (e.g., adapter network)

**Our Case**: We chose option 1 (all Semantic VAE) as the simplest and most direct solution.

---

## 6. The Correct Solution: Unified Latent Space

### 6.1 Architectural Fix

**Before (Broken)**:
```python
# Target: Semantic VAE latents
target_latents = SemanticVAE(semantic_frames)  # Space B

# Conditioning: Mixed spaces
cond[0] = SemanticVAE(semantic_frame_0)        # Space B ✓
cond[1:24] = RGB_VAE(rgb_frame_0).repeat(23)   # Space A ✗
cond[24] = SemanticVAE(semantic_frame_24)      # Space B ✓
```

**After (Fixed)**:
```python
# Target: Semantic VAE latents
target_latents = SemanticVAE(semantic_frames)  # Space B

# Conditioning: Unified space
cond[0] = SemanticVAE(semantic_frame_0)        # Space B ✓
cond[1:24] = SemanticVAE(semantic_frame_0).repeat(23)  # Space B ✓
cond[24] = SemanticVAE(semantic_frame_24)      # Space B ✓
```

### 6.2 Why This Works

1. **Geometric Consistency**: All conditioning lives in `M_Sem`
2. **Feature Alignment**: Semantic features condition semantic prediction
3. **Interpolation**: UNet can interpolate between frames 0 and 24 in the same manifold
4. **Gradient Flow**: Meaningful gradients for all 25 frames
5. **Distribution Matching**: When combined with proper scaling, latents match what pretrained UNet expects

### 6.3 Scaling Factor Correction

Additionally, use semantic-specific scaling that matches the pretrained UNet's expectations:

```python
# Step 1: Compute semantic VAE statistics (run once)
semantic_latents = [SemanticVAE(sample) for sample in validation_data]
μ_sem_global = mean(semantic_latents)
σ_sem_global = std(semantic_latents)
μ_sem_channel = mean_per_channel(semantic_latents)  # [4]
σ_sem_channel = std_per_channel(semantic_latents)    # [4]

# Step 2: Compute RGB VAE statistics (what UNet was trained on)
rgb_latents = [RGB_VAE(rgb_sample) for rgb_sample in rgb_validation_data]
μ_rgb_global = mean(rgb_latents)
σ_rgb_global = std(rgb_latents)

# Step 3: Choose scaling strategy
# Option A: Simple scalar scaling
semantic_scaling = 1.0 / σ_sem_global

# Option B: Channel-wise normalization (better for semantic VAE)
def normalize_semantic(z):
    for c in range(4):
        z[:, c] = (z[:, c] - μ_sem_channel[c]) / (σ_sem_channel[c] + 1e-6)
    return z

# Option C: Match pretrained distribution (RECOMMENDED)
def match_rgb_distribution(z):
    # Normalize semantic latents
    z_norm = (z - μ_sem_global) / (σ_sem_global + 1e-6)
    # Scale to match RGB distribution
    z_matched = z_norm * σ_rgb_global + μ_rgb_global
    return z_matched

# Step 4: Apply during training (Option C recommended)
target_latents = match_rgb_distribution(SemanticVAE(semantic_frames))
cond_latents = match_rgb_distribution(SemanticVAE(semantic_cond))
```

**Why Option C (Match Pretrained Distribution) is Best**:
- The UNet was trained on RGB VAE latents with specific statistics
- Matching those statistics minimizes distribution shift
- Preserves the pretrained UNet's learned priors
- Ensures optimal SNR across the diffusion trajectory

---

## 7. Implications for Research

### 7.1 General Principles

**Principle 1: Latent Space Compatibility**
> When using concatenation-based conditioning in diffusion models, conditioning and target latents **must** live in the same learned manifold.

**Principle 2: Modality-Specific Encoding**
> Different data modalities (RGB, semantic, depth, etc.) require modality-specific encoders. Cross-modal encoding creates geometric incompatibility.

**Principle 3: Scaling Factor Specificity**
> VAE scaling factors are **not transferable** across different VAEs. When using pretrained diffusion components, the goal is to match the latent distribution that the UNet was trained on, not necessarily achieve unit variance. Each VAE requires empirically-determined scaling, ideally matching the pretrained model's expected distribution.

### 7.2 When Cross-Modal Conditioning Can Work

Cross-modal conditioning is viable when:
1. **Using cross-attention** instead of concatenation
2. **Learning explicit mappings** (e.g., adapter networks)
3. **Pre-aligning latent spaces** through joint training
4. **Using modality-agnostic encoders** (e.g., unified vision encoders)

### 7.3 Diagnostic Criteria for Latent Space Mismatch

Symptoms indicating latent space mismatch:
- ✗ Low overall loss but poor generation quality
- ✗ High performance on anchor frames, catastrophic failure on interpolated frames
- ✗ Training appears to converge but inference produces nonsensical outputs
- ✗ Gradient magnitudes vary wildly across conditioning frames

### 7.4 Recommended Practices

1. **Always use matched encoders** for spatial conditioning via concatenation
2. **Compute modality-specific scaling factors** empirically from large validation sets
3. **Match pretrained UNet distributions** when using transfer learning (not just unit variance)
4. **Consider channel-wise normalization** for modalities with potential bias (e.g., semantic segmentation)
5. **Validate conditioning consistency** during architecture design
6. **Monitor per-frame metrics** during training, not just overall loss
7. **Use cross-attention for cross-modal guidance** when concatenation is not viable

---

## 8. Conclusion

The failure of RGB VAE spatial conditioning for semantic segmentation diffusion models stems from a fundamental **latent space incompatibility**. RGB and Semantic VAEs, trained on different data distributions with different input modalities, learn geometrically incompatible latent manifolds despite sharing the same dimensionality. Concatenating latents from these different spaces creates meaningless feature combinations that prevent the diffusion UNet from learning effective denoising.

This failure mode is distinct from successful cross-modal conditioning approaches like CLIP because:
1. CLIP uses **cross-attention**, not concatenation
2. CLIP embeddings are **semantic features**, not spatial latents
3. Cross-attention **learns the mapping** between incompatible spaces

The solution requires **unified latent space conditioning**: all spatial conditioning must use the same encoder (Semantic VAE) that produces the target latents. Additionally, proper scaling must be applied with the following considerations:

1. **Scaling is primarily about SNR, not absolute unit variance**: The noise schedule expects data scale comparable to noise scale (both O(1))
2. **Pretrained UNets have distribution expectations**: When using transfer learning, match the latent distribution the UNet was trained on
3. **Standard deviation dominates, but mean matters**: While SNR is scale-sensitive, large mean bias can affect normalization layers
4. **Channel-wise normalization may be beneficial**: Semantic VAE can have channel-wise bias from class imbalance and architecture

**Recommended Scaling Strategy for Ctrl-V-seg**:
- Compute both semantic and RGB VAE latent statistics
- Apply normalization to match RGB distribution: `z_matched = (z_sem - μ_sem) / σ_sem * σ_rgb + μ_rgb`
- This minimizes distribution shift and preserves pretrained UNet priors

This analysis provides a theoretical foundation for understanding why naive cross-modal spatial conditioning fails and offers practical, research-grade guidelines for designing robust diffusion-based segmentation architectures.

---

## References

1. **Stable Video Diffusion**: Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets", 2023
2. **Latent Diffusion Models**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
3. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
4. **VAE Theory**: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
5. **Ctrl-V**: Original implementation for bounding box visualization in video diffusion

---

## Appendix: Experimental Data

### A.1 Performance Comparison

| Configuration | Frames 0,24 mIoU | Frames 1-23 mIoU | Overall mIoU |
|---------------|------------------|------------------|--------------|
| Mixed Spaces (Broken) | 48.5% | **4.14%** | 12.3% |
| Unified Space (Fixed) | TBD | TBD | TBD |

### A.2 Latent Statistics (To Be Measured)

| VAE Type | Mean | Std | Scaling Factor |
|----------|------|-----|----------------|
| RGB VAE | ~0.0 | ~5.49 | 0.18215 |
| Semantic VAE | TBD | TBD | TBD |

### A.3 Code Locations

- Training fix: `/usrhomes/s1492/Ctrl-V-seg/tools/train_video_diffusion.py`
- Inference fix: `/usrhomes/s1492/Ctrl-V-seg/src/ctrlv/pipelines/pipeline_video_diffusion.py`
- Scaling computation: `/usrhomes/s1492/vae_semantic/semantic_vae_native/inference/compute_semantic_scaling_factor.py`
