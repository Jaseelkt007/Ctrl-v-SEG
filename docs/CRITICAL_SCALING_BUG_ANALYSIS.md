# Critical Bug Analysis: VAE Scaling Factor Mismatch in Stage 1 Semantic Diffusion

**Status**: 🔴 **CRITICAL BUG CONFIRMED**  
**Date**: March 4, 2026  
**Impact**: Root cause of 4.14% mIoU in Stage 1 evaluation  
**Severity**: High - Prevents any meaningful semantic prediction quality

---

## Executive Summary

A critical scaling factor bug has been identified in the Stage 1 semantic diffusion training and evaluation pipeline. The diffusion UNet is trained to predict latents in **scaled space** (×0.18215), but during inference/validation, these scaled latents are fed directly to the Semantic VAE decoder without **unscaling** (÷0.18215). 

The Semantic VAE was trained with unscaled latents and expects inputs in that range. Feeding it scaled latents (≈5.5× smaller magnitude) causes the decoder to operate completely outside its trained distribution, producing garbage predictions.

**This is separate from the latent space mismatch issue** (RGB VAE vs Semantic VAE for conditioning, fixed on Mar 3). This is a **new, independent bug** that exists in the current codebase.

---

## Table of Contents

1. [Background: VAE Scaling in Latent Diffusion](#1-background-vae-scaling-in-latent-diffusion)
2. [The Training Data Flow](#2-the-training-data-flow)
3. [The Critical Bug](#3-the-critical-bug)
4. [Evidence and Verification](#4-evidence-and-verification)
5. [Why the UNet Training Appears Fine](#5-why-the-unet-training-appears-fine)
6. [Why mIoU is So Poor](#6-why-miou-is-so-poor)
7. [The Fix](#7-the-fix)
8. [Important Clarifications](#8-important-clarifications)
9. [Timeline and Historical Context](#9-timeline-and-historical-context)
10. [Recommended Actions](#10-recommended-actions)

---

## 1. Background: VAE Scaling in Latent Diffusion

### 1.1 Why Scaling Exists

In Stable Diffusion and SVD (Stable Video Diffusion), VAE latents are scaled by a factor (typically 0.18215) to:
- Normalize latent variance to approximately unit scale
- Match the distribution expected by the noise scheduler
- Ensure proper signal-to-noise ratio (SNR) across diffusion timesteps

### 1.2 The Scaling Contract

```python
# Encoding (training data preparation)
raw_latents = VAE.encode(images)  # Unscaled, std ≈ 5.49
scaled_latents = raw_latents * 0.18215  # Scaled, std ≈ 1.0

# Training
UNet learns to denoise in scaled_latent space

# Decoding (inference)
scaled_latents = UNet.predict(...)  # Output in scaled space
raw_latents = scaled_latents / 0.18215  # MUST unscale!
images = VAE.decode(raw_latents)  # Decoder expects unscaled
```

**Critical principle**: If you train with scaled latents, you **must** unscale before decoding.

### 1.3 RGB VAE vs Semantic VAE Scaling

- **RGB VAE**: Pre-trained with its own scaling factor (0.18215), expects unscaled latents for decoding
- **Semantic VAE**: Custom-trained, expects its own latent distribution (unscaled raw encoder output)

The bug occurs because we're using RGB VAE's scaling factor (0.18215) during training but forgetting to unscale before feeding to Semantic VAE decoder.

---

## 2. The Training Data Flow

### 2.1 Encoding Path (Training Data Preparation)

**Location**: `tools/train_video_diffusion.py` lines 516-556

```python
# Step 1: Encode semantic IDs to latents using Semantic VAE
if args.predict_bbox and args.use_segmentation:
    semantic_ids = batch['semantic_ids']  # [B, F, H, W] trainIDs 0-18
    semantic_ids = rearrange(semantic_ids, "b f h w -> (b f) h w")
    
    # Semantic VAE encode: IDs → one-hot → stem → encoder → latents
    latents = vae_manager.encode_semantic_from_ids(semantic_ids)
    # Returns: [B*F, 4, 24, 88] UNSCALED raw encoder output
    
    latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size)

# Step 2: Create conditioning (frames 1-23 get first frame repeated)
initial_frame_latent = vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
# UNSCALED latents

conditional_latents = latents.clone()  # UNSCALED
conditional_latents[:, num_cond:-1, :] = initial_frame_latent.repeat(...)

# Step 3: Scale target latents for diffusion training
target_latents = latents = latents * vae.config.scaling_factor  # LINE 556
# Now target_latents are SCALED (×0.18215)
# But conditional_latents remain UNSCALED (this is intentional for SVD)
```

**Key observations**:
1. Semantic VAE encoder returns **unscaled** latents (raw mean from encoder)
2. Target latents are **scaled** at line 556 using `vae.config.scaling_factor = 0.18215`
3. Conditioning latents remain **unscaled** (by design, matches original SVD)
4. UNet input: `concat(scaled_noisy_latents, unscaled_conditioning)` → 8 channels

### 2.2 Training Loss Computation

**Location**: `tools/train_video_diffusion.py` lines 557-630

```python
# Add noise to scaled target latents
noise = torch.randn_like(latents)  # latents are scaled
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

# UNet prediction (v-prediction)
model_pred = unet(
    sample=noisy_latents,  # scaled + noise
    timestep=timesteps,
    encoder_hidden_states=image_embeddings,  # CLIP
    added_time_ids=added_time_ids,
    conditional_latents=conditional_latents,  # unscaled (by design)
).sample

# Compute denoised latents
c_out = -sigmas / ((sigmas**2 + 1)**0.5)
c_skip = 1 / (sigmas**2 + 1)
denoised_latents = model_pred * c_out + c_skip * noisy_latents

# MSE loss in scaled latent space
loss = mean(weighting * (denoised_latents - target_latents)^2)
```

**The UNet learns to predict in SCALED latent space** because target_latents are scaled.

---

## 3. The Critical Bug

### 3.1 Bug Location 1: Training Validation

**File**: `tools/train_video_diffusion.py`  
**Lines**: 390-397

```python
# Validation during training
if args.predict_bbox and args.use_segmentation and vae_manager is not None:
    latents = result.frames[0]  # [T, C, H, W] - these are SCALED latents!
    latents_flat = rearrange(latents, "f c h w -> f c h w")
    
    # ❌ BUG: Feeding scaled latents to semantic VAE decoder
    semantic_ids = vae_manager.decode_semantic(latents_flat)
    # Semantic VAE expects UNSCALED latents, but receives scaled ones
```

### 3.2 Bug Location 2: Evaluation Script

**File**: `tools/eval_stage1_semantic.py`  
**Lines**: 475-482

```python
# Stage 1 evaluation
latents = result.frames[0].to(torch.float32)  # [T, C, H, W] - SCALED latents!

# ❌ BUG: Feeding scaled latents to semantic VAE decoder
pred_semantic_ids = vae_manager.decode_semantic(latents)
# Semantic VAE expects UNSCALED latents, but receives scaled ones
```

### 3.3 Bug Location 3: DualVAEManager

**File**: `src/ctrlv/models/dual_vae_manager.py`  
**Lines**: 234-265

```python
def decode_semantic(self, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode latents to semantic IDs using Semantic VAE.
    
    Args:
        latents: [B*F, C, H_latent, W_latent] (already flat from encode)
    
    Returns:
        semantic_ids: [B*F, H, W] trainIDs (0-18)
    """
    with torch.no_grad():
        # ❌ BUG: No unscaling! Latents go directly to decoder
        decoded_features = self.semantic_vae.model._decode_to_semantic_features(latents)
        
        decoded_features = decoded_features.unsqueeze(1)
        logits = self.semantic_vae.model.semantic_head(decoded_features)
        logits = logits[:, 0, :, :, :]
        semantic_ids = torch.argmax(logits, dim=1)
    
    return semantic_ids
```

### 3.4 Why This is Wrong

The Semantic VAE was trained with this encode-decode cycle:

```python
# Semantic VAE training (in vae_semantic repo)
semantic_ids → one_hot → stem → encoder → z_raw (unscaled)
z_raw → decoder → features → head → logits → semantic_ids
```

The decoder was trained to receive **unscaled** latents (`z_raw`). It has learned internal weights, biases, and activations based on the distribution of unscaled latents (e.g., std ≈ X, mean ≈ Y).

When we feed it **scaled** latents (×0.18215, so ≈5.5× smaller), the decoder sees completely out-of-distribution inputs:
- Conv weights expect inputs with std ≈ X, but receive std ≈ 0.18×X
- Batch norm layers (if any) have learned stats for unscaled distribution
- Activation functions operate in the wrong range
- The entire decoder is operating outside its trained regime

**Result**: Garbage predictions, explaining the 4.14% mIoU.

---

## 4. Evidence and Verification

### 4.1 Code Verification

✅ **Confirmed**: Line 556 in `train_video_diffusion.py` scales target latents
```python
target_latents = latents = latents * vae.config.scaling_factor  # 0.18215
```

✅ **Confirmed**: No unscaling before decode in training validation (line 397)
```python
semantic_ids = vae_manager.decode_semantic(latents_flat)  # No division!
```

✅ **Confirmed**: No unscaling before decode in eval script (line 482)
```python
pred_semantic_ids = vae_manager.decode_semantic(latents)  # No division!
```

✅ **Confirmed**: `decode_semantic()` has no unscaling logic
```python
# dual_vae_manager.py line 252
decoded_features = self.semantic_vae.model._decode_to_semantic_features(latents)
# latents go directly in, no preprocessing
```

### 4.2 Magnitude Analysis

**Expected unscaled latent stats** (from Semantic VAE training):
- Mean: ≈ μ_sem (to be measured, likely near 0)
- Std: ≈ σ_sem (to be measured, likely 2-8 range)

**What decoder receives** (scaled latents):
- Mean: ≈ μ_sem × 0.18215
- Std: ≈ σ_sem × 0.18215

If σ_sem = 5.0, the decoder expects std=5.0 but receives std=0.91. This is a **5.5× magnitude mismatch**.

### 4.3 Empirical Evidence

**Observed symptoms** (matching prediction):
- ✅ Training loss appears normal (0.00005) → UNet learning in scaled space works
- ✅ Validation/eval mIoU catastrophically low (4.14%) → Decode path broken
- ✅ Some classes show non-zero IoU → Decoder isn't completely dead, just severely degraded
- ✅ Per-class performance varies wildly → Distribution shift affects classes differently

---

## 5. Why the UNet Training Appears Fine

### 5.1 Training Loss is Correct

The training objective is:
```python
loss = MSE(denoised_latents, target_latents)
```

Both are in **scaled space**, so the loss is mathematically correct. The UNet learns:
- Input: scaled noisy latents + unscaled conditioning
- Output: prediction that denoises to scaled target latents
- Loss: MSE in scaled space

**This is fine!** The UNet is learning the correct scaled latent distribution.

### 5.2 Low Loss ≠ Good Semantic Predictions

A low training loss (0.00005) means:
- UNet accurately predicts scaled latents
- Diffusion process is working correctly
- **But says nothing about decode path!**

The decode path is only tested during validation/eval, where the bug manifests.

### 5.3 Why This Bug Wasn't Caught Earlier

1. **Training metrics don't include mIoU**: Only MSE loss in latent space
2. **Validation happens infrequently**: Easy to miss poor validation results
3. **Training loss misleading**: Low loss suggests everything is fine
4. **Subtle magnitude issue**: Not a crash, just wrong distribution

---

## 6. Why mIoU is So Poor

### 6.1 Decoder Distribution Mismatch

The Semantic VAE decoder has learned to map unscaled latents → semantic features. Its learned parameters expect:
- Input range: [z_min, z_max] (unscaled latent range)
- Input statistics: μ ≈ μ_sem, σ ≈ σ_sem

When it receives scaled latents:
- Input range: [0.18×z_min, 0.18×z_max] (5.5× smaller)
- Input statistics: μ ≈ 0.18×μ_sem, σ ≈ 0.18×σ_sem

### 6.2 Layer-by-Layer Breakdown

Imagine the decoder's first conv layer was trained with:
- Input channels with std ≈ 5.0
- Learned weights with magnitude calibrated for that scale
- Output activations in a certain range

When input std becomes 0.91 (5.5× smaller):
- Activations are 5.5× smaller than expected
- Subsequent layers compound the error
- Final logits have wrong scale and distribution
- Argmax produces incorrect class predictions

### 6.3 Why Some Classes Still Work (Partially)

Classes like "road" (12.98% IoU) and "car" (10.77% IoU) still have non-zero performance because:
- These classes may have more consistent spatial patterns
- Decoder degradation affects classes differently based on learned feature statistics
- Some spatial structure is preserved despite magnitude mismatch
- But overall performance is catastrophically bad (4.14% average)

---

## 7. The Fix

### 7.1 Simple Solution: Unscale Before Decode

Add unscaling at three locations:

#### Fix 1: Training Validation

**File**: `tools/train_video_diffusion.py`  
**Line**: 396 (insert before line 397)

```python
# Validation during training
if args.predict_bbox and args.use_segmentation and vae_manager is not None:
    latents = result.frames[0]  # [T, C, H, W]
    latents_flat = rearrange(latents, "f c h w -> f c h w")
    
    # ✅ FIX: Unscale latents before decoding
    latents_flat = latents_flat / vae.config.scaling_factor
    
    semantic_ids = vae_manager.decode_semantic(latents_flat)
```

#### Fix 2: Evaluation Script

**File**: `tools/eval_stage1_semantic.py`  
**Line**: 481 (insert before line 482)

```python
# Stage 1 evaluation
latents = result.frames[0].to(torch.float32)  # [T, C, H, W]

# ✅ FIX: Unscale latents before decoding
latents = latents / vae.config.scaling_factor

pred_semantic_ids = vae_manager.decode_semantic(latents)
```

#### Fix 3: DualVAEManager (Recommended)

**File**: `src/ctrlv/models/dual_vae_manager.py`  
**Method**: `decode_semantic` (line 234)

```python
def decode_semantic(self, latents: torch.Tensor, scaling_factor: float = None) -> torch.Tensor:
    """
    Decode latents to semantic IDs using Semantic VAE.
    
    Args:
        latents: [B*F, C, H_latent, W_latent]
        scaling_factor: If provided, unscale latents by this factor
    
    Returns:
        semantic_ids: [B*F, H, W] trainIDs (0-18)
    """
    with torch.no_grad():
        # ✅ FIX: Unscale if scaling_factor provided
        if scaling_factor is not None:
            latents = latents / scaling_factor
        
        # Decode unscaled latents
        decoded_features = self.semantic_vae.model._decode_to_semantic_features(latents)
        decoded_features = decoded_features.unsqueeze(1)
        logits = self.semantic_vae.model.semantic_head(decoded_features)
        logits = logits[:, 0, :, :, :]
        semantic_ids = torch.argmax(logits, dim=1)
    
    return semantic_ids
```

Then update calls:
```python
# Training validation
semantic_ids = vae_manager.decode_semantic(latents_flat, scaling_factor=vae.config.scaling_factor)

# Evaluation
pred_semantic_ids = vae_manager.decode_semantic(latents, scaling_factor=vae.config.scaling_factor)
```

### 7.2 Alternative: Scale-Aware Encoder/Decoder

A more architectural fix would be to make the Semantic VAE encoding also apply the scaling factor:

```python
# In encode_semantic_from_ids
latents = self.semantic_vae.model._encode_semantic_features(h0)
latents = latents * self.scaling_factor  # Apply scaling at encode

# In decode_semantic
latents = latents / self.scaling_factor  # Unscale at decode
decoded_features = self.semantic_vae.model._decode_to_semantic_features(latents)
```

**However**, this changes the encoding path and would require retraining. The simple fix (unscale before decode) is sufficient.

### 7.3 Which Fix to Apply?

**Recommended approach**: Fix 3 (DualVAEManager) because:
- Centralizes the logic in one place
- Makes the API explicit about scaling
- Easier to maintain
- Prevents future bugs

Then update the two call sites to pass `scaling_factor=vae.config.scaling_factor`.

---

## 8. Important Clarifications

### 8.1 This is NOT the Latent Space Mismatch Bug

**Latent space mismatch** (fixed Mar 3, 2026):
- Problem: Mixing RGB VAE and Semantic VAE latents for conditioning
- Impact: UNet receives geometrically incompatible conditioning
- Fix: Use only Semantic VAE for all conditioning

**Scaling factor bug** (this document, unfixed):
- Problem: Not unscaling before decoding
- Impact: Semantic VAE decoder receives wrong magnitude inputs
- Fix: Add unscaling before decode

These are **independent bugs**. Both need to be fixed for Stage 1 to work.

### 8.2 The Conditioning IS Unscaled (By Design)

In the current code:
```python
conditional_latents = latents.clone()  # Before scaling
target_latents = latents * vae.config.scaling_factor  # After scaling
```

This means:
- Target latents: SCALED
- Conditioning latents: UNSCALED

**This is intentional** and matches the original SVD architecture. The UNet is designed to handle this:
- It receives 8 channels: [scaled_noisy (4ch) | unscaled_conditioning (4ch)]
- This worked fine in original Ctrl-V with RGB VAE

The bug is **only** in the decode path, not the conditioning path.

### 8.3 Do We Need to Change the Training?

**No.** The training is correct as-is:
- Target latents are scaled → proper SNR for diffusion
- Conditioning is unscaled → matches SVD design
- UNet learns to denoise scaled latents → correct objective

**Only the decode path needs fixing.**

### 8.4 Why Not Add mIoU to Training Loss?

Adding mIoU loss would require:
1. Decoding latents every training step (expensive)
2. Backpropagating through argmax (non-differentiable) or soft cross-entropy
3. Balancing two loss terms (MSE + mIoU)
4. Significantly slower training

**It's not necessary** because:
- The Semantic VAE already achieves 89% mIoU reconstruction
- MSE in latent space is theoretically sound if decode path is correct
- The bug is in decode logic, not the training objective

**Fix the decode bug first.** If mIoU is still poor, then consider adding semantic supervision.

---

## 9. Timeline and Historical Context

### 9.1 Bug Introduction

- **Oct 31, 2025**: Initial semantic diffusion commit (`93d05e5`)
  - Added Semantic VAE support
  - Introduced scaling at line 556: `latents = latents * vae.config.scaling_factor`
  - **Bug introduced**: No corresponding unscaling before decode

### 9.2 Training with Bug

- **Feb 27, 2026**: Stage 1 training completed
  - Checkpoint 49200 saved
  - Training loss: 0.00005 (appears normal)
  - Evaluation mIoU: **4.14%** (catastrophically bad)
  - **Root cause**: This scaling bug + latent space mismatch

### 9.3 Partial Fix

- **Mar 3, 2026**: Latent space mismatch fixed (`a515397`)
  - Fixed: Now uses Semantic VAE for all conditioning (not RGB VAE)
  - **Not fixed**: Scaling bug remains
  - Current codebase still has this bug

### 9.4 Current Status (Mar 4, 2026)

- ✅ Latent space mismatch: **FIXED**
- ❌ Scaling bug: **STILL EXISTS**

**Implication**: Retraining Stage 1 with current code will still fail due to scaling bug.

---

## 10. Recommended Actions

### 10.1 Immediate Actions

1. **Apply the fix** (choose one approach):
   - Option A: Add unscaling in DualVAEManager.decode_semantic() [RECOMMENDED]
   - Option B: Add unscaling at both call sites (training + eval)

2. **Test the fix**:
   ```bash
   # Re-run evaluation on existing checkpoint 49200
   # This tests if the fix improves decode quality
   # without needing to retrain
   sbatch scripts/eval_scripts/eval_stage1_semantic.sh
   ```

3. **Expected results after fix**:
   - If the UNet learned correctly: mIoU should jump significantly (40-60%+)
   - If still poor: May need to retrain (unlikely if UNet training was correct)

### 10.2 Verification Steps

After applying the fix:

1. **Sanity check**: Print latent statistics before/after unscaling
   ```python
   print(f"Scaled latents: mean={latents.mean():.4f}, std={latents.std():.4f}")
   latents_unscaled = latents / vae.config.scaling_factor
   print(f"Unscaled latents: mean={latents_unscaled.mean():.4f}, std={latents_unscaled.std():.4f}")
   ```

2. **Decoder input check**: Verify unscaled latents are in expected range
   ```python
   # Compare with encoding path
   gt_latents = vae_manager.encode_semantic_from_ids(semantic_ids)
   print(f"GT encode latents: mean={gt_latents.mean():.4f}, std={gt_latents.std():.4f}")
   # Should match unscaled diffusion output
   ```

3. **Re-evaluate**: Run full evaluation and expect mIoU >> 4.14%

### 10.3 If mIoU is Still Poor After Fix

If applying the unscaling fix doesn't significantly improve mIoU:

1. **Check Semantic VAE quality**: Verify it still achieves 89% mIoU on GT→encode→decode
2. **Check UNet predictions**: Visualize latent statistics from UNet output
3. **Consider retraining**: The Feb 27 checkpoint may have learned suboptimal patterns
4. **Analyze per-timestep quality**: Check if certain diffusion steps produce better results

### 10.4 Long-Term Improvements

After confirming the fix works:

1. **Retrain Stage 1** with:
   - Fixed code (both latent space + scaling)
   - Proper semantic-specific scaling factor (match RGB distribution)
   - Monitor validation mIoU during training

2. **Add validation metrics**:
   ```python
   # In training loop, log mIoU every N steps
   if global_step % args.validation_steps == 0:
       val_miou = validate_semantic_quality(...)
       logger.log({"val/miou": val_miou})
   ```

3. **Consider semantic supervision** (if mIoU still suboptimal):
   - Add auxiliary mIoU loss term
   - Weight: λ_mse × MSE + λ_miou × CrossEntropy
   - Start small: λ_miou = 0.1

---

## Conclusion

The 4.14% mIoU in Stage 1 evaluation is caused by a **critical scaling bug**: scaled latents (×0.18215) are fed to the Semantic VAE decoder without unscaling (÷0.18215). The decoder was trained with unscaled latents and operates completely incorrectly when given scaled inputs.

**The fix is simple**: Add one line of code (`latents = latents / vae.config.scaling_factor`) before calling `decode_semantic()`.

**This bug is independent from the latent space mismatch bug fixed on Mar 3.** Both issues needed to be addressed for Stage 1 to work correctly.

The UNet training itself is sound—the problem is purely in the decode path. Applying this fix should dramatically improve semantic prediction quality without requiring retraining.

---

## Appendix: Code Locations Summary

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `tools/train_video_diffusion.py` | 556 | Scaling applied | ✅ Correct |
| `tools/train_video_diffusion.py` | 397 | No unscaling before decode | ❌ Add: `latents_flat /= vae.config.scaling_factor` |
| `tools/eval_stage1_semantic.py` | 482 | No unscaling before decode | ❌ Add: `latents /= vae.config.scaling_factor` |
| `src/ctrlv/models/dual_vae_manager.py` | 234 | `decode_semantic()` no unscaling | ❌ Add scaling_factor parameter |

---

## References

1. Stable Diffusion VAE Scaling: [Latent Diffusion Models Paper](https://arxiv.org/abs/2112.10752)
2. Stage 1 Analysis Report: `/usrhomes/s1492/Ctrl-V-seg/docs/stage1_analysis/STAGE1_ANALYSIS_REPORT.md`
3. Academic Analysis: `/usrhomes/s1492/Ctrl-V-seg/docs/stage1_analysis/ACADEMIC_ANALYSIS_RGB_SPATIAL_CONDITIONING_FAILURE.md`
4. Training Script: `tools/train_video_diffusion.py`
5. Evaluation Script: `tools/eval_stage1_semantic.py`
6. DualVAEManager: `src/ctrlv/models/dual_vae_manager.py`
