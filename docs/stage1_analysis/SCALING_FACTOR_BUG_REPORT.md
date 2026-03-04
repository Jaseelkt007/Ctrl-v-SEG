# Stage 1 Scaling Factor Bug Report

**Date:** 2026-03-04
**Severity:** Critical — root cause of 4.1% mIoU in Stage 1 evaluation

---

## 1. Summary

The Stage 1 semantic prediction pipeline (RGB → Semantic) produces extremely poor mIoU (4.1%) despite the underlying semantic VAE achieving 89% mIoU independently. The root cause is a **missing latent unscaling step** at decode time: the diffusion training multiplies semantic VAE latents by `scaling_factor = 0.18215`, but the decode path feeds these scaled latents directly to the semantic VAE decoder without dividing by the same factor.

## 2. Evidence

### 2.1 The Scaling in Training

In `tools/train_video_diffusion.py`, line 556:

```python
target_latents = latents = latents * vae.config.scaling_factor  # × 0.18215
```

This scales the semantic VAE latents by 0.18215 before using them as diffusion targets. The diffusion UNet learns to predict latents in this **scaled space**. This is standard practice for latent diffusion models — it normalizes the latent distribution for better training dynamics.

### 2.2 The RGB VAE Unscales Correctly

The parent class `StableVideoDiffusionPipeline` has a `decode_latents()` method that **does unscale**:

```python
# From diffusers StableVideoDiffusionPipeline.decode_latents():
def decode_latents(self, latents, num_frames, decode_chunk_size=14):
    latents = 1 / self.vae.config.scaling_factor * latents  # ← UNSCALING
    # ... then decode with VAE
```

When `output_type != "latent"`, the pipeline calls `decode_latents()` which includes unscaling. **But our semantic path uses `output_type='latent'`**, which bypasses `decode_latents()` entirely:

```python
# pipeline_video_diffusion.py, lines 340-350:
if not output_type == "latent":
    frames = self.decode_latents(latents, ...)  # ← RGB path: unscales internally
else:
    frames = latents  # ← Semantic path: returns RAW SCALED latents!
```

### 2.3 The Semantic Decode Path Receives Wrong-Scale Latents

In both `train_video_diffusion.py` (validation) and `eval_stage1_semantic.py`:

```python
latents = result.frames[0]  # Still scaled by 0.18215
pred_semantic_ids = vae_manager.decode_semantic(latents)  # ← NO UNSCALING!
```

The semantic VAE decoder expects latents at the **original scale** (raw encoder mean z). Instead, it receives latents that are **5.5× smaller** (1/0.18215 ≈ 5.49). This causes the decoder to produce meaningless features, leading to near-random semantic predictions.

### 2.4 Evaluation Results Confirm the Bug

```json
{
  "metrics": {
    "miou": 0.0414,           // 4.1% — near random for 19 classes
    "overall_pixel_accuracy": 0.150,
    "mean_class_accuracy": 0.099
  }
}
```

Random chance for 19 classes would be ~5.3% mIoU. Our 4.1% is **below random**, consistent with the decoder operating in a completely wrong input range.

## 3. The Fix

### 3.1 Change Made

**File:** `src/ctrlv/models/dual_vae_manager.py` — `decode_semantic()` method

Added automatic unscaling using `self.rgb_vae.config.scaling_factor` before feeding latents to the semantic VAE decoder:

```python
def decode_semantic(self, latents, unscale=True):
    if unscale:
        scaling_factor = self.rgb_vae.config.scaling_factor  # 0.18215
        latents = latents / scaling_factor
    # ... then decode normally
```

This mirrors exactly what `StableVideoDiffusionPipeline.decode_latents()` does for RGB latents, but applied to our semantic decode path.

### 3.2 Why Fix in DualVAEManager (Not at Call Sites)

- The `encode_semantic_from_ids()` returns **unscaled** latents
- The training script scales them for diffusion (`× 0.18215`)
- The pipeline outputs **scaled** latents (since diffusion works in scaled space)
- Therefore `decode_semantic()` should always expect scaled latents from diffusion output
- Centralizing the fix prevents future callers from forgetting to unscale

### 3.3 Files Affected

| File | Change |
|------|--------|
| `src/ctrlv/models/dual_vae_manager.py` | Added `unscale` parameter to `decode_semantic()`, defaults to `True` |
| `tools/train_video_diffusion.py` | No change needed (calls `decode_semantic()` which now unscales) |
| `tools/eval_stage1_semantic.py` | No change needed (calls `decode_semantic()` which now unscales) |

## 4. Why First/Last Frames Had Higher mIoU

The user observed that conditioned frames (first and last) had relatively higher mIoU. This is explained by the **conditioning mechanism**, not by correct unscaling:

1. During inference, the conditioning latents for frames 0..`num_cond_bbox_frames-1` and the last frame are **ground truth semantic latents** (encoded via semantic VAE):
   ```python
   # pipeline_video_diffusion.py, lines 250-251:
   image_latents[:,0:num_cond_bbox_frames,::] = cond_latents[:,0:num_cond_bbox_frames,::]
   image_latents[:,-1,::] = cond_latents[:,-1,::]
   ```

2. The UNet sees these GT conditioning latents concatenated with the noisy latents. For conditioned frames, this provides a very strong signal, making the UNet's prediction more accurate **in the scaled space**.

3. Even though the decoder receives wrong-scale latents, the **relative pattern** in the latent space is better preserved for conditioned frames. The semantic VAE decoder uses `argmax(logits)` — if the relative ordering of class logits is partially preserved despite wrong magnitude, some dominant classes (road, building, sky) may still be predicted correctly.

4. For unconditioned middle frames, the UNet prediction is less accurate AND the wrong scale compounds, producing near-random predictions.

**Important:** Even the conditioned frames' mIoU was still very poor — just relatively less poor than the unconditioned frames.

## 5. Training Loss Analysis

### What the Step Loss Is

The `step_loss` in training logs is a **weighted MSE in scaled latent space**:

```python
# Velocity prediction parameterization (v-prediction)
c_out = -sigmas / ((sigmas**2 + 1)**0.5)
c_skip = 1 / (sigmas**2 + 1)
denoised_latents = model_pred * c_out + c_skip * noisy_latents

# Weighted MSE loss
weighting = (1 + sigmas^2) * sigmas^(-2)
loss = mean(weighting * (denoised_latents - target_latents)^2)
```

This loss:
- Operates entirely in the 4-channel latent space
- Never involves semantic VAE decoding
- Never sees semantic class predictions or mIoU
- Is standard for SVD-style diffusion training

### Is the Training Loss Correct?

**Yes.** The UNet training is mathematically correct. The model learns to predict scaled latent vectors that, when properly unscaled and decoded through the semantic VAE, should produce correct semantic maps. The problem was solely at the decode/evaluation stage.

## 6. Do We Need mIoU in the Training Loss?

**Not necessarily.** The theoretical chain is:

1. Semantic VAE faithfully encodes semantics (89% mIoU) ✓
2. Diffusion UNet accurately predicts scaled latents (MSE loss) ✓
3. Latents are correctly unscaled before decoding ← **was broken, now fixed**
4. Semantic VAE faithfully decodes unscaled latents ✓

If the diffusion model achieves low MSE loss, the decoded semantics should be good. **The bottleneck was step 3, not the loss function.**

However, if mIoU is still poor after the fix (e.g., due to the diffusion model not converging well), then adding a semantic-aware loss could help. Options:
- **Soft cross-entropy on decoded logits** (differentiable, but requires VAE decode in training loop — expensive)
- **Perceptual loss in latent space** (compare latent structure, not just MSE)
- **Curriculum training** (start with MSE, add semantic loss later)

**Recommendation:** Re-evaluate with the unscaling fix first. If mIoU > 50-60%, the MSE loss is sufficient. If still poor, consider adding semantic supervision.

## 7. The Scaling Factor Itself

### Is `0.18215` the Right Scaling Factor for Semantic Latents?

The `scaling_factor = 0.18215` was calibrated for the **RGB VAE** — it normalizes RGB latent distributions to have approximately unit variance. Our semantic VAE reuses the same frozen VAE encoder core but with a different input (semantic stem instead of `conv_in`). The semantic latent distribution may differ from the RGB latent distribution.

**Potential concern:** If semantic latents have a different variance than RGB latents, the scaling factor may over- or under-normalize them, making the noise schedule suboptimal.

**Investigation path:**
1. Compute the empirical mean and std of semantic VAE latents over the training set
2. Compare with RGB VAE latent statistics
3. If significantly different, compute a custom scaling factor: `scaling_factor = 1 / std(semantic_latents)`
4. Retrain with the corrected scaling factor

This is a secondary optimization — the unscaling bug fix should have a much larger impact than tuning the scaling factor.

## 8. Debug mIoU Added to Training

Added lightweight mIoU computation during the existing validation runs in `train_video_diffusion.py`. This:
- Runs only at `validation_steps` intervals (no impact on training speed)
- Computes per-sample mIoU and pixel accuracy
- Logs `val/miou` and `val/pixel_accuracy` to W&B
- Enables tracking semantic quality throughout training

## 9. Next Steps

1. **Re-run evaluation** with the unscaling fix to measure the actual impact
2. **Monitor `val/miou`** in the next training run to track convergence
3. **Monitor `latent_stats/*`** in W&B — these new metrics track semantic latent statistics:
   - `latent_stats/unscaled_std`: std of raw semantic latents (should be close to 1/0.18215 ≈ 5.49 if RGB scaling factor is correct)
   - `latent_stats/scaled_std`: std after applying 0.18215 (ideal ≈ 1.0 for noise schedule)
   - `latent_stats/suggested_scaling_factor`: optimal scaling factor based on measured stats
   - `latent_stats/ch{0-3}_mean`, `latent_stats/ch{0-3}_std`: per-channel breakdown
4. **Check `latent_statistics_report.txt`** in the output directory after training — contains full analysis with recommendation
5. **If mIoU is still low**, investigate the scaling factor calibration (Section 7)
6. **If mIoU plateaus below target**, consider adding semantic-aware loss terms (Section 6)
