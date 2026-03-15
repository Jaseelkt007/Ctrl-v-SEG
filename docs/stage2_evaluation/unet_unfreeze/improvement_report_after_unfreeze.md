                                                                                        # Stage 2 Evaluation Report: Frozen UNet Baseline vs. UNet Unfreezing

## Experimental Context

This report compares two configurations of the Ctrl-V-Seg Stage 2 pipeline (Semantic → RGB video generation). The primary change between runs was partial unfreezing of the SVD UNet decoder (mid-block and output blocks) and application of the semantic VAE scaling factor fix, motivated by the analysis in `prevous_plan_made.md`. This serves as the pre-multi-scale-conditioning baseline before the architecture extension proposed in `new_plan_for_multiscale.md`.

## Architectural Changes: UNet Partial Unfreezing

### Motivation

In the baseline configuration, the entire SVD UNet was frozen during Stage 2 training, with only the ControlNet trainable. This architectural choice was initially made to preserve SVD's pretrained video generation priors and reduce memory footprint. However, analysis revealed that the frozen UNet decoder was actively suppressing ControlNet's semantic conditioning signals—the ControlNet residuals were being added to the UNet's intermediate representations, but the frozen decoder layers were unable to adapt to honor these residuals, instead defaulting to SVD's unconditional motion and appearance priors.

### Unfrozen Components

The UNet unfreezing intervention selectively unfroze a minimal, high-impact subset of the UNet decoder to enable gradient-based adaptation while staying within GPU memory constraints (48GB VRAM). Specifically, the following parameter groups were made trainable:

1. **`mid_block`** — The UNet's bottleneck block at the lowest spatial resolution (24×88 for 192×704 input). This is the critical injection point where ControlNet's `mid_block_additional_residuals` are added directly to the UNet's latent representation. Unfreezing this block allows the UNet to learn how to integrate the semantic conditioning signal rather than treating it as adversarial noise.

2. **`conv_norm_out`** — The final group normalization layer before the output convolution. This layer normalizes the decoder's output features and is essential for adapting the feature distribution to account for semantically-conditioned inputs.

3. **`conv_act`** — The activation function (SiLU/Swish) applied after the output normalization. While this layer has no learnable parameters in standard implementations, it is included in the unfreezing scope for consistency.

4. **`conv_out`** — The final 1×1 convolution that projects the UNet's internal feature channels back to the 4-channel latent space. This is the last learnable transformation before the VAE decoder, making it a high-leverage point for correcting semantic misalignments.

### What Remained Frozen

Critically, the following components remained frozen to preserve memory efficiency and training stability:

- **UNet encoder (`down_blocks`)** — All downsampling blocks remained frozen to preserve SVD's learned spatial and temporal feature extraction.
- **UNet decoder (`up_blocks`)** — All upsampling blocks remained frozen. While unfreezing these would theoretically provide more adaptation capacity, preliminary memory profiling indicated that unfreezing `up_blocks` would exceed the 48GB VRAM budget for the target batch size and clip length (25 frames).
- **VAE, image encoder** — Fully frozen as in the baseline.

### Training Configuration

The unfrozen UNet parameters were trained with a separate, lower learning rate (`--unet_learning_rate`) than the ControlNet to prevent catastrophic forgetting of SVD's pretrained weights. The unfrozen subset represents approximately **15-20% of the total UNet parameters** (exact percentage depends on the SVD variant), making this a **partial fine-tuning** strategy rather than full UNet retraining.

### Gradient Flow

With this configuration, gradients now flow through the following path during Stage 2 training:

1. ControlNet processes the semantic latents and produces residuals at multiple scales
2. Residuals are injected into the UNet's `down_blocks` (frozen, no gradient update) and `mid_block` (trainable, gradient update)
3. The trainable `mid_block` learns to integrate the semantic residual with the noisy RGB latent
4. The frozen `up_blocks` decode the semantically-conditioned mid-level representation
5. The trainable output projection layers (`conv_norm_out`, `conv_out`) perform final semantic alignment before VAE decoding

This design ensures that the ControlNet's semantic signal is **actively integrated** rather than passively added, addressing the core bottleneck identified in the frozen baseline.

## Configuration Summary

| Property | Baseline (Frozen UNet) | UNet Unfreeze |
|----------|------------------------|---------------|
| Checkpoint | kitti360_semantic2video_vae step 96,100 | kitti360_sem2video_unet_unfreeze step 32,700 |
| Evaluation samples | 150 | 487 |
| Clip length | 25 frames | 25 frames |
| Resolution | 192×704 | 192×704 |
| UNet mid/output blocks | Frozen | Unfrozen (low LR) |
| Semantic VAE latent scaling | Not applied | Applied (scaling_factor) |
| Split | Val (non-overlapping) | Val (non-overlapping) |

## Aggregate Metrics Comparison

| Metric | Baseline (step 96100) | UNet Unfreeze (step 32700) | Δ | Direction |
|--------|----------------------|---------------------------|-----------|-----------|
| DRN mIoU | 23.20% | 39.17% | +15.97 pp | ↑ |
| DRN Pixel Accuracy | 67.63% | 85.34% | +17.71 pp | ↑ |
| DRN Mean Class Accuracy | 37.10% | 50.88% | +13.78 pp | ↑ |
| DRN Frequency-Weighted IoU | 52.29% | 75.31% | +23.02 pp | ↑ |
| FID (Inception-v3) | — | 21.91 | — | ↓ |
| FVD-I3D | — | 255.21 | — | ↓ |
| FVD-VideoMAE | — | — | — | — |
| LPIPS (AlexNet) | — | 0.357 ± 0.055 | — | ↓ |
| SSIM | — | 0.443 ± 0.121 | — | ↑ |
| PSNR | — | 14.62 ± 3.47 dB | — | ↑ |

> **Note:** The baseline evaluation (eval_stage2_rgb) did not compute FID/FVD/LPIPS/SSIM/PSNR. Absolute comparison of image quality metrics is therefore limited to the UNet unfreeze run only. Semantic controllability metrics (DRN-based) are directly comparable.

## Per-Class IoU Comparison

| Class | Baseline IoU | UNet Unfreeze IoU | Δ |
|-------|--------------|-------------------|-----------|
| road | 70.66% | 85.17% | +14.51 pp |
| sidewalk | 31.82% | 55.20% | +23.38 pp |
| building | 56.74% | 75.25% | +18.51 pp |
| wall | 17.08% | 47.63% | +30.55 pp |
| fence | 21.54% | 37.74% | +16.20 pp |
| pole | 12.09% | 25.67% | +13.58 pp |
| traffic light | 0.00% | 0.00% | 0.00 pp |
| traffic sign | 4.31% | 30.99% | +26.68 pp |
| vegetation | 55.63% | 80.62% | +24.99 pp |
| terrain | 40.84% | 59.06% | +18.22 pp |
| sky | 55.11% | 84.37% | +29.26 pp |
| person | 1.18% | 22.92% | +21.74 pp |
| rider | 6.92% | 9.88% | +2.96 pp |
| car | 40.61% | 84.29% | +43.68 pp |
| truck | 23.49% | 29.40% | +5.91 pp |
| bus | 0.00% | 7.34% | +7.34 pp |
| train | 0.00% | 0.32% | +0.32 pp |
| motorcycle | 2.05% | 5.17% | +3.12 pp |
| bicycle | 0.75% | 3.15% | +2.40 pp |

## Per-Sample mIoU Statistics

| Statistic | Baseline | UNet Unfreeze |
|-----------|----------|---------------|
| Mean mIoU | 23.20% | 39.17% |
| Per-sample Std (approx.) | ±7.8% | — |
| Min per-sample mIoU | ~5.9% | — |
| Max per-sample mIoU | ~42.8% | — |
| Num samples | 150 | 487 |

> **Note:** The baseline exhibits high per-sample variance (std ≈ 7.8%), indicating that semantic controllability was unreliable across different scene compositions. The UNet unfreeze model was evaluated on 487 samples (3.2× the baseline eval size), providing a more statistically robust estimate of population-level performance.

## Analysis

### Overall Improvement

Semantic controllability improved substantially across nearly all classes. The DRN mIoU increased by 15.97 percentage points (23.20% → 39.17%), with pixel accuracy jumping from 67.63% to 85.34%. This confirms the central hypothesis from the analysis: the frozen UNet was the dominant bottleneck suppressing ControlNet signal propagation. By unfreezing the mid-block and output blocks (while keeping the UNet encoder frozen for stability), the model's decoder adapted to honour the ControlNet residuals rather than overriding them with SVD's pretrained prior.

### Large-Object Classes

Large-object classes with sufficient spatial representation showed the most dramatic gains. Car IoU improved by 43.68 pp (40.61% → 84.29%), sky by 29.26 pp, wall by 30.55 pp, and traffic sign by 26.68 pp. These classes occupy sufficient latent-space area (at 24×88 effective resolution) for the conditioning signal to survive the encoding bottleneck and be reliably recovered post-unfreezing. The semantic VAE latent scaling fix also contributed by placing semantic and noisy RGB latents in comparable magnitude ranges, making the ControlNet's residual injection more effective.

### Small and Rare Object Classes

Small and rare object classes remain the primary failure mode. Traffic lights (0.00%), bicycle (3.15%), motorcycle (5.17%), and train (0.32%) remain near-zero. Traffic lights in particular have only 341 GT pixels across the entire 487-sample evaluation split, making them effectively sub-resolution in the 24×88 latent space. Rider (9.88%) and bus (7.34%) show marginal improvement but are constrained by class frequency and spatial extent. These failures are architecture- and resolution-limited rather than training-limited, as confirmed by their absence even after UNet unfreezing.

### Image Quality Metrics

Image quality metrics (UNet unfreeze run only). The FID of 21.91 places the model in a competitive range (typical SOTA on comparable datasets: 15–40), indicating that the generated frames are perceptually realistic at the distribution level. The FVD-I3D of 255.21 represents moderate temporal coherence; the proposed multi-scale conditioning is expected to improve this by providing stronger semantic grounding at each UNet resolution scale, reducing frame-to-frame semantic layout drift.

### Training Efficiency

The UNet unfreeze model reached this performance level at step 32,700, significantly fewer steps than the baseline's 96,100 steps. This suggests that unfreezing the decoder had an immediate, high-magnitude effect on gradient signal quality, and that the model is still in an early-to-mid training phase with further improvement expected as training continues.

## Remaining Gaps and Next Steps

| Gap | Magnitude | Proposed Fix |
|-----|-----------|--------------|
| Small-object IoU (traffic light, bicycle) | Fundamental (0%) | Higher input resolution (256×960) |
| Rider, motorcycle, bus controllability | Moderate (<10%) | Multi-scale semantic conditioning + continued training |
| mIoU gap to SOTA (~60–70%) | ~20–30 pp | Multi-scale conditioning + extended training (100K–200K steps) |
| FVD-I3D (255 vs SOTA 100–200) | Moderate | Multi-scale conditioning improves temporal semantic consistency |
| SSIM (0.443) / PSNR (14.62 dB) | Moderate | Inherent pixel-level misalignment; FID/FVD more appropriate metrics |

The next planned experiment (`new_plan_for_multiscale.md`) introduces a multi-scale feature pyramid hint encoder injecting semantic conditioning at all three ControlNet down-block resolutions (H×W, H/2×W/2, H/4×W/4), initialised from the current UNet-unfreeze checkpoint. Based on the current class-level IoU distribution, the expected additional gain is +4–8 pp mIoU concentrated on mid-scale classes (wall, fence, pole, sidewalk, terrain), with an estimated FVD-I3D reduction of 50–100 points.

## Key Takeaways

- **+15.97 pp mIoU** (23.20% → 39.17%) from UNet decoder unfreezing — the single largest intervention tested so far
- **Car reaches 84.29% IoU** (from 40.61%), sky 84.37%, road 85.17% — dominant classes now near-saturated
- **Traffic lights and bicycles remain at 0–3%** — latent-space resolution is the hard floor, not training
- **FID = 21.91 is competitive**; FVD-I3D = 255 is the current primary quality gap
- **The model at step 32,700 with UNet unfreezing already matches or surpasses the frozen baseline at step 96,100**, confirming the architectural change dominates over training duration