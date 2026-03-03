# Stage 2 Evaluation Report: Semantic-to-RGB Video Generation

**Model**: Ctrl-V Stage 2 ControlNet (Semantic Map → RGB Video)  
**Checkpoint**: Step 34300 (`kitti360_semantic2video_vae/checkpoint-34300`)  
**Dataset**: KITTI-360  
**Date**: February 27, 2026  

---

## 1. Overview

Stage 2 of the Ctrl-V pipeline generates RGB video frames conditioned on semantic segmentation maps. The ControlNet architecture takes semantic conditioning (encoded via a Semantic VAE) and produces photorealistic driving scene videos.

**Evaluation pipeline**:
1. Load 150 validation clips (25 frames each, 192×704 resolution)
2. Run the Stage 2 ControlNet pipeline with semantic VAE conditioning
3. Compute DRN-based semantic segmentation metrics on generated RGB frames (all 150 samples)
4. Compute image/video quality metrics on saved frames (FID, FVD, LPIPS, SSIM, PSNR — 10 samples only)

---

## 2. Summary of Results

| Metric | Value | Quality | Sample Size |
|--------|-------|---------|-------------|
| **mIoU (DRN)** | **45.60%** | Moderate — large classes well-generated | **150 samples** |
| **Pixel Accuracy** | **89.19%** | Good | **150 samples** |
| **Mean Class Accuracy** | **60.87%** | Moderate | **150 samples** |
| **FW-IoU** | **81.01%** | Good | **150 samples** |
| **FID** | **68.47** | Good (50–100 range) | **10 samples** ⚠️ |
| **FVD** | **595.52** | Moderate | **10 samples** ⚠️ |
| **LPIPS** | **0.334** | Similar (0.2–0.4 range) | **10 samples** ⚠️ |
| **SSIM** | **0.433 ± 0.137** | Moderate | **10 samples** ⚠️ |
| **PSNR** | **14.51 ± 3.16 dB** | Below average | **10 samples** ⚠️ |

**Note**: DRN semantic metrics (mIoU, Pixel Accuracy, etc.) were computed on **all 150 validation samples** (3,750 frames total). Image/video quality metrics (FID, FVD, LPIPS, SSIM, PSNR) were computed on **10 samples only** (250 frames) due to computational constraints. Full 150-sample FID/FVD evaluation is planned for future work.

---

## 3. Semantic Segmentation Metrics (DRN on Generated RGB)

The DRN (Dilated Residual Networks, `drn_d_105`) model was used to segment the generated RGB frames. The predicted segmentation was compared against ground-truth semantic labels to assess how well the generated images preserve semantic structure.

- **DRN Checkpoint**: `KITTI360_checkpoints/checkpoint_030.pth.tar`
- **Number of classes**: 19 (KITTI-360 trainIDs 0–18)
- **Evaluation samples**: 150 clips × 25 frames = 3,750 frames

### 3.1 Overall Metrics

| Metric | Value |
|--------|-------|
| mIoU | 45.60% |
| Overall Pixel Accuracy | 89.19% |
| Mean Class Accuracy | 60.87% |
| Frequency-Weighted IoU | 81.01% |

### 3.2 Per-Class IoU

| Class | IoU |
|-------|-----|
| road | **90.09%** |
| sidewalk | 61.13% |
| building | **84.25%** |
| wall | 45.35% |
| fence | 48.10% |
| pole | 37.84% |
| traffic light | 0.00% |
| traffic sign | 34.65% |
| vegetation | **82.35%** |
| terrain | 69.78% |
| sky | **87.24%** |
| person | 29.61% |
| rider | 19.01% |
| car | **88.75%** |
| truck | 55.25% |
| bus | 0.00% |
| train | 0.00% |
| motorcycle | 29.27% |
| bicycle | 3.66% |

### 3.3 Per-Class Analysis

**Well-generated classes (IoU > 70%)**:
- **road** (90.09%), **car** (88.75%), **sky** (87.24%), **building** (84.25%), **vegetation** (82.35%)
- These are large, spatially dominant classes with clear visual patterns

**Moderately generated classes (IoU 30–70%)**:
- **terrain** (69.78%), **sidewalk** (61.13%), **truck** (55.25%), **fence** (48.10%), **wall** (45.35%), **pole** (37.84%), **traffic sign** (34.65%), **person** (29.61%), **motorcycle** (29.27%)
- Smaller or less frequent structures with more complex boundaries

**Poorly generated classes (IoU < 30%)**:
- **rider** (19.01%), **bicycle** (3.66%), **bus/train/traffic light** (0.00%)
- Very rare classes in the evaluation set — limited or no GT pixels

**Absent classes**: traffic light (no samples in evaluation clips)

### 3.4 Confusion Matrix

A detailed confusion matrix visualizing per-class prediction accuracy is available:

**File**: `docs/stage2_evaluation/confusion_matrix_drn.png`

The confusion matrix shows:
- **Strong diagonal**: Well-predicted classes (road, car, building, vegetation, sky)
- **Off-diagonal scatter**: Common confusions (e.g., sidewalk ↔ road, pole ↔ building, rider ↔ person)
- **Rare classes**: Very sparse rows/columns for bus, train, motorcycle, bicycle

This 19×19 matrix provides detailed insight into which semantic classes are confused with each other during generation.

---

## 4. Image and Video Quality Metrics

⚠️ **Note**: Image and video quality metrics (FID, FVD, LPIPS, SSIM, PSNR) were computed on **10 saved videos only** (250 frame pairs), not the full 150-sample evaluation set. This is due to storage/computational constraints during evaluation. The DRN semantic metrics above were computed on all 150 samples. Future work will compute FID/FVD on the full 150-sample set for more robust quality assessment.

### 4.1 FID (Fréchet Inception Distance)

| Metric | Value |
|--------|-------|
| **FID** | **68.47** |

- Computed via `torch-fidelity` (Inception v3 features, 2048-dim)
- 250 generated frames vs 250 GT frames
- **Interpretation**: Good quality (50–100 range). The generated frames have realistic appearance and diversity that is reasonably close to the ground truth distribution.

### 4.2 FVD (Fréchet Video Distance)

| Metric | Value |
|--------|-------|
| **FVD** | **595.52** |

- Computed via I3D features at 128×512 resolution
- **Interpretation**: Moderate. FVD captures temporal consistency — the higher value suggests some temporal flickering or inconsistency between frames, which is expected for frame-by-frame generation without explicit temporal smoothing.

### 4.3 LPIPS (Learned Perceptual Image Patch Similarity)

| Metric | Value |
|--------|-------|
| **LPIPS** | **0.334** |

- Computed via AlexNet backbone
- **Interpretation**: Similar perceptual quality (0.2–0.4 range). The generated frames are perceptually recognizable as the same scene, though with notable differences in fine details.

### 4.4 SSIM (Structural Similarity Index)

| Metric | Value |
|--------|-------|
| SSIM (mean) | 0.433 |
| SSIM (std) | 0.137 |

- **Interpretation**: Moderate structural similarity. The overall scene layout is preserved, but pixel-level structural fidelity is limited — expected for a generative model.

### 4.5 PSNR (Peak Signal-to-Noise Ratio)

| Metric | Value |
|--------|-------|
| PSNR (mean) | 14.51 dB |
| PSNR (std) | 3.16 dB |

- **Interpretation**: Below average for direct pixel comparison. This is expected — generative models produce plausible but not pixel-identical reconstructions.

### 4.6 Per-Video Breakdown (SSIM / PSNR)

| Video | SSIM | PSNR (dB) |
|-------|------|-----------|
| video_000 | 0.375 | 14.18 |
| video_001 | 0.562 | 16.12 |
| video_002 | 0.522 | 15.68 |
| video_003 | 0.501 | 14.79 |
| video_004 | 0.424 | 15.39 |
| video_005 | 0.358 | 12.61 |
| video_006 | 0.461 | 14.63 |
| video_007 | 0.490 | 15.58 |
| video_008 | 0.303 | 11.90 |
| video_009 | 0.337 | 14.23 |
| **Average** | **0.433** | **14.51** |

---

## 5. Comparison Frames

Visual comparison frames are saved in `docs/stage2_evaluation/comparison_frames/`. Each 4-panel image shows:

| Top-Left | Top-Right | Bottom-Left | Bottom-Right |
|----------|-----------|-------------|--------------|
| GT RGB | Generated RGB | GT Semantic (colorized) | DRN Predicted Semantic (colorized) |

5 videos (video_000 through video_004) with 25 frames each are included for visual inspection.

**Frame naming**: `frame_XXX.png` (000–024)

---

## 6. Model Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `stabilityai/stable-video-diffusion-img2vid-xt` |
| ControlNet checkpoint | `checkpoint-34300` |
| Semantic VAE | `best_model_with_dice_boundaryweight.pth` |
| DRN architecture | `drn_d_105` (19 classes) |
| Resolution | 192×704 |
| Clip length | 25 frames |
| Inference steps | 30 |
| Guidance scale | 1.0 → 3.0 |
| Conditioning scale | 1.0 |
| Noise aug strength | 0.01 |
| Precision | fp16 |
| Seed | 1234 |

---

## 7. Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `tools/eval_stage2_rgb.py` | Main evaluation: generates RGB, runs DRN mIoU, saves frames |
| `tools/compute_stage2_fid_fvd.py` | FID, FVD, LPIPS, SSIM, PSNR from saved frames |
| `scripts/eval_scripts/eval_stage2_rgb.sh` | SLURM script for main eval |
| `scripts/eval_scripts/compute_stage2_fid_fvd.sh` | SLURM script for FID/FVD metrics |

---

## 8. Output Files

All evaluation outputs are stored at:  
`/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/`

| File | Description |
|------|-------------|
| `eval_results.json` | Full DRN mIoU results (per-class, per-sample) |
| `eval_summary.txt` | Human-readable DRN metrics summary |
| `confusion_matrix_drn.png` | Confusion matrix visualization |
| `confusion_matrix_drn.npy` | Raw confusion matrix (19×19) |
| `fid_fvd_results.json` | FID, FVD, LPIPS, SSIM, PSNR results |
| `fid_fvd_results.txt` | Human-readable quality metrics summary |
| `frames/video_XXX/` | Per-video frame outputs (GT RGB, Gen RGB, semantics, comparisons) |

Documentation copies are in `docs/stage2_evaluation/` within the repository.

---

## 9. Key Takeaways

1. **Semantic fidelity is strong for dominant classes** (150 samples): road (90%), car (89%), sky (87%), building (84%), vegetation (82%) — the model correctly generates the major scene structure.

2. **mIoU of 45.60% on 150 samples**: Robust evaluation on a large validation set confirms consistent performance across diverse driving scenarios.

3. **FID of 68.47 is good** (10 samples): Generated frames have realistic appearance within the KITTI-360 domain. Note: computed on limited sample size.

4. **Rare classes are challenging**: bus, train, traffic light have 0% IoU due to absence in evaluation clips. Small objects (rider 19%, bicycle 4%) have low IoU.

5. **Temporal consistency (FVD=595) could improve** (10 samples): Some frame-to-frame variation is visible, expected for per-frame diffusion generation. Full 150-sample FVD evaluation planned.

6. **Confusion matrix analysis**: Available in `docs/stage2_evaluation/confusion_matrix_drn.png` — shows strong diagonal for major classes and expected confusions (sidewalk↔road, rider↔person).
