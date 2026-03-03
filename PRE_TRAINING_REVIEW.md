# Pre-Training Review & Issues

## Test Training Results ✅

**Job ID**: 196504  
**Status**: COMPLETED successfully  
**Completion Time**: Mon Feb 16 06:15:32 PM CET 2026  

### ✅ Successful Initialization
```
02/16/2026 18:14:25 - Initializing DualVAEManager with semantic VAE
02/16/2026 18:14:31 - ✓ DualVAEManager initialized for semantic VAE encoding
```

**Verdict**: Test training initialized correctly with Semantic VAE.

---

## 🔴 CRITICAL ISSUES FOUND

### Issue 1: **Stage 1 Script - Wrong `data_root` Argument** ❌

**File**: `scripts/train_scripts/train_kitti360_bbox_predict.sh`  
**Line 136**: 
```bash
--data_root $DATASET_PATH \
```

**Problem**: 
- `DATASET_PATH="/no_backups/s1492/"` (line 73)
- But `KITTI360OfficialDataset` **does NOT use `data_root`**
- It uses official KITTI-360 paths internally

**Evidence from test script** (which works):
```bash
# Line 87 in test_short_training.sh
--data_root "" \  # Empty string - correct!
```

**Fix**: Change line 136 to:
```bash
--data_root "" \
```

And remove/comment line 73:
```bash
# DATASET_PATH="/no_backups/s1492/"  # Not needed for KITTI360OfficialDataset
```

---

### Issue 2: **Stage 2 Script - Same `data_root` Issue** ❌

**File**: `scripts/train_scripts/train_kitti360_sem2video.sh`  
**Line 141**:
```bash
--data_root $DATASET_PATH \
```

**Problem**: Same as Stage 1

**Fix**: Change line 141 to:
```bash
--data_root "" \
```

And remove/comment line 67:
```bash
# DATASET_PATH="/no_backups/s1492/"  # Not needed for KITTI360OfficialDataset
```

---

### Issue 3: **Stage 1 - Missing Critical Flags** ⚠️

**File**: `scripts/train_scripts/train_kitti360_bbox_predict.sh`

**Current flags** (lines 162-163):
```bash
--predict_bbox \
--use_segmentation \
```

**Status**: ✅ CORRECT - These are required for Stage 1

**Verification**: Both flags present and correct.

---

### Issue 4: **Stage 2 - Missing `--num_inference_steps`** ⚠️

**File**: `scripts/train_scripts/train_kitti360_sem2video.sh`

**Current validation args**: Missing `--num_inference_steps`

**Fix**: Add after line 165:
```bash
--use_segmentation \
--num_inference_steps 30 \   # ADD THIS LINE
--train_H 192 \
```

This is needed for validation inference during training.

---

### Issue 5: **Stage 1 - Inconsistent Job Name** ⚠️

**File**: `scripts/train_scripts/train_kitti360_bbox_predict.sh`

**Line 2**:
```bash
#SBATCH --job-name=kittisemantic_train
```

**Line 26 comment**:
```bash
echo "Starting KITTI360 Semantic RGB Prediction Training"
```

**Confusion**: Comment says "Semantic RGB Prediction" but should say "RGB to Semantic Prediction"

**Fix**: Update line 26:
```bash
echo "Starting KITTI360 RGB-to-Semantic Prediction Training (Stage 1)"
```

---

### Issue 6: **Stage 2 - Wrong Description** ⚠️

**File**: `scripts/train_scripts/train_kitti360_sem2video.sh`

**Line 20**:
```bash
echo "Starting KITTI360 Semantic2Video Training (Step 3)"
```

**Problem**: Says "Step 3" but this is "Stage 2"

**Fix**: Update line 20:
```bash
echo "Starting KITTI360 Semantic-to-RGB Generation Training (Stage 2)"
```

---

### Issue 7: **Stage 1 - Commented Resume Checkpoint** ⚠️

**File**: `scripts/train_scripts/train_kitti360_bbox_predict.sh`

**Line 169**:
```bash
# --resume_from_checkpoint latest
```

**Status**: Commented out (correct for fresh training)

**Recommendation**: Keep commented for initial training. Uncomment only if resuming.

---

### Issue 8: **Both Scripts - WandB URL Placeholder** ⚠️

**Both scripts, last lines**:
```bash
echo "WandB URL: https://wandb.ai/<your_username>/${PROJECT_NAME}/runs/${NAME}"
```

**Fix**: Replace `<your_username>` with actual entity:
```bash
echo "WandB URL: https://wandb.ai/jaseelkt1-university-of-stuttgart/${PROJECT_NAME}/runs/${NAME}"
```

---

## ✅ VERIFIED CORRECT

### Stage 1 (train_kitti360_bbox_predict.sh)
- ✓ Checkpoint directory: `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae`
- ✓ Semantic VAE checkpoint: Implicit (loaded by DualVAEManager)
- ✓ Clip length: 25 frames
- ✓ Training resolution: 192x704
- ✓ WandB entity: `jaseelkt1-university-of-stuttgart`
- ✓ WandB project: `ctrl_v_kitti360`
- ✓ Learning rate: 5e-6
- ✓ Epochs: 10
- ✓ Flags: `--predict_bbox` and `--use_segmentation` ✅
- ✓ Dataset: `kitti360` (uses KITTI360OfficialDataset)

### Stage 2 (train_kitti360_sem2video.sh)
- ✓ Checkpoint directory: `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video_vae`
- ✓ Can run in parallel (doesn't require Stage 1 checkpoint)
- ✓ Uses ground truth semantics for conditioning
- ✓ Clip length: 25 frames
- ✓ Training resolution: 192x704
- ✓ WandB entity: `jaseelkt1-university-of-stuttgart`
- ✓ WandB project: `ctrl_v_kitti360`
- ✓ Learning rate: 1e-5 (higher than Stage 1 - appropriate for ControlNet)
- ✓ Flag: `--use_segmentation` ✅
- ✓ Uses `train_video_controlnet.py` ✅

---

## 📋 Required Fixes Summary

### Critical (Must Fix):
1. **Stage 1, Line 136**: Change `--data_root $DATASET_PATH \` to `--data_root "" \`
2. **Stage 2, Line 141**: Change `--data_root $DATASET_PATH \` to `--data_root "" \`

### Recommended (Should Fix):
3. **Stage 2**: Add `--num_inference_steps 30 \` after line 165
4. **Both scripts**: Update WandB URL to use actual entity name

### Optional (Clarity):
5. **Stage 1, Line 26**: Update comment to "RGB-to-Semantic Prediction Training (Stage 1)"
6. **Stage 2, Line 20**: Update comment to "Semantic-to-RGB Generation Training (Stage 2)"
7. **Both scripts**: Remove/comment `DATASET_PATH` variables (lines 73 and 67)

---

## ✅ Checkpoint Verification

### Test Training Checkpoint
Location: `/no_backups/s1492/Ctrl-V/test_checkpoints/test_semantic_short/checkpoint-50/`

**Check**:
```bash
ls -lh /no_backups/s1492/Ctrl-V/test_checkpoints/test_semantic_short/checkpoint-50/
```

### Semantic VAE Checkpoint (Implicit)
Location: `/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth`

**Verified**: DualVAEManager successfully loaded this in test training ✅

---

## 🎯 WandB Verification Checklist

After test training (Job 196504), verify on WandB:

1. **Project**: `ctrl_v_kitti360`
2. **Run name**: `test_semantic_vae_short`
3. **Check for**:
   - ✅ Log shows `gt_semantic_frames` (NOT `gt_bbox_frames`)
   - ✅ Validation videos show semantic segmentation
   - ✅ Training loss is logged
   - ✅ No errors during training

**URL**: https://wandb.ai/jaseelkt1-university-of-stuttgart/ctrl_v_kitti360/runs/test_semantic_vae_short

---

## 🚀 Ready to Train After Fixes

### Order of Execution:

**Option 1: Sequential (Recommended)**
1. Fix scripts (apply critical fixes 1-2)
2. Run Stage 1: `sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh`
3. Wait for completion (~4 days)
4. Run Stage 2: `sbatch scripts/train_scripts/train_kitti360_sem2video.sh`

**Option 2: Parallel (Faster)**
1. Fix scripts (apply critical fixes 1-2)
2. Run Stage 1: `sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh`
3. Run Stage 2 immediately: `sbatch scripts/train_scripts/train_kitti360_sem2video.sh`
   - Stage 2 uses ground truth semantics, doesn't need Stage 1 checkpoint
   - Both train in parallel, saves ~3 days

**Recommendation**: Use **Parallel** training since Stage 2 has ground truth semantic labels.

---

## 📊 Expected Training Time

| Stage | Duration | GPU Hours |
|-------|----------|-----------|
| Stage 1 (RGB→Semantic) | ~96 hours | ~96 |
| Stage 2 (Semantic→RGB) | ~72 hours | ~72 |
| **Parallel Total** | ~96 hours | ~168 |
| **Sequential Total** | ~168 hours | ~168 |

**Parallel saves 72 hours of wall-clock time!**

---

## Final Checklist Before Training

- [ ] Apply critical fixes to both scripts
- [ ] Verify test training checkpoint exists
- [ ] Check WandB for `gt_semantic_frames` in test run
- [ ] Ensure conda env `kitti` is activated
- [ ] Verify GPU availability: `nvidia-smi`
- [ ] Check disk space: `/no_backups/s1492/Ctrl-V/` has enough space
- [ ] Submit Stage 1: `sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh`
- [ ] (Optional) Submit Stage 2 in parallel: `sbatch scripts/train_scripts/train_kitti360_sem2video.sh`
