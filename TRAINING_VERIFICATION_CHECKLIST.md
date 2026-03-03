# ✅ Training Verification Checklist

After running the test scripts, verify the following to ensure semantic VAE training is correct.

---

## 📋 Pre-Training Verification

### ✅ Training Script Configuration

**File:** `scripts/train_scripts/train_kitti360_bbox_predict.sh`

Verify these parameters are set:
- [x] `--dataset_name kitti360` ✅
- [x] `--data_root /no_backups/s1492/` ✅
- [x] `--use_segmentation` ✅
- [x] `--predict_bbox` ✅
- [x] `--train_H 192` ✅
- [x] `--train_W 704` ✅
- [x] `--clip_length 25` ✅
- [x] `--train_batch_size 1` ✅
- [x] `--gradient_accumulation_steps 6` ✅
- [x] `--validation_steps 500` ✅
- [x] `--checkpointing_steps 200` ✅

**Missing:** `--return_semantic_ids` flag is NOT needed in script!
- The training code automatically enables it when `use_segmentation=True`
- See `tools/train_video_diffusion.py:250`

---

## 🧪 Test 1: Semantic VAE Fix Validation

**Run:**
```bash
sbatch scripts/test_scripts/test_semantic_vae_fix.sh
```

**Expected Output:**
```
✅ TEST 1 PASSED: Dataset loads grayscale semantic IDs
✅ TEST 2 PASSED: Collate function creates 'semantic_ids' key
✅ TEST 3 PASSED: DualVAEManager encodes semantic IDs
✅ TEST 4 PASSED: Training code uses semantic VAE path
🟢 ALL TESTS PASSED - FIXES ARE CORRECT!
```

**Check Log:**
```bash
tail -100 /no_backups/s1492/Ctrl-V/logs/test_semantic_fix_*.out
```

**If any test fails:**
- ❌ DO NOT proceed to training
- Review error messages
- Check semantic dataset paths
- Verify KITTI360BDDDataset is imported correctly

---

## 🧪 Test 2: Short Training Test (50 steps)

**Run:**
```bash
sbatch scripts/test_scripts/test_short_training.sh
```

**Monitor Progress:**
```bash
# Watch live output
tail -f /no_backups/s1492/Ctrl-V/logs/test_train_short_*.out

# Check for errors
tail -f /no_backups/s1492/Ctrl-V/logs/test_train_short_*.err
```

**Expected Behavior:**

### ✅ Console Output Should Show:
1. **DualVAEManager initialization:**
   ```
   Initializing DualVAEManager with semantic VAE from ...
   ✓ DualVAEManager initialized for semantic VAE encoding
   ```

2. **Dataset loading:**
   ```
   Loading dataset: kitti360
   return_semantic_ids: True
   ```

3. **Training progress:**
   ```
   Steps:  X%|███ | X/50 [XX:XX<XX:XX, Xs/it, lr=5e-6, step_loss=X.XXXX]
   ```

4. **Validation at step 50:**
   ```
   Running validation...
   Saving checkpoint to checkpoint-50
   ```

### ✅ WandB Dashboard Checks:

**URL:** https://wandb.ai/jaseelkt1-university-of-stuttgart/ctrl_v_kitti360/runs/test_semantic_vae_short

**Verify:**
1. **Log Names (CRITICAL):**
   - ✅ `gt_semantic_frames` exists
   - ❌ `gt_bbox_frames` does NOT exist

2. **Media Tab:**
   - Should show validation videos at step 50
   - Check `gt_semantic_frames` shows semantic predictions

3. **Charts:**
   - `train/loss` should be logging
   - `train/lr` should show 5e-6

4. **System:**
   - GPU utilization should be high (~90%+)
   - Memory usage should be stable

### ✅ Checkpoint Verification:

```bash
ls -lh /no_backups/s1492/Ctrl-V/test_checkpoints/test_semantic_short/checkpoint-50/
```

**Should contain:**
- `unet/` directory (Stage 1 UNet model)
- `optimizer.bin` (~12GB)
- `scheduler.bin`
- `scaler.pt`
- `random_states_0.pkl`

---

## 🔍 What to Look For (Good vs Bad)

### ✅ GOOD Behavior:

1. **Logs mention:**
   - "DualVAEManager initialized"
   - "Semantic VAE"
   - "semantic_ids"
   - "encode_semantic_from_ids"

2. **WandB shows:**
   - `gt_semantic_frames` in media
   - Training loss decreasing
   - No OOM errors

3. **Training progresses:**
   - Steps increment: 1 → 2 → 3 → ... → 50
   - Loss values are reasonable (0.05 - 0.15 range initially)
   - Validation completes without crashes

### ❌ BAD Behavior (Indicates Bug):

1. **Logs show:**
   - Only mentions "RGB VAE" (no Semantic VAE)
   - No mention of "semantic_ids"
   - Uses `bbox_images` instead of `semantic_ids`

2. **WandB shows:**
   - `gt_bbox_frames` (OLD naming - wrong!)
   - RGB colorized semantic maps (not grayscale)
   - Loss pattern similar to previous failed run

3. **Training fails:**
   - `KeyError: 'semantic_ids'` → Dataset not returning semantic IDs
   - `KeyError: 'bbox_ids'` → Old bug still present
   - CUDA OOM → Need to reduce batch size further

---

## 🚀 Full Training Readiness Check

**Before running full training, confirm:**

- [ ] Test 1 passed (all 4 tests ✅)
- [ ] Test 2 passed (50 steps completed ✅)
- [ ] WandB shows `gt_semantic_frames` (not `gt_bbox_frames`) ✅
- [ ] Checkpoint-50 saved correctly ✅
- [ ] No errors in logs ✅

**If all checks pass:**
```bash
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh
```

**Monitor:**
```bash
# Check queue
squeue -u s1492

# Watch progress
tail -f /no_backups/s1492/Ctrl-V/logs/train_new_vae_*.out

# Check for errors
tail -f /no_backups/s1492/Ctrl-V/logs/train_new_vae_*.err

# Check WandB
# URL will be in the logs or at:
# https://wandb.ai/jaseelkt1-university-of-stuttgart/ctrl_v_kitti360
```

---

## 📊 Expected Full Training Behavior

### First 1000 Steps:
- Loss should decrease from ~0.15 → ~0.08
- Steps take ~12-15 seconds each
- Memory usage stable at ~40GB
- GPU utilization ~95%

### After 5000 Steps (~18 hours):
- Loss should be ~0.05-0.07
- Validation images show semantic predictions improving
- WandB charts show steady progress

### Warning Signs:
- ❌ Loss not decreasing → Check if semantic VAE is being used
- ❌ OOM errors → Reduce batch size or increase grad accumulation
- ❌ Steps too slow (>20s) → Check GPU is not throttling
- ❌ Loss unstable (jumping) → May need to adjust learning rate

---

## 🔄 If Training Needs to be Restarted

**To resume from checkpoint:**
```bash
# Edit train_kitti360_bbox_predict.sh
# Uncomment line:
# --resume_from_checkpoint latest
```

**To start fresh:**
```bash
# Delete test checkpoints first
rm -rf /no_backups/s1492/Ctrl-V/test_checkpoints/*

# Submit fresh training
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh
```

---

## 📝 Post-Training Verification

**After training completes (or reaches 10k+ steps):**

1. **Check final checkpoint:**
   ```bash
   ls -lh /no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae/
   ```

2. **Review WandB metrics:**
   - Final loss should be < 0.05
   - Validation images should show good semantic predictions

3. **Test inference:**
   - Use checkpoint to generate semantic predictions
   - Compare with ground truth

---

**Created:** February 12, 2026  
**Status:** Ready for testing
