# RGB Semantic Issue - Root Cause & Solution

## 🔴 Problem Summary

Your quick test evaluation showed:
- ✅ **Good news**: Black frames bug is fixed! Frames now have content
- ❌ **New issue**: Predicted semantic images are **grayscale (ID-based)** instead of **RGB colored**
- 💥 **Impact**: Generated videos are poor quality because Stage 2 was trained on RGB semantics

## 🎯 Root Cause Identified

### What Happened (Timeline)

1. **✅ Step 1**: You correctly preprocessed validation data with RGB semantics using:
   ```bash
   python src/ctrlv/utils/preprocess_kitti360_semantic.py \
       --split val --symlink
   ```
   This created symlinks to: `/data/.../semantic_rgb/*.png` (RGB colored)

2. **❌ Step 2**: You later ran `/usrhomes/s1492/Ctrl-V/scripts/eval_scripts/fix_semantic_symlinks.py`
   This script **converted RGB → ID-based** for DRN evaluation:
   ```python
   # Line 50 in fix_semantic_symlinks.py
   new_target = current_target.replace('semantic_rgb', 'semantic')
   ```

3. **💥 Result**: Validation semantics now point to wrong folder:
   - **Current (WRONG)**: `/data/.../semantic/*.png` (ID-based, grayscale)
   - **Should be**: `/data/.../semantic_rgb/*.png` (RGB colored)

### Proof

Check current validation symlink:
```bash
$ ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/.../000001.png
→ .../image_00/semantic/0000000386.png  # ❌ ID-based (grayscale)
```

Should point to:
```bash
→ .../image_00/semantic_rgb/0000000386.png  # ✅ RGB colored
```

Check training data (still correct):
```bash
$ ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/train/.../000001.png
-rw-rw-r-- 14112 bytes  # ✅ Actual RGB file (not symlink)
```

## ✅ Solution Created

### Script: `restore_rgb_semantic_symlinks.py`

Location: `/usrhomes/s1492/Ctrl-V-seg/scripts/eval_scripts/restore_rgb_semantic_symlinks.py`

This script:
- Converts symlinks from `/semantic/` → `/semantic_rgb/`
- Verifies RGB targets exist
- Provides dry-run mode for safety

### How to Fix

#### Step 1: Dry Run (Check what will be changed)
```bash
cd /usrhomes/s1492/Ctrl-V-seg
python scripts/eval_scripts/restore_rgb_semantic_symlinks.py --dry_run
```

This shows you what will be updated without actually changing anything.

#### Step 2: Apply Fix
```bash
python scripts/eval_scripts/restore_rgb_semantic_symlinks.py
```

This updates all validation semantic symlinks to point to RGB versions.

#### Step 3: Verify Fix
```bash
# Check one symlink
ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000001.png

# Should now point to:
# → .../semantic_rgb/0000000386.png  (not .../semantic/...)
```

#### Step 4: Re-run Quick Test
```bash
cd /usrhomes/s1492/Ctrl-V-seg/scripts/eval_scripts
sbatch eval_kitti360_sem_overall_QUICKTEST.sh
```

## Why This Matters

### Training vs Evaluation Consistency

**Your Model Was Trained On:**
- ✅ RGB semantic maps (3 channels, colored)
- ✅ VAE pre-trained on RGB images
- ✅ Each semantic class has a specific RGB color

**Evaluation Was Using:**
- ❌ ID-based semantic maps (1 channel, grayscale)
- ❌ Incompatible with RGB-trained VAE
- ❌ Values are class IDs (0-33), not RGB colors

**Result:**
- Stage 1 predictions look grayscale because model expects RGB input
- Stage 2 RGB generation is poor because it receives wrong semantic format

## Dataset Structure

### Correct Structure (After Fix)

```
/no_backups/s1492/kitti360_ctrlv/
├── semantics/track/
│   ├── train/
│   │   └── scene_*/
│   │       └── *.png  (actual RGB files, 3-channel color)
│   └── val/
│       └── scene_*/
│           └── *.png  (symlinks → .../semantic_rgb/*.png) ✅
```

### Original KITTI-360 Structure

```
/data/public/kitti-360/KITTI-360/data_2d_semantics/train/
└── 2013_05_28_drive_*_sync/
    └── image_00/
        ├── semantic/          (ID-based, grayscale, 1-channel)
        └── semantic_rgb/      (RGB colored, 3-channel) ← Use this!
```

## File Comparison

### ID-based Semantic (WRONG for Ctrl-V)
```
Path: .../semantic/0000000386.png
Type: PNG grayscale (1 channel)
Size: ~3KB
Values: 0-33 (class IDs)
Example: Road=7, Sky=10, Building=11
```

### RGB Semantic (CORRECT for Ctrl-V)
```
Path: .../semantic_rgb/0000000386.png
Type: PNG color (3 channels)
Size: ~15KB
Values: RGB colors
Example: Road=(255,0,255) magenta, Sky=(70,130,180) blue
```

## Why DRN Needed ID-based Maps

The `fix_semantic_symlinks.py` script was created for a different purpose:
- DRN semantic segmentation evaluation expects ID-based maps
- For computing mIoU with prediction IDs vs ground truth IDs
- This is NOT needed for Ctrl-V which works with RGB

**DO NOT run `fix_semantic_symlinks.py` for Ctrl-V evaluation!**

## Prevention

To avoid this in the future:

1. **Keep separate validation sets** for different evaluation types:
   - `/semantics/track/val/` - RGB for Ctrl-V
   - `/semantics/track/val_drn/` - ID-based for DRN (if needed)

2. **Document which format each tool expects**

3. **Check symlink targets** before running evaluation:
   ```bash
   ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/*/00001.png | head -3
   # Should contain "semantic_rgb" not just "semantic"
   ```

## Expected Results After Fix

### Before Fix (Current State)
- ❌ Predicted semantics: Grayscale, ID-based looking
- ❌ Generated videos: Poor quality (wrong semantic conditioning)
- ✅ Ground truth semantics: Also showing as ID-based (symlinks are wrong)

### After Fix
- ✅ Predicted semantics: RGB colored (magenta roads, blue sky, etc.)
- ✅ Generated videos: Good quality (correct semantic conditioning)
- ✅ Ground truth semantics: RGB colored (correct reference)
- ✅ Both stages work as trained

## Commands Summary

```bash
# 1. Check current state (should show semantic/ not semantic_rgb/)
ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/*/00001.png | head -2

# 2. Dry run to see what will change
python scripts/eval_scripts/restore_rgb_semantic_symlinks.py --dry_run

# 3. Apply fix
python scripts/eval_scripts/restore_rgb_semantic_symlinks.py

# 4. Verify fix applied
ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/*/00001.png | head -2
# Should now show semantic_rgb/

# 5. Re-run evaluation
sbatch scripts/eval_scripts/eval_kitti360_sem_overall_QUICKTEST.sh

# 6. Check results - should now have RGB colored semantics!
```

## Files Created

1. **restore_rgb_semantic_symlinks.py** - Fix script (safe, has dry-run mode)
2. **RGB_SEMANTIC_ISSUE_EXPLAINED.md** - This document

## Questions?

- ❓ **Why did this happen?** You ran `fix_semantic_symlinks.py` meant for DRN evaluation
- ❓ **Is my training wrong?** No! Training on RGB is correct
- ❓ **Will this break DRN evaluation?** DRN uses different paths, won't affect it
- ❓ **Do I need to retrain?** No! Just fix the validation symlinks
- ❓ **What about train set?** Train is fine (actual RGB files, not symlinks)

## Next Steps

1. ✅ Run `restore_rgb_semantic_symlinks.py` (already created)
2. ✅ Re-run quick test to verify RGB semantics
3. ✅ If successful, run full evaluation
4. 📊 Enjoy proper RGB semantic predictions and better video quality!
