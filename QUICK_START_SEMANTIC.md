# Quick Start: Semantic RGB Images for KITTI-360

## Summary
This guide helps you switch from bbox overlays to semantic RGB images as input conditioning for your Ctrl-V Stage 1 training.

**Status:** ✅ All files created and ready to use!

---

## 📋 What I've Created For You

### 1. Main Roadmap Document
**File:** `SEMANTIC_MIGRATION_ROADMAP.md`
- Complete implementation strategy
- Two approaches (preprocessing vs direct loading)
- Detailed file-by-file modifications
- Testing procedures
- Troubleshooting guide

### 2. Preprocessing Script
**File:** `src/ctrlv/utils/preprocess_kitti360_semantic.py`
- Copies/symlinks semantic images into your dataset structure
- Reads original frame numbers from image symlinks
- Handles missing frames gracefully
- Provides detailed statistics

---

## 🚀 Next Steps (In Order)

### Step 1: Test Preprocessing (Dry Run) - 5 minutes
```bash
cd /usrhomes/s1492/Ctrl-V-seg

# Dry run to see what would happen (no actual changes)
python src/ctrlv/utils/preprocess_kitti360_semantic.py \
    --kitti360_src /data/public/kitti-360/KITTI-360 \
    --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
    --split train \
    --camera image_00 \
    --dry_run
```

**Expected Output:**
- List of semantic images that would be copied
- Statistics on coverage
- Any warnings about missing images

### Step 2: Run Preprocessing (Symlink Mode) - 30-60 minutes
```bash
# Use symlinks (fast, saves disk space)
python src/ctrlv/utils/preprocess_kitti360_semantic.py \
    --kitti360_src /data/public/kitti-360/KITTI-360 \
    --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
    --split train \
    --camera image_00 \
    --symlink
```

**What happens:**
- Creates `/no_backups/s1492/kitti360_ctrlv/semantics/track/train/`
- Symlinks semantic images matching your dataset frames
- Shows progress bar and final statistics

### Step 3: Verify Preprocessing - 2 minutes
```bash
# Check structure was created
ls /no_backups/s1492/kitti360_ctrlv/semantics/track/train/ | head -5

# Check one sequence
ls /no_backups/s1492/kitti360_ctrlv/semantics/track/train/2013_05_28_drive_0000_sync_0000/ | wc -l

# Verify it's a symlink
ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/train/2013_05_28_drive_0000_sync_0000/ | head -3
```

**Expected:**
- Same number of scenes as in `bboxes/track/train/`
- Each scene has matching frame count
- Files are symlinks pointing to KITTI-360 semantic images

### Step 4: Update Dataloader Code - 10 minutes

#### A. Add Constant to BDD100K Dataset
**File:** `src/ctrlv/datasets/bdd100k.py` (around line 49)

Add this line:
```python
TO_SEMANTIC_DIR = 'semantics/track'
```

So it looks like:
```python
TO_IMAGE_DIR = 'images/track'
TO_BBOX_DIR = 'bboxes/track'
TO_LABEL_DIR = 'labels'
TO_BBOX_LABELS = 'labels/box_track_20'
TO_SEG_LABELS = 'labels/seg_track_20/colormaps'
TO_SEMANTIC_DIR = 'semantics/track'  # NEW
```

#### B. Update get_bbox_image_file_by_index() Method
**File:** `src/ctrlv/datasets/bdd100k.py` (lines 219-224)

Replace:
```python
def get_bbox_image_file_by_index(self, index=None, image_file=None):
    if image_file is None:
        image_file = self.get_image_file_by_index(index)
    if self.use_segmentation:
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_SEG_LABELS)[:-4]+'.png'
    return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_BBOX_DIR)
```

With:
```python
def get_bbox_image_file_by_index(self, index=None, image_file=None):
    if image_file is None:
        image_file = self.get_image_file_by_index(index)
    if self.use_segmentation:
        # Use dedicated semantics directory for KITTI-360 semantic RGB images
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_SEMANTIC_DIR)
    return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_BBOX_DIR)
```

### Step 5: Update Training Script - 2 minutes
**File:** `scripts/train_scripts/train_kitti360_bbox_predict.sh` (around line 153)

Find the line with `--predict_bbox` and add `--use_segmentation` after it:

```bash
    --predict_bbox \
    --use_segmentation \
    --num_inference_steps 30 \
```

### Step 6: Test Dataloader - 5 minutes
Create a simple test script:

```bash
cd /usrhomes/s1492/Ctrl-V-seg
python -c "
from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset

print('Testing KITTI-360 semantic dataloader...')
dataset = KITTI360BDDDataset(
    root='/no_backups/s1492/',
    train=True,
    data_type='clip',
    clip_length=8,
    if_return_bbox_im=True,
    use_segmentation=True,
    use_preplotted_bbox=True
)

print(f'✓ Dataset loaded: {len(dataset)} clips')
print(f'✓ Sequences: {len(dataset.clip_folders)}')

# Load first sample
print('\\nLoading first sample...')
images, targets, prompt, idx, semantic_imgs = dataset[0]
print(f'✓ Images shape: {images.shape}')
print(f'✓ Semantic images shape: {semantic_imgs.shape}')
print(f'✓ Value range: [{semantic_imgs.min():.2f}, {semantic_imgs.max():.2f}]')
print('\\n✅ Test passed!')
"
```

**Expected Output:**
```
Testing KITTI-360 semantic dataloader...
✓ Dataset loaded: XXXX clips
✓ Sequences: XX

Loading first sample...
✓ Images shape: torch.Size([8, 3, 128, 512])
✓ Semantic images shape: torch.Size([8, 3, 128, 512])
✓ Value range: [-1.00, 1.00]

✅ Test passed!
```

### Step 7: Quick Training Test - 5 minutes
```bash
cd /usrhomes/s1492/Ctrl-V-seg
export WANDB_MODE=offline

accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    tools/train_video_diffusion.py \
    --run_name test_semantic_$(date +%y%m%d_%H%M) \
    --data_root /no_backups/s1492/ \
    --dataset_name kitti360 \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir /tmp/test_semantic_train \
    --variant fp16 \
    --train_batch_size 1 \
    --max_train_steps 5 \
    --use_segmentation \
    --predict_bbox \
    --clip_length 8 \
    --num_cond_bbox_frames 3 \
    --train_H 128 \
    --train_W 512
```

**What to check:**
- No errors during dataloader initialization
- Training starts successfully
- First few steps complete
- Check logged images (should show semantic images, not bbox overlays)

### Step 8: Full Training
Once everything works, use your original training script:

```bash
cd /usrhomes/s1492/Ctrl-V-seg
bash scripts/train_scripts/train_kitti360_bbox_predict.sh
```

(Make sure you added `--use_segmentation` flag in Step 5)

---

## 📁 Files Modified Summary

### ✅ Created (by me):
1. `SEMANTIC_MIGRATION_ROADMAP.md` - Detailed roadmap
2. `QUICK_START_SEMANTIC.md` - This file
3. `src/ctrlv/utils/preprocess_kitti360_semantic.py` - Preprocessing script

### 📝 To Modify (by you):
1. `src/ctrlv/datasets/bdd100k.py` - Add 1 constant, modify 1 method (5 lines total)
2. `scripts/train_scripts/train_kitti360_bbox_predict.sh` - Add 1 flag (1 line)

---

## 🔍 Key Differences: BBox vs Semantic

| Aspect | BBox Input | Semantic RGB Input |
|--------|-----------|-------------------|
| **Directory** | `bboxes/track/train/` | `semantics/track/train/` |
| **Content** | Rendered bbox overlays | Colored semantic segmentation |
| **Flag** | (default) | `--use_segmentation` |
| **Use Case** | Stage 1: BBox prediction | Stage 1: Semantic-conditioned generation |

---

## ⚠️ Important Notes

1. **Both pipelines coexist**: You can keep both bbox and semantic directories. Switch between them using the `--use_segmentation` flag.

2. **Frame alignment**: The preprocessing script automatically handles frame number mapping by reading symlink targets.

3. **Semantic coverage**: Not all frames may have semantic annotations. The script will report coverage statistics.

4. **Disk space** (with symlinks):
   - Negligible (just symlink metadata)
   - Original semantic images remain in `/data/public/kitti-360/`

5. **Disk space** (with copy):
   - Estimate: ~500MB - 2GB depending on your dataset size
   - Check before running: `du -sh /data/public/kitti-360/KITTI-360/data_2d_semantics/train/`

---

## 🆘 Troubleshooting

### Issue: "Cannot determine original frame number"
**Cause:** Image files are not symlinks or path format is different
**Solution:** Check if `/no_backups/s1492/kitti360_ctrlv/images/track/train/` contains symlinks:
```bash
ls -la /no_backups/s1492/kitti360_ctrlv/images/track/train/2013_05_28_drive_0000_sync_0000/ | head -3
```

### Issue: "Semantic image not found"
**Cause:** Frame doesn't have semantic annotation
**Solution:** This is expected for some frames. Check coverage percentage in preprocessing output.

### Issue: Shape mismatch in training
**Cause:** Semantic images have different resolution
**Solution:** The dataloader should handle resizing automatically. Check `train_H` and `train_W` parameters.

### Issue: Training is slow
**Cause:** Semantic images are larger files
**Solution:** This is expected. Consider using fp16 mixed precision (already enabled in your script).

---

## ✅ Checklist

Before starting training with semantic images:

- [ ] Ran preprocessing script (dry run)
- [ ] Ran preprocessing script (actual)
- [ ] Verified semantic directory structure
- [ ] Added `TO_SEMANTIC_DIR` constant to `bdd100k.py`
- [ ] Modified `get_bbox_image_file_by_index()` in `bdd100k.py`
- [ ] Added `--use_segmentation` flag to training script
- [ ] Tested dataloader with simple Python script
- [ ] Ran short training test (5-10 steps)
- [ ] Checked logged images show semantics (not bbox)
- [ ] Ready for full training!

---

## 📊 Expected Timeline

| Phase | Time | Complexity |
|-------|------|------------|
| Preprocessing (dry run) | 5 min | Easy |
| Preprocessing (actual) | 30-60 min | Easy |
| Code modifications | 10-15 min | Easy |
| Testing | 10-15 min | Easy |
| **Total** | **~1-1.5 hours** | **Low** |

---

## 🎉 Next After This Works

Once training with semantic images works:
1. Monitor training metrics on W&B
2. Compare with bbox-based training
3. Adjust hyperparameters if needed
4. Proceed to Stage 2 (Box2Video) with the trained model

---

**Questions or Issues?** Refer to `SEMANTIC_MIGRATION_ROADMAP.md` for detailed information!

**Ready to start?** Jump to Step 1 above! 🚀
