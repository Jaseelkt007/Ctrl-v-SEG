# Semantic RGB Images Migration Roadmap
## From BBox Input to Semantic RGB Images for KITTI-360 Ctrl-V Training

**Date:** Oct 27, 2025  
**Context:** Transitioning from bounding box images to semantic RGB images as input conditioning for Stage 1 (bbox generator) of the Ctrl-V two-stage pipeline.

---

## 📋 Current Setup Summary

### What You Have
- **Dataset Location:** `/no_backups/s1492/kitti360_ctrlv/`
- **Current Input:** Bounding box images (rendered overlays)
- **Dataloader:** `KITTI360BDDDataset` (adapted from BDD100K format)
- **Training Script:** `scripts/train_scripts/train_kitti360_bbox_predict.sh`
- **Stage:** Stage 1 - Bbox prediction model

### Semantic Images Available
- **Location:** `/data/public/kitti-360/KITTI-360/data_2d_semantics/`
- **Structure:**
  ```
  data_2d_semantics/
  └── train/
      ├── 2013_05_28_drive_0000_sync/
      │   ├── image_00/semantic_rgb/*.png
      │   └── image_01/semantic_rgb/*.png
      ├── 2013_05_28_drive_0002_sync/
      └── ...
  ```
- **Format:** PNG files with frame numbers (e.g., `0000000250.png`)

---

## 🎯 Goal
Replace bbox overlays with semantic RGB images as conditioning input while maintaining the existing BDD100K-compatible data pipeline with **minimal code changes**.

---

## 🗺️ Migration Strategy

### Option A: Preprocessing Approach (Recommended) ⭐
**Copy semantic images into your existing dataset structure**

**Pros:**
- Minimal code changes (only 1-2 files)
- Leverages existing dataloader infrastructure
- Consistent with current workflow
- Easy to switch back to bbox if needed

**Cons:**
- Requires disk space for copying files
- One-time preprocessing step needed

### Option B: Direct Loading Approach
**Modify dataloader to load semantic images directly from KITTI-360 path**

**Pros:**
- No disk space duplication
- Direct access to source data

**Cons:**
- More code changes required
- Path mapping logic between datasets
- Harder to debug

---

## 📝 Implementation Plan (Option A - Recommended)

### Phase 1: Preprocessing - Create Semantic Image Dataset Structure
**Estimated Time:** 1-2 hours (mostly disk I/O)

#### Step 1.1: Create Preprocessing Script
**File to Create:** `src/ctrlv/utils/preprocess_kitti360_semantic.py`

**What it does:**
1. Reads the existing label JSON files from `/no_backups/s1492/kitti360_ctrlv/labels/box_track_20/`
2. For each sequence (e.g., `2013_05_28_drive_0000_sync_0000.json`):
   - Identifies which frames are used
   - Maps frame IDs to semantic image filenames
   - Copies semantic RGB images from `/data/public/kitti-360/KITTI-360/data_2d_semantics/train/{sequence}/image_00/semantic_rgb/`
   - Saves to `/no_backups/s1492/kitti360_ctrlv/semantics/track/train/{sequence}/`

**Key Functions:**
```python
def map_frame_to_semantic_path(sequence_name, frame_id)
def copy_semantic_images(src_semantic_dir, dst_semantic_dir, frame_list)
def process_sequence(sequence_json, semantic_src_root, semantic_dst_root)
```

#### Step 1.2: Run Preprocessing
```bash
python src/ctrlv/utils/preprocess_kitti360_semantic.py \
    --kitti360_src /data/public/kitti-360/KITTI-360 \
    --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
    --split train
```

**Expected Output Structure:**
```
/no_backups/s1492/kitti360_ctrlv/
├── images/track/train/        # Original images (unchanged)
├── bboxes/track/train/        # Bbox overlays (keep for comparison)
├── semantics/track/train/     # NEW: Semantic RGB images
│   ├── 2013_05_28_drive_0000_sync_0000/
│   │   ├── 2013_05_28_drive_0000_sync_0000-0000001.png
│   │   ├── 2013_05_28_drive_0000_sync_0000-0000002.png
│   │   └── ...
│   └── ...
└── labels/box_track_20/train/ # Labels (unchanged)
```

---

### Phase 2: Modify Dataloader
**Estimated Time:** 30 minutes

#### Step 2.1: Update BDD100K Dataset Class Constants
**File to Modify:** `src/ctrlv/datasets/bdd100k.py` (lines 45-50)

**Current:**
```python
TO_IMAGE_DIR = 'images/track'
TO_BBOX_DIR = 'bboxes/track'
TO_LABEL_DIR = 'labels'
TO_BBOX_LABELS = 'labels/box_track_20'
TO_SEG_LABELS = 'labels/seg_track_20/colormaps'
```

**Add:**
```python
TO_SEMANTIC_DIR = 'semantics/track'  # NEW: for semantic RGB images
```

#### Step 2.2: Update `get_bbox_image_file_by_index()` Method
**File to Modify:** `src/ctrlv/datasets/bdd100k.py` (lines 219-224)

**Current Logic:**
- If `use_segmentation=True`, uses `TO_SEG_LABELS` path
- Otherwise, uses `TO_BBOX_DIR` path

**New Logic:**
```python
def get_bbox_image_file_by_index(self, index=None, image_file=None):
    if image_file is None:
        image_file = self.get_image_file_by_index(index)
    
    if self.use_segmentation:
        # For semantic RGB images, use dedicated semantics directory
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, 
                                  BDD100KDataset.TO_SEMANTIC_DIR)
    
    # For bbox overlays (original behavior)
    return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, 
                              BDD100KDataset.TO_BBOX_DIR)
```

#### Step 2.3: Verify Image Loading in `_getimageitem()`
**File:** `src/ctrlv/datasets/bdd100k.py` (lines 131-142)

**Current code already handles this correctly:**
```python
if self.use_preplotted_bbox or self.use_segmentation:
    bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
    bbox_im = Image.open(bbox_file)
    if self.use_segmentation:
        bbox_im = bbox_im.convert('RGB')  # Ensures RGB format
    if not self.transform is None:
        bbox_im = self.transform(bbox_im)
```

**No changes needed here!** ✅

---

### Phase 3: Update KITTI360 Wrapper (Optional Enhancement)
**Estimated Time:** 15 minutes

#### Step 3.1: Override Initialization in KITTI360BDDDataset
**File:** `src/ctrlv/datasets/kitti360_bdd_format.py` (lines 139-146)

**Current:**
```python
else:
    seg_label_dir = os.path.join(self.root, self.version, 
                                 BDD100KDataset.TO_SEG_LABELS, self._location)
    self.clip_folders = sorted(os.listdir(seg_label_dir))
    self.clip_folder_lengths = {
        k: len(os.listdir(os.path.join(seg_label_dir, k))) 
        for k in self.clip_folders
    }
```

**Updated:**
```python
else:
    # Use semantic images directory instead of SEG_LABELS
    semantic_dir = os.path.join(self.root, self.version, 
                                BDD100KDataset.TO_SEMANTIC_DIR, self._location)
    self.clip_folders = sorted(os.listdir(semantic_dir))
    self.clip_folder_lengths = {
        k: len(os.listdir(os.path.join(semantic_dir, k))) 
        for k in self.clip_folders
    }
```

---

### Phase 4: Update Training Script
**Estimated Time:** 5 minutes

#### Step 4.1: Modify Training Arguments
**File:** `scripts/train_scripts/train_kitti360_bbox_predict.sh` (lines 120-158)

**Current:**
```bash
--predict_bbox \
```

**Add AFTER this line:**
```bash
--use_segmentation \
```

**What this does:**
- Tells the dataloader to use semantic images instead of bbox overlays
- Already supported by existing code in `get_dataloader()` (line 70-71 in util.py)

---

### Phase 5: Testing & Validation
**Estimated Time:** 1 hour

#### Test 5.1: Verify Preprocessing
```bash
# Check if semantic images were copied correctly
ls /no_backups/s1492/kitti360_ctrlv/semantics/track/train/ | wc -l
ls /no_backups/s1492/kitti360_ctrlv/bboxes/track/train/ | wc -l
# Should have same number of sequences

# Verify file counts match
python -c "
import os
sem_dir = '/no_backups/s1492/kitti360_ctrlv/semantics/track/train'
for seq in os.listdir(sem_dir):
    sem_count = len(os.listdir(os.path.join(sem_dir, seq)))
    print(f'{seq}: {sem_count} frames')
"
```

#### Test 5.2: Test Dataloader
**Create:** `test/test_kitti360_semantic_loader.py`

```python
from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset

# Test with semantic images
dataset = KITTI360BDDDataset(
    root='/no_backups/s1492/',
    train=True,
    data_type='clip',
    clip_length=8,
    if_return_bbox_im=True,
    use_segmentation=True,  # KEY FLAG
    use_preplotted_bbox=True
)

print(f"Dataset length: {len(dataset)}")
print(f"Clip folders: {len(dataset.clip_folders)}")

# Load first sample
images, targets, prompt, idx, semantic_imgs = dataset[0]
print(f"Images shape: {images.shape}")
print(f"Semantic images shape: {semantic_imgs.shape}")
print(f"Semantic image dtype: {semantic_imgs.dtype}")
print(f"Semantic value range: [{semantic_imgs.min()}, {semantic_imgs.max()}]")

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(images[i].permute(1, 2, 0))
    axes[0, i].set_title(f'Frame {i}')
    axes[1, i].imshow(semantic_imgs[i].permute(1, 2, 0))
    axes[1, i].set_title(f'Semantic {i}')
plt.savefig('/tmp/semantic_test.png')
print("Saved visualization to /tmp/semantic_test.png")
```

#### Test 5.3: Dry-run Training
```bash
# Quick test with small number of steps
cd /usrhomes/s1492/Ctrl-V-seg
export WANDB_MODE=offline

accelerate launch \
    --num_processes 1 \
    tools/train_video_diffusion.py \
    --run_name test_semantic_$(date +%y%m%d_%H%M) \
    --data_root /no_backups/s1492/ \
    --dataset_name kitti360 \
    --train_batch_size 1 \
    --max_train_steps 10 \
    --use_segmentation \
    --predict_bbox \
    --clip_length 8 \
    --num_cond_bbox_frames 3 \
    --output_dir /tmp/test_semantic_train
```

**Expected:**
- Should load without errors
- Check logged sample images in W&B or output directory
- Semantic images should appear instead of bbox overlays

---

## 📁 Files to Create/Modify Summary

### New Files (1 file)
1. **`src/ctrlv/utils/preprocess_kitti360_semantic.py`** - Preprocessing script
   - ~200 lines
   - Copies semantic images into BDD100K-compatible structure

### Files to Modify (2-3 files)

#### Required Changes:
1. **`src/ctrlv/datasets/bdd100k.py`**
   - Add `TO_SEMANTIC_DIR` constant (1 line)
   - Update `get_bbox_image_file_by_index()` method (3-5 lines)
   
2. **`scripts/train_scripts/train_kitti360_bbox_predict.sh`**
   - Add `--use_segmentation` flag (1 line)

#### Optional Enhancement:
3. **`src/ctrlv/datasets/kitti360_bdd_format.py`**
   - Update `__init__` to use `TO_SEMANTIC_DIR` when `use_segmentation=True` (5 lines)

---

## 🔄 Alternative: Option B Implementation (Direct Loading)

If you prefer not to copy files, here's the direct loading approach:

### Files to Modify:
1. **`src/ctrlv/datasets/kitti360_bdd_format.py`**
   - Add `kitti360_semantic_root` parameter
   - Override `get_bbox_image_file_by_index()` to map to KITTI-360 semantic paths
   - Add frame number conversion logic

**Key Challenge:**
- Your BDD format uses: `{sequence_name}-{frame_number:07d}.png`
- KITTI-360 uses: `{absolute_frame_number:010d}.png`
- Need mapping between sequence-relative and absolute frame numbers

**Code Complexity:** Medium (50-80 lines of new code)

---

## ⚠️ Important Considerations

### 1. Semantic Image Format
- Semantic images are **RGB encoded** (not class IDs)
- Each color represents a different class
- Already compatible with the model (treats as RGB input)

### 2. Frame Alignment
- Semantic images may not exist for ALL frames in your dataset
- Preprocessing script should handle missing frames gracefully
- Consider using a mapping file to track available semantic frames

### 3. Camera Selection
- KITTI-360 has `image_00` and `image_01` (stereo cameras)
- Your dataset likely uses one camera - verify which one
- Default to `image_00` (left camera) if unsure

### 4. Resolution Differences
- Original images: 376 x 1408
- Semantic images: Same resolution
- Training resolution: 128 x 512 (from training script)
- Dataloader handles resizing automatically ✅

---

## 🧪 Validation Checklist

- [ ] Preprocessing script runs without errors
- [ ] Semantic directory structure matches bbox directory
- [ ] File counts match between semantic and bbox directories
- [ ] Dataloader loads semantic images correctly
- [ ] Image shapes are correct (B, C, H, W)
- [ ] Image value ranges are appropriate [0, 1] or [-1, 1]
- [ ] Training starts without errors
- [ ] W&B logs show semantic images (not bbox overlays)
- [ ] First few training steps complete successfully

---

## 📊 Estimated Total Time

| Phase | Time |
|-------|------|
| Phase 1: Preprocessing | 1-2 hours |
| Phase 2: Dataloader | 30 min |
| Phase 3: KITTI360 Wrapper | 15 min |
| Phase 4: Training Script | 5 min |
| Phase 5: Testing | 1 hour |
| **Total** | **3-4 hours** |

---

## 🎯 Next Steps

1. **Review this roadmap** - Confirm the approach makes sense
2. **Choose option** - Option A (preprocessing) vs Option B (direct loading)
3. **Start with Phase 1** - Create preprocessing script
4. **Test incrementally** - Don't move to next phase until current works
5. **Keep bbox pipeline** - Don't delete bbox images (useful for comparison)

---

## 💡 Tips for Success

1. **Test on small subset first** - Use 1-2 sequences for initial testing
2. **Keep both pipelines** - Maintain ability to train with bbox or semantic
3. **Version control** - Commit before making changes
4. **Log everything** - Add print statements in preprocessing script
5. **Visual verification** - Always look at a few loaded images

---

## 🆘 Troubleshooting

### Issue: "FileNotFoundError: semantic image not found"
**Solution:** Check frame number mapping in preprocessing script

### Issue: "Shape mismatch in dataloader"
**Solution:** Verify semantic images have same naming convention as bbox images

### Issue: "Model expects 3 channels, got different"
**Solution:** Ensure semantic images are converted to RGB (already handled in code)

### Issue: "Training slower than with bbox"
**Solution:** Semantic images are larger files - this is expected, consider using GPU with more memory

---

**Ready to proceed?** Start with Phase 1 (preprocessing script) and I'll help implement it step by step!
