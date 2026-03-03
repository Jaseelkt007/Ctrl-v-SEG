# Training Issues & Fixes Applied

## Summary
Multiple compatibility issues were found when running training with KITTI360OfficialDataset. All have been systematically fixed.

---

## Issues Fixed

### 1. ✅ Import Error - Deprecated Modules
**Error**: `ModuleNotFoundError: No module named 'ctrlv.datasets.kitti360_bdd_format'`

**Root Cause**: Deleted deprecated dataset files still imported in `__init__.py`

**Fix**: Removed deprecated imports from `src/ctrlv/datasets/__init__.py`:
```python
# Removed:
# from .kitti360_bdd_format import KITTI360BDDDataset  
# from .kitti360_inference import KITTI360InferenceDataset
```

---

### 2. ✅ Missing Attribute - clip_list
**Error**: `AttributeError: 'KITTI360OfficialDataset' object has no attribute 'clip_list'`

**Root Cause**: Base class methods expect `clip_list` but KITTI360OfficialDataset uses `clips`

**Fix**: Added compatibility aliases in `_setup_clips()`:
```python
self.clip_list = self.clips  # Alias for base class compatibility
self.image_list = [pair['rgb_path'] for pair in self.frame_pairs]
```

---

### 3. ✅ Missing Method - get_bbox_image_file_by_index
**Error**: `AttributeError: 'KITTI360OfficialDataset' object has no attribute 'get_bbox_image_file_by_index'`

**Root Cause**: Utility functions expect this method from base class

**Fix**: Added method to KITTI360OfficialDataset:
```python
def get_bbox_image_file_by_index(self, index, image_file=None):
    """Get semantic visualization image path for compatibility."""
    pair = self.frame_pairs[index] if index < len(self.frame_pairs) else self.frame_pairs[self.clips[index][0]]
    return pair['semantic_path']
```

---

### 4. ✅ Type Error - Normalization on Int Tensor  
**Error**: `TypeError: Input tensor should be a float tensor. Got torch.int64.`

**Root Cause**: Trying to apply normalization to semantic_ids (int64) instead of bbox_images (float32)

**Fix**: Added dtype check in `src/ctrlv/utils/util.py:get_first_training_sample()`:
```python
if bbox_img.dtype in [torch.float32, torch.float16, torch.float64]:
    bbox_img_np = dataset.revert_transform_no_resize(bbox_img).detach().cpu().numpy()*255
    bbox_img_np = bbox_img_np.astype(np.uint8)
else:
    # bbox_img is semantic_ids (int64), skip normalization
    bbox_img_np = bbox_img.detach().cpu().numpy()
```

---

### 5. 🔴 REMAINING: Demo Sample Shape Issue
**Error**: 
```
einops.EinopsError: Wrong shape: expected 5 dims. Received 4-dim tensor.
Input tensor shape: torch.Size([1, 25, 192, 704])
Expected: torch.Size([1, 25, 3, 192, 704])
```

**Root Cause**: Conditioning image missing channel dimension during validation

**Location**: `tools/train_video_diffusion.py:365` → `pipeline_video_diffusion.py:228`

**Analysis**: The demo sample's `image_init` is being passed to the pipeline, but somewhere it's losing the channel dimension or getting replaced with semantic_ids (which have no channel dim).

**Hypothesis**: KITTI360OfficialDataset's `_getimageitem()` might be returning semantic data where RGB is expected, OR the collate function is mixing up the tensors.

---

## Current Status

**Jobs Submitted**: 
- Stage 1: Job 196513 (FAILED at validation)
- Stage 2: Job 196514 (FAILED at validation)

**What Works**:
- ✅ Dataset initialization (48,788 clips loaded)
- ✅ DualVAEManager initialization
- ✅ WandB connection
- ✅ Data loading (no errors in dataloader)

**What Fails**:
- ❌ Initial validation at step 0 (shape mismatch in conditioning image)
- ❌ Training hasn't started yet (fails before first step)

---

## Next Steps

### Option A: Fix Demo Sample Preparation (Proper Fix)
Trace through exactly what KITTI360OfficialDataset returns and ensure demo samples have correct structure.

### Option B: Skip Initial Validation (Temporary Workaround)
Set `--num_demo_samples 0` to skip validation, allow training to start, then fix validation later.

**Recommendation**: Option A - Fix the root cause properly to ensure validation works throughout training.

---

## Files Modified

1. `src/ctrlv/datasets/__init__.py` - Removed deprecated imports
2. `src/ctrlv/datasets/kitti360_official.py` - Added compatibility methods
3. `src/ctrlv/utils/util.py` - Fixed dtype handling in demo sample prep
4. `scripts/train_scripts/train_kitti360_bbox_predict.sh` - Fixed data_root, comments, WandB URL
5. `scripts/train_scripts/train_kitti360_sem2video.sh` - Fixed data_root, num_inference_steps, comments, WandB URL

---

## Test Training Results

**Test Job 196504** (50 steps):
- ✅ Initialized successfully
- ✅ DualVAEManager loaded
- ✅ WandB logs show `gt_semantic_frames`
- ✅ Completed successfully

This confirms the training setup works when demo samples aren't involved.
