# KITTI-360 Segmentation Mode Documentation

## Overview

This document explains how the `--use_segmentation` flag changes the conditioning input from bounding boxes to semantic segmentation maps in the Ctrl-V evaluation pipeline.

---

## Table of Contents

1. [Path Switching Mechanism](#path-switching-mechanism)
2. [Directory Structure](#directory-structure)
3. [Code Flow](#code-flow)
4. [Visual Explanation](#visual-explanation)
5. [Verification](#verification)

---

## Path Switching Mechanism

### Key Configuration Point

The path switching happens in **`BDD100KDataset.get_bbox_image_file_by_index()`** method:

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/datasets/bdd100k.py
# Lines: 220-226

def get_bbox_image_file_by_index(self, index=None, image_file=None):
    if image_file is None:
        image_file = self.get_image_file_by_index(index)
    
    if self.use_segmentation:   
        # Use dedicated semantics directory for KITTI-360 semantic RGB images
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_SEMANTIC_DIR)
    
    return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_BBOX_DIR)
```

### Path Constants

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/datasets/bdd100k.py
# Lines: 45-50

TO_IMAGE_DIR = 'images/track'           # Original RGB frames
TO_BBOX_DIR = 'bboxes/track'            # Bounding box visualizations
TO_SEMANTIC_DIR = 'semantics/track'     # Semantic segmentation maps (RGB)
```

### Path Transformation Examples

**Example 1: Bounding Box Mode (`--use_segmentation` NOT set)**
```
Input image path:
/no_backups/s1492/kitti360_ctrlv/images/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000007.png

Conditioning path (bbox):
/no_backups/s1492/kitti360_ctrlv/bboxes/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000007.png
```

**Example 2: Semantic Mode (`--use_segmentation` enabled)**
```
Input image path:
/no_backups/s1492/kitti360_ctrlv/images/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000007.png

Conditioning path (semantic):
/no_backups/s1492/kitti360_ctrlv/semantics/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000007.png
```

---

## Directory Structure

```
/no_backups/s1492/kitti360_ctrlv/
├── images/track/
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync_0000/
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000001.png  # Original RGB frame
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000002.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
│
├── bboxes/track/                    # Used when use_segmentation=False
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync_0000/
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000001.png  # RGB with bbox overlays
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000002.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
│
├── semantics/track/                 # Used when use_segmentation=True
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync_0000/
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000001.png  # RGB semantic map
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000002.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
│
└── labels/
    ├── box_track_20/                # Bounding box annotations (JSON)
    └── seg_track_20/colormaps/      # Semantic segmentation annotations
```

---

## Code Flow

### 1. Script Execution

```bash
# File: /usrhomes/s1492/Ctrl-V-seg/scripts/eval_scripts/eval_kitti360_sem_overall.sh

accelerate launch tools/eval_overall.py \
    --dataset_name kitti360 \
    --use_segmentation \          # ← This flag triggers semantic mode
    ...
```

### 2. Argument Parsing

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/utils/parser.py
# Line: 431-432

if args.use_segmentation:
    assert args.dataset_name in ["bdd100k", "davis", "kitti360"], \
        "Segmentation is only supported for bdd100k, davis, and kitti360 datasets."
```

### 3. Dataset Initialization

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/utils/util.py
# Lines: 68-75

elif dset_name.lower() == 'kitti360':
    from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset
    if use_segmentation:
        use_preplotted_bbox = True  # Force use of preplotted images
    dset = KITTI360BDDDataset(
        root=dset_root, 
        train=if_train, 
        data_type=data_type, 
        clip_length=clip_length, 
        if_return_bbox_im=if_return_bbox_im, 
        train_H=train_H, 
        train_W=train_W, 
        use_segmentation=use_segmentation,  # ← Passed to dataset
        use_preplotted_bbox=use_preplotted_bbox
    )
```

### 4. Dataset Stores Flag

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/datasets/kitti360_bdd_format.py
# Line: 118

self.use_segmentation = use_segmentation  # ← Stored as instance variable
```

### 5. Data Loading (Per Sample)

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/datasets/bdd100k.py
# Lines: 133-139

if return_bbox_im or self.if_return_bbox_im:
    if self.use_preplotted_bbox or self.use_segmentation:
        bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
        bbox_im = Image.open(bbox_file)  # ← Loads from semantics/ or bboxes/
        if self.use_segmentation:
            bbox_im = bbox_im.convert('RGB')  # Ensure RGB format
        if not self.transform is None:
            bbox_im = self.transform(bbox_im)
```

### 6. Path Resolution

```python
# File: /usrhomes/s1492/Ctrl-V-seg/src/ctrlv/datasets/bdd100k.py
# Lines: 220-226

def get_bbox_image_file_by_index(self, index=None, image_file=None):
    if image_file is None:
        image_file = self.get_image_file_by_index(index)
    
    if self.use_segmentation:   
        # Replace 'images/track' with 'semantics/track'
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, 
                                  BDD100KDataset.TO_SEMANTIC_DIR)
    
    # Replace 'images/track' with 'bboxes/track'
    return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, 
                              BDD100KDataset.TO_BBOX_DIR)
```

### 7. Pipeline Conditioning

```python
# File: /usrhomes/s1492/Ctrl-V-seg/tools/eval_overall.py
# Lines: 85-95

# Stage 1: Semantic/BBox Prediction
bbox_im = bbox_pipeline(
    sample['image_init'], 
    height=dataset.train_H, 
    width=dataset.train_W, 
    bbox_images=sample_bbox.unsqueeze(0),  # ← Contains semantic or bbox images
    ...
).frames[0]

# Stage 2: Video Generation (conditioned on Stage 1 output)
frames = ctrl_pipeline(
    sample['image_init'], 
    cond_images=2*(best_generation_bbox-0.5).unsqueeze(0),  # ← Stage 1 output
    ...
).frames[0]
```

---

## Visual Explanation

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CTRL-V TWO-STAGE PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: First Frame (RGB)
   ↓
   ├─────────────────────────────────────────────────────────────────────┐
   │                                                                       │
   │  STAGE 1: Semantic/BBox Predictor                                   │
   │  ┌─────────────────────────────────────────────────────────────┐   │
   │  │                                                               │   │
   │  │  Conditioning Input (depends on --use_segmentation flag):    │   │
   │  │                                                               │   │
   │  │  ┌───────────────────────┐  OR  ┌──────────────────────────┐│   │
   │  │  │ BOUNDING BOX MODE     │      │ SEMANTIC MODE            ││   │
   │  │  │ (use_segmentation=0)  │      │ (use_segmentation=1)     ││   │
   │  │  ├───────────────────────┤      ├──────────────────────────┤│   │
   │  │  │ Path:                 │      │ Path:                    ││   │
   │  │  │ kitti360_ctrlv/       │      │ kitti360_ctrlv/          ││   │
   │  │  │   bboxes/track/val/   │      │   semantics/track/val/   ││   │
   │  │  │                       │      │                          ││   │
   │  │  │ Content:              │      │ Content:                 ││   │
   │  │  │ - RGB image           │      │ - RGB semantic map       ││   │
   │  │  │ - Colored bboxes      │      │ - Colored segmentation   ││   │
   │  │  │ - Background visible  │      │ - Per-pixel labels       ││   │
   │  │  └───────────────────────┘      └──────────────────────────┘│   │
   │  │                                                               │   │
   │  └─────────────────────────────────────────────────────────────┘   │
   │                                                                       │
   │  Output: Predicted semantic/bbox masks (25 frames)                  │
   │                                                                       │
   └───────────────────────────────────┬───────────────────────────────────┘
                                       ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                       │
   │  STAGE 2: Semantic2Video Generator                                  │
   │  ┌─────────────────────────────────────────────────────────────┐   │
   │  │                                                               │   │
   │  │  Input:                                                       │   │
   │  │  - First frame (RGB)                                         │   │
   │  │  - Stage 1 predicted masks (25 frames)                       │   │
   │  │                                                               │   │
   │  │  Output:                                                      │   │
   │  │  - Generated RGB video (25 frames)                           │   │
   │  │                                                               │   │
   │  └─────────────────────────────────────────────────────────────┘   │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
                                       ↓
OUTPUT: Generated Video Sequence (25 RGB frames)
```

### Data Flow Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    BOUNDING BOX MODE (Default)                            │
└──────────────────────────────────────────────────────────────────────────┘

Frame 0001:
  images/track/val/.../0000001.png  ──┐
                                      │ (String replace)
  bboxes/track/val/.../0000001.png  ──┘
  
  ┌─────────────────┐     ┌─────────────────┐
  │  Original RGB   │     │  BBox Overlay   │
  │                 │     │  ┌─────┐        │
  │                 │     │  │ Car │        │
  │    [Scene]      │ →   │  └─────┘        │
  │                 │     │     ┌──┐        │
  │                 │     │     │🚶│        │
  └─────────────────┘     └─────────────────┘
                                ↓
                          Stage 1 Input


┌──────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC MODE (--use_segmentation)                     │
└──────────────────────────────────────────────────────────────────────────┘

Frame 0001:
  images/track/val/.../0000001.png  ──┐
                                      │ (String replace)
  semantics/track/val/.../0000001.png ┘
  
  ┌─────────────────┐     ┌─────────────────┐
  │  Original RGB   │     │  Semantic Map   │
  │                 │     │  ████████████   │ ← Road (gray)
  │                 │     │  ████▓▓██████   │ ← Car (blue)
  │    [Scene]      │ →   │  ████████████   │
  │                 │     │  ████░░██████   │ ← Person (red)
  │                 │     │  ████████████   │
  └─────────────────┘     └─────────────────┘
                                ↓
                          Stage 1 Input
```

### Naming Convention in Outputs

When `--use_segmentation` is enabled, the logged outputs use semantic-specific suffixes:

```
┌────────────────────────────────────────────────────────────────────┐
│                    LOGGED OUTPUT NAMING                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Original Frame:                                                   │
│    2013_05_28_drive_0000_sync_0000-0000007.png                    │
│                                                                     │
│  ┌─────────────────────────┬────────────────────────────────────┐ │
│  │  BBOX MODE              │  SEMANTIC MODE                     │ │
│  │  (use_segmentation=0)   │  (use_segmentation=1)              │ │
│  ├─────────────────────────┼────────────────────────────────────┤ │
│  │  Stage 1 Prediction:    │  Stage 1 Prediction:               │ │
│  │  ...-0000007_predbbox   │  ...-0000007_pred_sem.png          │ │
│  │                         │                                    │ │
│  │  Stage 2 Generated:     │  Stage 2 Generated:                │ │
│  │  ...-0000007_generated  │  ...-0000007_generated.png         │ │
│  │                         │                                    │ │
│  │  Ground Truth RGB:      │  Ground Truth RGB:                 │ │
│  │  ...-0000007_gt.png     │  ...-0000007_gt.png                │ │
│  │                         │                                    │ │
│  │  Ground Truth Cond:     │  Ground Truth Cond:                │ │
│  │  ...-0000007_gt_bbox    │  ...-0000007_gt_sem.png            │ │
│  │                         │                                    │ │
│  │  Video Logs:            │  Video Logs:                       │ │
│  │  scene_0_stage1_pred_   │  scene_0_stage1_pred_sem_video.gif │ │
│  │    bbox_video.gif       │  scene_0_gt_sem_video.gif          │ │
│  │  scene_0_gt_bbox_video  │                                    │ │
│  └─────────────────────────┴────────────────────────────────────┘ │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Verification

### Check Current Configuration

Run this command to verify the paths are correctly configured:

```bash
cd /usrhomes/s1492/Ctrl-V-seg
python3 << 'EOF'
from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset

# Test with segmentation enabled
dataset_sem = KITTI360BDDDataset(
    root='/no_backups/s1492/',
    train=False,
    data_type='clip',
    clip_length=25,
    if_return_bbox_im=True,
    use_segmentation=True,  # ← SEMANTIC MODE
    use_preplotted_bbox=True
)

# Test with segmentation disabled
dataset_bbox = KITTI360BDDDataset(
    root='/no_backups/s1492/',
    train=False,
    data_type='clip',
    clip_length=25,
    if_return_bbox_im=True,
    use_segmentation=False,  # ← BBOX MODE
    use_preplotted_bbox=True
)

# Get sample paths
image_file = dataset_sem.get_image_file_by_index(0)
sem_file = dataset_sem.get_bbox_image_file_by_index(image_file=image_file)
bbox_file = dataset_bbox.get_bbox_image_file_by_index(image_file=image_file)

print("=" * 80)
print("PATH VERIFICATION")
print("=" * 80)
print(f"\nOriginal Image Path:\n  {image_file}")
print(f"\nSemantic Mode Path (use_segmentation=True):\n  {sem_file}")
print(f"\nBBox Mode Path (use_segmentation=False):\n  {bbox_file}")
print("\n" + "=" * 80)

# Verify files exist
import os
print("\nFILE EXISTENCE CHECK:")
print(f"  Original image exists: {os.path.exists(image_file)}")
print(f"  Semantic file exists:  {os.path.exists(sem_file)}")
print(f"  BBox file exists:      {os.path.exists(bbox_file)}")
print("=" * 80)
EOF
```

### Expected Output

```
================================================================================
PATH VERIFICATION
================================================================================

Original Image Path:
  /no_backups/s1492/kitti360_ctrlv/images/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000001.png

Semantic Mode Path (use_segmentation=True):
  /no_backups/s1492/kitti360_ctrlv/semantics/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000001.png

BBox Mode Path (use_segmentation=False):
  /no_backups/s1492/kitti360_ctrlv/bboxes/track/val/2013_05_28_drive_0000_sync_0000/2013_05_28_drive_0000_sync_0000-0000001.png

================================================================================

FILE EXISTENCE CHECK:
  Original image exists: True
  Semantic file exists:  True
  BBox file exists:      True
================================================================================
```

### Directory Structure Verification

```bash
# Check all three directories exist with same structure
ls -la /no_backups/s1492/kitti360_ctrlv/images/track/val/ | head -10
ls -la /no_backups/s1492/kitti360_ctrlv/bboxes/track/val/ | head -10
ls -la /no_backups/s1492/kitti360_ctrlv/semantics/track/val/ | head -10

# Verify a specific scene has files in all three locations
SCENE="2013_05_28_drive_0000_sync_0000"
echo "Images:    $(ls /no_backups/s1492/kitti360_ctrlv/images/track/val/$SCENE/ | wc -l) files"
echo "BBoxes:    $(ls /no_backups/s1492/kitti360_ctrlv/bboxes/track/val/$SCENE/ | wc -l) files"
echo "Semantics: $(ls /no_backups/s1492/kitti360_ctrlv/semantics/track/val/$SCENE/ | wc -l) files"
```

---

## Summary

### Key Points

1. **Flag**: `--use_segmentation` in the evaluation script
2. **Switch Location**: `BDD100KDataset.get_bbox_image_file_by_index()` method
3. **Path Change**: `images/track` → `semantics/track` (instead of `bboxes/track`)
4. **Content**: RGB semantic segmentation maps (not bounding box overlays)
5. **Naming**: Outputs use `_pred_sem.png` and `_gt_sem.png` suffixes

### Configuration Status ✅

- ✅ Parser allows `kitti360` for segmentation mode
- ✅ Dataset initialization passes `use_segmentation` flag
- ✅ Path switching logic is implemented correctly
- ✅ Semantic directory exists at `/no_backups/s1492/kitti360_ctrlv/semantics/`
- ✅ Naming conventions distinguish semantic vs bbox outputs

### Verification Commands

```bash
# Quick verification
cd /usrhomes/s1492/Ctrl-V-seg
python3 -c "from ctrlv.datasets.bdd100k import BDD100KDataset; print('TO_SEMANTIC_DIR:', BDD100KDataset.TO_SEMANTIC_DIR)"

# Full path test (run the verification script above)
```

---

**Last Updated**: December 3, 2025  
**Verified For**: Ctrl-V-seg directory (`/usrhomes/s1492/Ctrl-V-seg/`)
