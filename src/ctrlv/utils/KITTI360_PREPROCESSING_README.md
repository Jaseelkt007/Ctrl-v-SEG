# KITTI-360 Preprocessing for Ctrl-V Training

This guide explains how to preprocess KITTI-360 dataset to match the BDD100K format used in Ctrl-V.

## Overview

The preprocessing pipeline converts KITTI-360's per-frame text annotations into a BDD100K-compatible structure, allowing you to train Ctrl-V without modifying the model code.

## Input Requirements

Your KITTI-360 dataset should have:

1. **Per-frame text annotations**: `/data/public/kitti-360/KITTI-360/data_2d_raw/<SCENE>/label_00/*.txt`
   - Format: `frame trackID type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z rot_y`

2. **Train/Val split files**:
   - `/data/public/kitti-360/KITTI-360/train_images.txt`
   - `/data/public/kitti-360/KITTI-360/val_images.txt`
   - Each line: `data_2d_raw/<SCENE>/image_00/data_rect/<FRAME>.png`

3. **Image files**: `/data/public/kitti-360/KITTI-360/data_2d_raw/<SCENE>/image_00/data_rect/*.png`

## Output Structure

After preprocessing, you'll have:

```
kitti360/
├── images/track/
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync_0000/
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000001.png
│   │   │   ├── 2013_05_28_drive_0000_sync_0000-0000002.png
│   │   │   └── ...
│   │   ├── 2013_05_28_drive_0000_sync_0001/
│   │   └── ...
│   └── val/
│       └── (same structure)
├── bboxes/track/
│   ├── train/
│   │   └── (same structure as images, with bbox overlays)
│   └── val/
├── labels/box_track_20/
│   ├── train/
│   │   ├── 2013_05_28_drive_0000_sync_0000.json
│   │   ├── 2013_05_28_drive_0000_sync_0001.json
│   │   └── ...
│   └── val/
```

This matches the BDD100K structure, so you can use `BDD100KDataset` dataloader directly.

## Preprocessing Steps

### Step 1: Detect Scene Boundaries and Create Dataset Structure

Run the main preprocessing script:

```bash
python preprocess_kitti360_bdd_format.py \
  --kitti_root /data/public/kitti-360/KITTI-360 \
  --output_root /no_backups/s1492/kitti360_ctrlv \
  --frame_jump_threshold 100
```

**Parameters:**
- `--kitti_root`: KITTI-360 dataset root directory
- `--output_root`: Where to save processed dataset
- `--frame_jump_threshold`: Frame number jump to consider a new scene (default: 100)
- `--no_symlink`: Copy images instead of symlinking (slower, use for portability)

**What it does:**
1. Reads `train_images.txt` and `val_images.txt`
2. Detects scene boundaries by finding frame number jumps > threshold
3. Groups continuous frames into scenes (e.g., `2013_05_28_drive_0000_sync_0000`, `_0001`, etc.)
4. Creates directory structure matching BDD100K
5. Symlinks (or copies) images to `images/track/{train,val}/scene_xxxxx/`
6. Parses text annotations and creates JSON files in `labels/box_track_20/{train,val}/`

**Expected output:**
```
TRAIN Split Statistics:
  Total scenes: ~18-20 per sequence
  Total frames: ~11380 for scene 0000
  Scene names: 2013_05_28_drive_0000_sync_0000 ... 2013_05_28_drive_0009_sync_0018
```

### Step 2: Generate Bounding Box Overlays (Optional)

Generate bbox overlay images for training:

```bash
python generate_kitti360_bbox_overlays.py \
  --root /no_backups/s1492/kitti360_ctrlv \
  --mode track \
  --workers 8
```

**Parameters:**
- `--root`: Processed dataset root (output from Step 1)
- `--mode`: 
  - `track`: Colored by track ID (default, solid filled boxes on black)
  - `class`: Colored by object class
  - `binary`: White boxes on black
- `--alpha`: Alpha blending for RGB modes (0.0-1.0)
- `--border`: Border thickness for RGB modes
- `--workers`: Parallel workers (0=sequential, >0=parallel)
- `--png_comp`: PNG compression (0=fastest, 9=smallest)

**What it does:**
1. Reads JSON annotations from `labels/box_track_20/`
2. Generates bbox overlay images
3. Saves to `bboxes/track/{train,val}/scene_xxxxx/`

## Usage in Training

### Option 1: Use BDD100KDataset Directly (Recommended)

Since the structure matches BDD100K, modify your training config:

```python
from ctrlv.datasets.bdd100k import BDD100KDataset

# In your training script
train_dataset = BDD100KDataset(
    root='/no_backups/s1492/',  # Parent of kitti360/
    train=True,
    data_type='clip',
    clip_length=8,
    if_return_bbox_im=True,
    use_preplotted_bbox=True
)
```

**Note:** You may need to adjust paths in `bdd100k.py`:
- Change `self.version = 'bdd100k'` to `self.version = 'kitti360'`
- Or create a symlink: `ln -s kitti360 bdd100k` in your dataset root

### Option 2: Use KITTI360PreprocessedDataset (if you prefer)

The existing `Kitti360PreprocessedDataset` can work, but requires different structure:

```python
from ctrlv.datasets.kitti360_preprocessed import Kitti360PreprocessedDataset

train_dataset = Kitti360PreprocessedDataset(
    root='/no_backups/s1492/kitti360_ctrlv/train',
    train=True,
    data_type='clip',
    clip_length=8
)
```

## Scene Boundary Detection Logic

The script detects scene boundaries using two criteria:

1. **Sequence change**: When the sequence name changes (e.g., `drive_0000` → `drive_0001`)
2. **Frame jump**: When frame numbers jump by > threshold (default: 100)

Example from your `train_images.txt`:
```
...
data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000384.png
data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000385.png
data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000001980.png  ← Jump of 1595!
...
```

This creates two scenes:
- `2013_05_28_drive_0000_sync_0000`: frames 250-385
- `2013_05_28_drive_0000_sync_0001`: frames 1980-...

## Class Mapping

The script maps KITTI-360 object types to BDD100K-compatible class IDs:

```python
CLASS_IDS_LOOKUP = {
    'car': 3,
    'van': 3,        # Mapped to car
    'truck': 4,
    'bus': 5,
    'train': 6,
    'motorcycle': 7,
    'bicycle': 8,
    'caravan': 4,    # Mapped to truck
    'trailer': 4,    # Mapped to truck
    'pedestrian': 1,
    'person': 1,
    'cyclist': 2,
}
```

Only vehicle types are kept by default. To include other objects, modify `VEHICLE_TYPES` in the script.

## Troubleshooting

### Issue: "FileNotFoundError: train_images.txt not found"
**Solution:** Ensure paths are correct:
```bash
ls /data/public/kitti-360/KITTI-360/train_images.txt
ls /data/public/kitti-360/KITTI-360/val_images.txt
```

### Issue: "Cannot read image"
**Solution:** Check image paths in text files match actual file locations:
```bash
# Test one path from train_images.txt
cat /data/public/kitti-360/KITTI-360/train_images.txt | head -1
# Should be: data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png
```

### Issue: Scenes are too short/long
**Solution:** Adjust `--frame_jump_threshold`:
- Lower value → more, shorter scenes
- Higher value → fewer, longer scenes

### Issue: Out of disk space
**Solution:** Use symlinks (default) instead of copying:
```bash
python preprocess_kitti360_bdd_format.py \
  --kitti_root /data/public/kitti-360/KITTI-360 \
  --output_root /no_backups/s1492/kitti360_ctrlv
  # No --no_symlink flag
```

## Variable Clip Length During Training

Unlike the old preprocessing (which fixed `clip_length=8`), this approach gives you flexibility:

1. **At preprocessing**: Scenes are variable length (e.g., 136 frames, 200 frames, etc.)
2. **At training**: You choose `clip_length` in the dataloader config

The dataloader will:
- For training: Use sliding window with overlap (more clips)
- For validation: Use non-overlapping clips

Example:
```python
# Scene has 100 frames, clip_length=8
# Training: Samples clips [0-7], [1-8], [2-9], ..., [92-99] = 93 clips
# Validation: Samples clips [0-7], [8-15], [16-23], ..., [88-95] = 12 clips
```

## Verification

After preprocessing, verify the structure:

```bash
# Check scene count
ls /no_backups/s1492/kitti360_ctrlv/images/track/train/ | wc -l

# Check a scene
ls /no_backups/s1492/kitti360_ctrlv/images/track/train/2013_05_28_drive_0000_sync_0000/

# Check JSON annotation
cat /no_backups/s1492/kitti360_ctrlv/labels/box_track_20/train/2013_05_28_drive_0000_sync_0000.json | head -50

# Check bbox overlays
ls /no_backups/s1492/kitti360_ctrlv/bboxes/track/train/2013_05_28_drive_0000_sync_0000/
```

## Next Steps

1. Run preprocessing (Step 1)
2. Optionally generate bbox overlays (Step 2)
3. Update your training config to point to the new dataset
4. Train Ctrl-V as you would with BDD100K!

## Performance Tips

- **Use symlinks** (default): Saves disk space and preprocessing time
- **Parallel bbox generation**: Use `--workers 8` or more
- **Low PNG compression**: Use `--png_comp 1` for faster bbox generation
- **SSD storage**: Store processed dataset on fast storage for training

## Contact

If you encounter issues, check:
1. File paths and permissions
2. Disk space
3. Python dependencies: `opencv-python`, `numpy`, `tqdm`
