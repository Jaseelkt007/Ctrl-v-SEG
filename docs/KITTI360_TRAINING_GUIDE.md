# KITTI360 Bbox Generator Training Guide

This guide explains how to train the bbox prediction diffusion model (first stage) on your preprocessed KITTI360 dataset.

## Dataset Structure

Your preprocessed KITTI360 dataset should be organized as follows:

```
kitti_preprocessed_ctrlv/
├── train/
│   ├── clip_00000/
│   │   ├── frames/
│   │   │   ├── 0000000250.png
│   │   │   └── ...
│   │   ├── bboxes/
│   │   │   ├── 0000000250.png
│   │   │   └── ...
│   │   └── annotations.json
│   ├── clip_00001/
│   └── ...
└── val/
    ├── clip_00000/
    └── ...
```

## Dataset Implementation

The dataset is implemented in `src/ctrlv/datasets/kitti360_preprocessed.py` using the `Kitti360PreprocessedDataset` class.

**Key Features:**
- Inherits from `KittiAbstract` for compatibility with existing training pipeline
- Supports both `image` and `clip` data types
- Loads annotations from JSON files per clip
- Supports preplotted bbox images or on-the-fly bbox drawing
- Compatible with overlapping and non-overlapping clip sampling

## Training Steps

### 1. Verify Dataset

First, test that your dataset loads correctly:

```python
cd /usrhomes/s1492/Ctrl-V

python -c "
from ctrlv.datasets import Kitti360PreprocessedDataset

# Test dataset loading
dataset = Kitti360PreprocessedDataset(
    root='/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/',
    train=True,
    data_type='clip',
    clip_length=8
)

print(f'Dataset size: {len(dataset)}')
print(f'Loading first sample...')
sample = dataset[0]
print(f'Images shape: {sample[0].shape}')
print(f'Number of frames: {len(sample[1])}')
print(f'Frame 0 objects: {len(sample[1][0])}')
print('Dataset loaded successfully!')
"
```

### 2. Configure Training

Edit the config file at `src/ctrlv/bbox_generator_baseline/cfgs/config_kitti360.yaml`:

```yaml
# Key parameters to adjust:
data_root: /no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv
dataset: kitti360_preprocessed
num_timesteps: 8  # Match your clip length
train_batch_size: 2  # Adjust based on GPU memory
max_steps: 70000
```

### 3. Run Training

```bash
cd /usrhomes/s1492/Ctrl-V

# Basic training (without wandb)
python -m ctrlv.bbox_generator_baseline.train \
    --config-name config_kitti360

# With wandb logging
python -m ctrlv.bbox_generator_baseline.train \
    --config-name config_kitti360 \
    wandb_track=True \
    run_name=my_kitti360_run
```

### 4. Training with Custom Parameters

Override config parameters from command line:

```bash
python -m ctrlv.bbox_generator_baseline.train \
    --config-name config_kitti360 \
    train_batch_size=4 \
    num_timesteps=8 \
    lr=1e-4 \
    max_steps=50000
```

## Important Configuration Parameters

### Dataset Parameters
- `data_root`: Path to your preprocessed dataset
- `dataset`: Set to `kitti360_preprocessed`
- `num_timesteps`: Number of frames per clip (should match your preprocessing)
- `dataloader_workers`: Number of data loading workers

### Model Parameters
- `hidden_dim`: Transformer hidden dimension (256)
- `num_decoder_layers`: Number of decoder layers (4)
- `num_encoder_layers`: Number of encoder layers (2)
- `max_num_agents`: Max objects per frame (15)

### Training Parameters
- `train_batch_size`: Batch size for training
- `lr`: Learning rate (5e-4)
- `max_steps`: Total training steps (70000)
- `val_freq`: Validation frequency

### Conditioning Parameters
- `condition_last_frame`: Use last frame as condition
- `initial_frames_condition_num`: Number of initial frames to condition on (3)
- `only_keep_initial_agents`: Only track agents present in initial frames

## Checkpoint Management

Checkpoints are saved to:
```
/home/mila/a/anthony.gosselin/scratch/wandb/diffuser/{run_name}/
```

To resume training from a checkpoint:
```bash
python -m ctrlv.bbox_generator_baseline.train \
    --config-name config_kitti360 \
    run_name=my_previous_run  # Use same run_name to auto-resume
```

## Monitoring Training

### Without W&B
Check console output for:
- Training loss
- Validation loss
- Learning rate
- Steps per second

### With W&B
```bash
# Enable wandb tracking in config
python -m ctrlv.bbox_generator_baseline.train \
    --config-name config_kitti360 \
    wandb_track=True \
    run_name=kitti360_exp1
```

Then monitor at: https://wandb.ai/

## Troubleshooting

### Dataset Not Found
```
Error: Dataset not found at /no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/
```
**Solution:** Check that `data_root` in config matches your actual dataset path.

### Clip Length Mismatch
```
Error: clip length X is too long for clip folder length Y
```
**Solution:** Reduce `num_timesteps` in config to match or be less than your actual clip lengths.

### Out of Memory
```
Error: CUDA out of memory
```
**Solutions:**
- Reduce `train_batch_size` (e.g., from 2 to 1)
- Reduce `num_timesteps` (fewer frames per clip)
- Reduce `hidden_dim` or `num_decoder_layers`

### Missing Bbox Images
```
Error: File not found: bboxes/XXXXXXX.png
```
**Solution:** Set `load_bbox_image=False` in config if you don't have preplotted bbox images.

## Next Steps

After training the bbox generator:
1. Evaluate the model on validation set
2. Use trained model as first stage in full Ctrl-V pipeline
3. Train the video diffusion model (second stage) using predicted bboxes

## Dataset Comparison

Your `Kitti360PreprocessedDataset` is similar to `BDD100KDataset`:
- Both use clip-based organization
- Both have JSON annotations per clip
- Both support preplotted bbox images
- Key difference: KITTI360 has 3D bbox annotations, BDD100K is 2D only

## References

- Original KITTI dataset: `src/ctrlv/datasets/kitti.py`
- BDD100K dataset: `src/ctrlv/datasets/bdd100k.py`
- Training script: `src/ctrlv/bbox_generator_baseline/train.py`
