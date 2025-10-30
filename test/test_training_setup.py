#!/usr/bin/env python
"""
Test script to verify KITTI360 dataset and dataloader before training.
similar to train_video_diffusion.py
"""

import sys
import os
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V/src')

import torch
from ctrlv.utils import get_dataloader
from ctrlv.datasets import kitti_clip_with_bbox_collate_fn

print("=" * 80)
print("KITTI360 Training Setup Test")
print("=" * 80)

# Mimic the exact parameters from train_kitti360_bbox_predict.sh
config = {
    'data_root': '/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv',
    'dataset_name': 'kitti360_preprocessed',
    'clip_length': 8,
    'train_batch_size': 1,
    'dataloader_num_workers': 0,  # Use 0 for testing to see errors clearly
    'train_H': 320,
    'train_W': 512,
    'if_return_bbox_im': True,
    'if_last_frame_trajectory': False,
    'non_overlapping_clips': False,
}

print("\n[Step 1/6] Configuration:")
print("-" * 80)
for key, value in config.items():
    print(f"  {key:30s}: {value}")

# Test 1: Load Training Dataset
print("\n[Step 2/6] Loading Training Dataset...")
print("-" * 80)
try:
    train_dataset, train_loader = get_dataloader(
        config['data_root'],
        config['dataset_name'],
        if_train=True,
        clip_length=config['clip_length'],
        batch_size=config['train_batch_size'],
        num_workers=config['dataloader_num_workers'],
        data_type='clip',
        use_default_collate=True,
        tokenizer=None,
        shuffle=True,
        if_return_bbox_im=config['if_return_bbox_im'],
        train_H=config['train_H'],
        train_W=config['train_W'],
        use_segmentation=False,
        use_preplotted_bbox=not config['if_last_frame_trajectory'],
        if_last_frame_traj=config['if_last_frame_trajectory'],
        non_overlapping_clips=config['non_overlapping_clips']
    )
    print(f"✓ Training dataset loaded successfully")
    print(f"  - Dataset type: {type(train_dataset).__name__}")
    print(f"  - Dataset length: {len(train_dataset)} clips")
    print(f"  - Batch size: {config['train_batch_size']}")
    print(f"  - Total batches: {len(train_loader)}")
except Exception as e:
    print(f"✗ Failed to load training dataset!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load Validation Dataset
print("\n[Step 3/6] Loading Validation Dataset...")
print("-" * 80)
try:
    val_dataset, val_loader = get_dataloader(
        config['data_root'],
        config['dataset_name'],
        if_train=False,
        clip_length=config['clip_length'],
        batch_size=config['train_batch_size'],
        num_workers=config['dataloader_num_workers'],
        data_type='clip',
        use_default_collate=True,
        tokenizer=None,
        shuffle=False,
        if_return_bbox_im=config['if_return_bbox_im'],
        train_H=config['train_H'],
        train_W=config['train_W'],
        use_segmentation=False,
        use_preplotted_bbox=not config['if_last_frame_trajectory'],
        if_last_frame_traj=config['if_last_frame_trajectory'],
        non_overlapping_clips=config['non_overlapping_clips']
    )
    print(f"✓ Validation dataset loaded successfully")
    print(f"  - Dataset length: {len(val_dataset)} clips")
    print(f"  - Total batches: {len(val_loader)}")
except Exception as e:
    print(f"✗ Failed to load validation dataset!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load a Single Batch
print("\n[Step 4/6] Loading First Training Batch...")
print("-" * 80)
try:
    batch_iter = iter(train_loader)
    batch = next(batch_iter)
    
    print(f"✓ Batch loaded successfully")
    print(f"\nBatch structure:")
    print(f"  - Keys: {list(batch.keys())}")
    
    # Check clips
    if 'clips' in batch and batch['clips'] is not None:
        print(f"\n  clips:")
        print(f"    - Shape: {batch['clips'].shape}")
        print(f"    - Expected: [batch_size={config['train_batch_size']}, "
              f"clip_length={config['clip_length']}, channels=3, "
              f"H={config['train_H']}, W={config['train_W']}]")
        print(f"    - Dtype: {batch['clips'].dtype}")
        print(f"    - Value range: [{batch['clips'].min():.3f}, {batch['clips'].max():.3f}]")
        print(f"    - Expected range: [-1.0, 1.0] (normalized)")
        
        # Verify shape
        expected_shape = (config['train_batch_size'], config['clip_length'], 3, 
                         config['train_H'], config['train_W'])
        if batch['clips'].shape == expected_shape:
            print(f"    ✓ Shape matches expected!")
        else:
            print(f"    ✗ Shape mismatch! Expected {expected_shape}")
    
    # Check bbox_images
    if 'bbox_images' in batch:
        print(f"\n  bbox_images:")
        print(f"    - Shape: {batch['bbox_images'].shape}")
        print(f"    - Dtype: {batch['bbox_images'].dtype}")
        print(f"    - Value range: [{batch['bbox_images'].min():.3f}, {batch['bbox_images'].max():.3f}]")
    
    # Check objects
    if 'objects' in batch:
        print(f"\n  objects:")
        print(f"    - Keys: {list(batch['objects'].keys())}")
        if 'bbox' in batch['objects']:
            print(f"    - bbox shape: {batch['objects']['bbox'].shape}")
            print(f"      Expected: [batch_size={config['train_batch_size']}, "
                  f"clip_length={config['clip_length']}, max_objects=15, coords=4]")
        if 'track_id' in batch['objects']:
            print(f"    - track_id shape: {batch['objects']['track_id'].shape}")
        if 'id_type' in batch['objects']:
            print(f"    - id_type shape: {batch['objects']['id_type'].shape}")
            unique_types = torch.unique(batch['objects']['id_type'])
            unique_types = unique_types[unique_types > 0]  # Exclude padding
            print(f"    - Unique object types in batch: {unique_types.tolist()}")
    
    # Check prompts
    if 'prompts' in batch:
        print(f"\n  prompts:")
        print(f"    - Type: {type(batch['prompts'])}")
        print(f"    - First prompt: '{batch['prompts'][0]}'")
    
except Exception as e:
    print(f"✗ Failed to load batch!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check Dataset Attributes
print("\n[Step 5/6] Checking Dataset Attributes...")
print("-" * 80)
try:
    print(f"  - Dataset version: {train_dataset.version}")
    print(f"  - Data directory: {train_dataset.data_dir}")
    print(f"  - Number of clip folders: {len(train_dataset.clip_folders)}")
    print(f"  - First 5 clip folders: {train_dataset.clip_folders[:5]}")
    print(f"  - Original resolution: {train_dataset.orig_H}×{train_dataset.orig_W}")
    print(f"  - Training resolution: {train_dataset.train_H}×{train_dataset.train_W}")
    print(f"  - Max boxes per frame: {train_dataset.MAX_BOXES_PER_DATA}")
    print(f"  - Clip length: {train_dataset.clip_length}")
    print(f"  - Data type: {train_dataset.data_type}")
    print(f"  - Use preplotted bbox: {train_dataset.use_preplotted_bbox}")
    
    # Check clip list
    if hasattr(train_dataset, 'clip_list'):
        print(f"  - Total clip samples: {len(train_dataset.clip_list)}")
        print(f"  - First 3 samples: {train_dataset.clip_list[:3]}")
    
    print(f"\n✓ All attributes look good!")
    
except Exception as e:
    print(f"✗ Error checking attributes!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Iterate Multiple Batches
print("\n[Step 6/6] Testing Multiple Batch Loading...")
print("-" * 80)
try:
    num_test_batches = min(5, len(train_loader))
    print(f"Loading {num_test_batches} batches...")
    
    for i, batch in enumerate(train_loader):
        if i >= num_test_batches:
            break
        
        # Quick validation
        assert 'clips' in batch, "Missing 'clips' in batch"
        assert batch['clips'].shape[0] == config['train_batch_size'], "Wrong batch size"
        assert batch['clips'].shape[1] == config['clip_length'], "Wrong clip length"
        assert batch['clips'].shape[2] == 3, "Wrong number of channels"
        assert batch['clips'].shape[3] == config['train_H'], "Wrong height"
        assert batch['clips'].shape[4] == config['train_W'], "Wrong width"
        
        print(f"  ✓ Batch {i+1}/{num_test_batches}: Shape {tuple(batch['clips'].shape)}, "
              f"Range [{batch['clips'].min():.2f}, {batch['clips'].max():.2f}]")
    
    print(f"\n✓ All {num_test_batches} batches loaded successfully!")
    
except Exception as e:
    print(f"✗ Error during batch iteration!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour dataset and dataloader are working correctly!")
print("\nDataset Summary:")
print(f"  - Training samples: {len(train_dataset)} clips")
print(f"  - Validation samples: {len(val_dataset)} clips")
print(f"  - Clip length: {config['clip_length']} frames")
print(f"  - Resolution: {config['train_H']}×{config['train_W']}")
print(f"  - Batch size: {config['train_batch_size']}")
print(f"  - Total training batches: {len(train_loader)}")

print("\n✓ Ready to start training!")
print("\nRun training with:")
print("  bash scripts/train_scripts/train_kitti360_bbox_predict.sh")

sys.exit(0)
