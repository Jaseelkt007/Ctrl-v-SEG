#!/usr/bin/env python
"""
Test script for Kitti360PreprocessedDataset
Run this to verify your dataset is working correctly before training.

Usage:
    python test_kitti360_dataset.py
"""

import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V/src')

from ctrlv.datasets import Kitti360PreprocessedDataset


def test_dataset():
    """Test loading the KITTI360 preprocessed dataset."""
    
    print("=" * 60)
    print("Testing Kitti360PreprocessedDataset")
    print("=" * 60)
    
    # Test parameters
    data_root = '/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/'
    clip_length = 8
    
    print(f"\nDataset configuration:")
    print(f"  - Root: {data_root}")
    print(f"  - Clip length: {clip_length}")
    print(f"  - Data type: clip")
    
    # Load training dataset
    print("\n[1/5] Loading training dataset...")
    try:
        train_dataset = Kitti360PreprocessedDataset(
            root=data_root,
            train=True,
            data_type='clip',
            clip_length=clip_length,
            if_return_bbox_im=False
        )
        print(f"✓ Training dataset loaded")
        print(f"  - Number of clips: {len(train_dataset)}")
        print(f"  - Number of clip folders: {len(train_dataset.clip_folders)}")
    except Exception as e:
        print(f"✗ Failed to load training dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load validation dataset
    print("\n[2/5] Loading validation dataset...")
    try:
        val_dataset = Kitti360PreprocessedDataset(
            root=data_root,
            train=False,
            data_type='clip',
            clip_length=clip_length,
            if_return_bbox_im=False
        )
        print(f"✓ Validation dataset loaded")
        print(f"  - Number of clips: {len(val_dataset)}")
    except Exception as e:
        print(f"✗ Failed to load validation dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading a sample
    print("\n[3/5] Loading first training sample...")
    try:
        sample = train_dataset[0]
        images, targets = sample[0], sample[1]
        
        print(f"✓ Sample loaded successfully")
        print(f"  - Images tensor shape: {images.shape}")
        print(f"  - Expected shape: [clip_length, 3, H, W]")
        print(f"  - Number of frames: {len(targets)}")
        print(f"  - Clip length matches: {len(targets) == clip_length}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check annotations
    print("\n[4/5] Checking annotations...")
    try:
        total_objects = sum(len(frame_targets) for frame_targets in targets)
        print(f"✓ Annotations parsed successfully")
        print(f"  - Total objects in clip: {total_objects}")
        
        # Show first frame details
        frame_0_targets = targets[0]
        print(f"  - Frame 0 objects: {len(frame_0_targets)}")
        
        if len(frame_0_targets) > 0:
            obj = frame_0_targets[0]
            print(f"\n  First object details:")
            print(f"    - Type: {obj['type']}")
            print(f"    - Track ID: {obj['trackID']}")
            print(f"    - BBox 2D: {obj['bbox']}")
            print(f"    - Dimensions: {obj['dimensions']}")
            print(f"    - Location: {obj['location']}")
    except Exception as e:
        print(f"✗ Failed to check annotations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with dataloader
    print("\n[5/5] Testing with DataLoader...")
    try:
        from torch.utils.data import DataLoader
        from ctrlv.datasets import kitti_clip_collate_fn
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: kitti_clip_collate_fn(x, None),
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print(f"✓ DataLoader working")
        print(f"  - Batch clips shape: {batch['clips'].shape}")
        print(f"  - Batch size: {batch['clips'].shape[0]}")
        print(f"  - Objects keys: {list(batch['objects'].keys())}")
        
    except Exception as e:
        print(f"✗ Failed DataLoader test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Dataset is ready for training.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review config: src/ctrlv/bbox_generator_baseline/cfgs/config_kitti360.yaml")
    print("  2. Start training: python -m ctrlv.bbox_generator_baseline.train --config-name config_kitti360")
    print("  3. Read guide: docs/KITTI360_TRAINING_GUIDE.md")
    
    return True


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
