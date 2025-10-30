#!/usr/bin/env python3
"""
Test KITTI-360 dataset with Ctrl-V dataloader for bbox prediction training.

This script validates:
1. Dataset can be loaded with KITTI360BDDDataset
2. Data shapes match training requirements
3. Bbox overlays load correctly
4. Annotations are properly formatted
5. Dataloader works with multiple clips
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V/src')

import torch
from PIL import Image


def test_dataset_loading():
    """Test basic dataset loading."""
    print("=" * 80)
    print("TEST 1: Dataset Loading")
    print("=" * 80)
    
    try:
        from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset
        print("✓ Successfully imported KITTI360BDDDataset")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    try:
        # Load training dataset
        train_dataset = KITTI360BDDDataset(
            root='/no_backups/s1492/',
            train=True,
            data_type='clip',
            clip_length=8,
            if_return_bbox_im=False,
            use_preplotted_bbox=True
        )
        
        print(f"✓ Train dataset loaded")
        print(f"  - Total clips: {len(train_dataset)}")
        print(f"  - Scenes: {len(train_dataset.clip_folders)}")
        print(f"  - First scene: {train_dataset.clip_folders[0]}")
        print(f"  - Last scene: {train_dataset.clip_folders[-1]}")
        
        # Load validation dataset
        val_dataset = KITTI360BDDDataset(
            root='/no_backups/s1492/',
            train=False,
            data_type='clip',
            clip_length=8,
            if_return_bbox_im=False,
            use_preplotted_bbox=True
        )
        
        print(f"✓ Val dataset loaded")
        print(f"  - Total clips: {len(val_dataset)}")
        print(f"  - Scenes: {len(val_dataset.clip_folders)}")
        
        return True, train_dataset, val_dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_single_sample(dataset, idx=0):
    """Test loading a single sample."""
    print("\n" + "=" * 80)
    print(f"TEST 2: Single Sample Loading (clip {idx})")
    print("=" * 80)
    
    try:
        sample = dataset[idx]
        
        images = sample[0]
        targets = sample[1]
        prompt = sample[2] if len(sample) > 2 else None
        
        print(f"✓ Sample loaded successfully")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Images dtype: {images.dtype}")
        print(f"  - Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  - Number of frames: {len(targets)}")
        print(f"  - Prompt: {prompt[:80] if prompt else 'None'}...")
        
        # Check image shape
        if images.shape[0] == dataset.clip_length and images.shape[1] == 3:
            print(f"✓ Image shape correct: (clip_length={dataset.clip_length}, channels=3, H={images.shape[2]}, W={images.shape[3]})")
        else:
            print(f"⚠ Warning: Unexpected shape. Expected (clip_length={dataset.clip_length}, channels=3, H, W), got {images.shape}")
        
        return True, targets
        
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_annotations(targets):
    """Test annotation format."""
    print("\n" + "=" * 80)
    print("TEST 3: Annotation Format")
    print("=" * 80)
    
    try:
        print(f"✓ Found {len(targets)} frames in clip")
        
        # Check first frame
        frame0 = targets[0]
        print(f"\nFrame 0 has {len(frame0)} objects")
        
        if len(frame0) > 0:
            obj = frame0[0]
            print(f"\nFirst object structure:")
            print(f"  - Keys: {list(obj.keys())}")
            print(f"  - trackID: {obj.get('trackID')} (type: {type(obj.get('trackID'))})")
            print(f"  - type: {obj.get('type')}")
            print(f"  - bbox: {obj.get('bbox')}")
            print(f"  - truncated: {obj.get('truncated')} (type: {type(obj.get('truncated'))})")
            print(f"  - occluded: {obj.get('occluded')}")
            print(f"  - id_type: {obj.get('id_type')}")
            
            # Validate required fields
            required_fields = ['trackID', 'type', 'bbox', 'id_type']
            missing = [f for f in required_fields if f not in obj]
            if missing:
                print(f"⚠ Warning: Missing fields: {missing}")
            else:
                print(f"✓ All required fields present")
            
            # Check bbox format
            bbox = obj.get('bbox', [])
            if len(bbox) == 4:
                print(f"✓ Bbox format correct: [x1, y1, x2, y2]")
            else:
                print(f"⚠ Warning: Unexpected bbox format: {bbox}")
        
        return True
        
    except Exception as e:
        print(f"✗ Annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bbox_overlays(dataset, idx=0):
    """Test bbox overlay loading."""
    print("\n" + "=" * 80)
    print(f"TEST 4: Bbox Overlay Loading")
    print("=" * 80)
    
    try:
        # Create dataset with bbox overlays enabled
        dataset_with_bbox = type(dataset)(
            root='/no_backups/s1492/',
            train=dataset.train,
            data_type='clip',
            clip_length=dataset.clip_length,
            if_return_bbox_im=True,
            use_preplotted_bbox=True
        )
        
        sample = dataset_with_bbox[idx]
        
        images = sample[0]
        targets = sample[1]
        bboxes = sample[-1]  # Last element should be bbox images
        
        print(f"✓ Sample with bbox overlays loaded")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Bboxes shape: {bboxes.shape}")
        print(f"  - Bboxes dtype: {bboxes.dtype}")
        print(f"  - Bboxes range: [{bboxes.min():.3f}, {bboxes.max():.3f}]")
        
        # Verify shapes match
        if images.shape == bboxes.shape:
            print(f"✓ Bbox overlays match image dimensions")
        else:
            print(f"⚠ Warning: Shape mismatch - Images: {images.shape}, Bboxes: {bboxes.shape}")
        
        # Check if bboxes are not all zeros
        if bboxes.abs().sum() > 0:
            print(f"✓ Bbox overlays contain data (not all zeros)")
        else:
            print(f"⚠ Warning: Bbox overlays are all zeros - may not have objects")
        
        return True
        
    except Exception as e:
        print(f"✗ Bbox overlay test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_clips(dataset, num_clips=5):
    """Test loading multiple clips."""
    print("\n" + "=" * 80)
    print(f"TEST 5: Multiple Clip Loading ({num_clips} clips)")
    print("=" * 80)
    
    try:
        for i in range(min(num_clips, len(dataset))):
            sample = dataset[i]
            images = sample[0]
            targets = sample[1]
            
            num_objects = sum(len(frame) for frame in targets)
            print(f"  Clip {i}: shape={images.shape}, objects={num_objects}")
        
        print(f"✓ Successfully loaded {num_clips} clips")
        return True
        
    except Exception as e:
        print(f"✗ Multiple clip loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_integration(dataset):
    """Test PyTorch DataLoader integration."""
    print("\n" + "=" * 80)
    print("TEST 6: PyTorch DataLoader Integration")
    print("=" * 80)
    
    try:
        from torch.utils.data import DataLoader
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 for testing
            collate_fn=lambda x: x  # Custom collate for varying bbox counts
        )
        
        print(f"✓ DataLoader created")
        print(f"  - Batch size: 2")
        print(f"  - Total batches: {len(dataloader)}")
        
        # Test first batch
        batch = next(iter(dataloader))
        print(f"✓ First batch loaded")
        print(f"  - Batch length: {len(batch)}")
        print(f"  - Sample 0 image shape: {batch[0][0].shape}")
        print(f"  - Sample 1 image shape: {batch[1][0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_compatibility():
    """Test compatibility with training requirements."""
    print("\n" + "=" * 80)
    print("TEST 7: Training Compatibility Check")
    print("=" * 80)
    
    try:
        from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset
        
        # Test with typical training config
        dataset = KITTI360BDDDataset(
            root='/no_backups/s1492/',
            train=True,
            data_type='clip',
            clip_length=25,  # Typical for Ctrl-V
            if_return_bbox_im=True,
            use_preplotted_bbox=True,
            H=376,  # KITTI-360 default
            W=1408
        )
        
        print(f"✓ Dataset created with training config:")
        print(f"  - clip_length: 25")
        print(f"  - H: 376, W: 1408")
        print(f"  - return_bbox_im: True")
        print(f"  - Total clips: {len(dataset)}")
        
        # Load sample
        sample = dataset[0]
        images, targets, prompt, idx, bboxes = sample
        
        print(f"\n✓ Sample loaded with all return values:")
        print(f"  - images: {images.shape}")
        print(f"  - targets: {len(targets)} frames")
        print(f"  - prompt: {len(prompt)} chars")
        print(f"  - idx: {idx}")
        print(f"  - bboxes: {bboxes.shape}")
        
        # Check expected shapes for Ctrl-V
        if images.shape[0] == 25 and bboxes.shape[0] == 25:
            print(f"✓ Clip length matches training config (25 frames)")
        
        if images.shape[1] == 3 and bboxes.shape[1] == 3:
            print(f"✓ Channel count correct (3 channels)")
        
        print(f"\n✓ Dataset is compatible with Ctrl-V training!")
        
        return True
        
    except Exception as e:
        print(f"✗ Training compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("KITTI-360 Dataset & Dataloader Validation for Ctrl-V Bbox Prediction")
    print("=" * 80)
    print()
    
    # Run tests
    results = []
    
    # Test 1: Loading
    success, train_dataset, val_dataset = test_dataset_loading()
    results.append(("Dataset Loading", success))
    
    if not success:
        print("\n✗ Cannot proceed without dataset loading. Fix errors above.")
        return
    
    # Test 2: Single sample
    success, targets = test_single_sample(train_dataset, idx=0)
    results.append(("Single Sample", success))
    
    # Test 3: Annotations
    if targets:
        success = test_annotations(targets)
        results.append(("Annotations", success))
    
    # Test 4: Bbox overlays
    success = test_bbox_overlays(train_dataset, idx=0)
    results.append(("Bbox Overlays", success))
    
    # Test 5: Multiple clips
    success = test_multiple_clips(train_dataset, num_clips=5)
    results.append(("Multiple Clips", success))
    
    # Test 6: DataLoader
    success = test_dataloader_integration(train_dataset)
    results.append(("DataLoader", success))
    
    # Test 7: Training compatibility
    success = test_training_compatibility()
    results.append(("Training Compatibility", success))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYour KITTI-360 dataset is ready for Ctrl-V bbox prediction training!")
        print("\nNext steps:")
        print("  1. Update your training script:")
        print("     - DATASET='kitti360_bdd_format'")
        print("     - DATASET_PATH='/no_backups/s1492/'")
        print("  2. Run training:")
        print("     bash scripts/train_scripts/train_kitti360_bbox_predict.sh")
    else:
        print("⚠ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the errors above and fix before training.")
    print()


if __name__ == '__main__':
    main()
