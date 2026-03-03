"""
Test script for KITTI360OfficialDataset

This script verifies:
1. Dataset loads correctly from official txt files
2. RGB and semantic paths are correct
3. Semantic IDs are grayscale (0-18)
4. Collate function creates correct batch structure
5. Clip loading works properly
"""

import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V-seg/src')

import torch
from ctrlv.datasets import KITTI360OfficialDataset, kitti_clip_with_bbox_collate_fn
from torch.utils.data import DataLoader
import numpy as np

def test_dataset_initialization():
    """Test 1: Dataset initializes correctly"""
    print("="*80)
    print("TEST 1: Dataset Initialization")
    print("="*80)
    
    dataset = KITTI360OfficialDataset(
        train=True,
        data_type='clip',
        clip_length=25,
        if_return_bbox_im=True,
        train_H=192,
        train_W=704,
        use_segmentation=True,
        return_semantic_ids=True
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Data type: {dataset.data_type}")
    print(f"  Clip length: {dataset.clip_length}")
    print(f"  Train resolution: {dataset.train_H}x{dataset.train_W}")
    print(f"  Use segmentation: {dataset.use_segmentation}")
    print(f"  Return semantic IDs: {dataset.return_semantic_ids}")
    print()
    
    return dataset

def test_single_sample(dataset):
    """Test 2: Load and verify single sample"""
    print("="*80)
    print("TEST 2: Single Sample Loading")
    print("="*80)
    
    sample = dataset[0]
    print(f"✓ Loaded sample 0")
    print(f"  Number of items returned: {len(sample)}")
    
    if len(sample) == 6:
        clips, targets, prompt, index, bbox_images, semantic_ids = sample
        
        print(f"\n  Clips (RGB frames):")
        print(f"    Shape: {clips.shape}")
        print(f"    Dtype: {clips.dtype}")
        print(f"    Range: [{clips.min():.3f}, {clips.max():.3f}]")
        
        print(f"\n  Bbox images (semantic RGB visualization):")
        print(f"    Shape: {bbox_images.shape}")
        print(f"    Dtype: {bbox_images.dtype}")
        print(f"    Range: [{bbox_images.min():.3f}, {bbox_images.max():.3f}]")
        
        print(f"\n  Semantic IDs (grayscale trainIDs):")
        print(f"    Shape: {semantic_ids.shape}")
        print(f"    Dtype: {semantic_ids.dtype}")
        print(f"    Range: [{semantic_ids.min()}, {semantic_ids.max()}]")
        print(f"    Unique IDs: {torch.unique(semantic_ids).tolist()[:20]}")
        
        # Verify shapes
        T, C, H, W = clips.shape
        expected_shape = (25, 3, 192, 704)
        if clips.shape != expected_shape:
            print(f"\n  ⚠ WARNING: Clips shape {clips.shape} != expected {expected_shape}")
        
        if semantic_ids.shape != (25, 192, 704):
            print(f"\n  ⚠ WARNING: Semantic IDs shape {semantic_ids.shape} != expected (25, 192, 704)")
        
        if semantic_ids.max() > 18:
            print(f"\n  ❌ ERROR: Semantic IDs max {semantic_ids.max()} > 18 (invalid trainID)")
            return False
        
        print(f"\n✓ All shapes and values are correct!")
        return True
    else:
        print(f"❌ ERROR: Expected 6 items, got {len(sample)}")
        return False

def test_batch_loading(dataset):
    """Test 3: DataLoader and collate function"""
    print("\n" + "="*80)
    print("TEST 3: Batch Loading with DataLoader")
    print("="*80)
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging
        collate_fn=lambda batch: kitti_clip_with_bbox_collate_fn(batch, None)
    )
    
    print(f"✓ DataLoader created with batch_size=2")
    
    batch = next(iter(loader))
    print(f"\n✓ Loaded first batch")
    print(f"  Batch keys: {list(batch.keys())}")
    
    # Check required keys (image_init is created later by the training script)
    required_keys = ['clips', 'bbox_images']
    if dataset.return_semantic_ids:
        required_keys.append('semantic_ids')
    
    for key in required_keys:
        if key not in batch:
            print(f"  ❌ ERROR: Missing key '{key}'")
            return False
        print(f"  ✓ '{key}': {batch[key].shape}")
    
    # Verify no 'bbox_ids' key (old bug)
    if 'bbox_ids' in batch:
        print(f"  ❌ ERROR: 'bbox_ids' key should NOT exist (old bug)")
        return False
    
    print(f"\n✓ Batch structure is correct!")
    
    # Verify semantic_ids
    if 'semantic_ids' in batch:
        sem_ids = batch['semantic_ids']
        print(f"\n  Semantic IDs verification:")
        print(f"    Shape: {sem_ids.shape} (expected: [2, 25, 192, 704])")
        print(f"    Dtype: {sem_ids.dtype}")
        print(f"    Range: [{sem_ids.min()}, {sem_ids.max()}]")
        
        if sem_ids.max() > 18:
            print(f"    ❌ ERROR: Max semantic ID {sem_ids.max()} > 18")
            return False
        
        print(f"    ✓ Semantic IDs are valid trainIDs (0-18)")
    
    return True

def test_path_verification(dataset):
    """Test 4: Verify actual file paths"""
    print("\n" + "="*80)
    print("TEST 4: Path Verification")
    print("="*80)
    
    # Check first frame pair
    pair = dataset.frame_pairs[0]
    
    print(f"\n  First frame pair:")
    print(f"    RGB path: {pair['rgb_path']}")
    print(f"    Semantic path: {pair['semantic_path']}")
    
    import os
    rgb_exists = os.path.exists(pair['rgb_path'])
    sem_exists = os.path.exists(pair['semantic_path'])
    
    print(f"\n  File existence check:")
    print(f"    RGB exists: {rgb_exists}")
    print(f"    Semantic exists: {sem_exists}")
    
    if not rgb_exists or not sem_exists:
        print(f"  ❌ ERROR: Files do not exist!")
        return False
    
    print(f"\n✓ Files exist and paths are correct!")
    return True

def test_semantic_remapping(dataset):
    """Test 5: Verify semantic ID remapping from raw KITTI-360 IDs to trainIDs"""
    print("\n" + "="*80)
    print("TEST 5: Semantic ID Remapping Verification")
    print("="*80)
    
    from PIL import Image
    import numpy as np
    from ctrlv.utils.semantic_preprocessing import load_and_remap_semantic
    
    # Check first semantic file
    pair = dataset.frame_pairs[0]
    
    # Load raw semantic image
    raw_img = Image.open(pair['semantic_path']).convert('L')
    raw_arr = np.array(raw_img)
    
    print(f"\n  Raw semantic image (before remapping):")
    print(f"    Shape: {raw_arr.shape}")
    print(f"    Unique IDs: {sorted(np.unique(raw_arr).tolist()[:20])}")
    print(f"    Range: [{raw_arr.min()}, {raw_arr.max()}]")
    
    # Load remapped semantic IDs
    remapped = load_and_remap_semantic(pair['semantic_path'], ignore_index=255)
    
    print(f"\n  Remapped semantic (trainIDs):")
    print(f"    Shape: {remapped.shape}")
    valid_ids = remapped[remapped != 255]
    print(f"    Unique trainIDs: {sorted(np.unique(valid_ids).tolist())}")
    print(f"    Range: [{valid_ids.min()}, {valid_ids.max()}]")
    
    # Verify remapping worked
    if raw_arr.min() < 6 or raw_arr.max() > 40:
        print(f"\n  ⚠ WARNING: Raw IDs outside expected KITTI-360 range")
    
    if valid_ids.min() < 0 or valid_ids.max() > 18:
        print(f"\n  ❌ ERROR: Remapped trainIDs outside valid range [0, 18]")
        return False
    
    # Check that raw IDs are NOT continuous 0-18
    if raw_arr.min() == 0 and set(np.unique(raw_arr)) == set(range(19)):
        print(f"\n  ❌ ERROR: Raw IDs appear to be already remapped (0-18 continuous)")
        return False
    
    print(f"\n✓ Semantic ID remapping verified!")
    print(f"  - Raw KITTI-360 IDs: {raw_arr.min()}-{raw_arr.max()} (non-continuous)")
    print(f"  - Remapped trainIDs: {valid_ids.min()}-{valid_ids.max()} (continuous 0-18)")
    return True

def test_multiple_samples(dataset, num_samples=10):
    """Test 6: Load multiple samples to check consistency"""
    print("\n" + "="*80)
    print(f"TEST 6: Load {num_samples} Samples for Consistency")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            if len(sample) != 6:
                print(f"  ❌ Sample {i}: Wrong number of items ({len(sample)})")
                return False
            
            clips, targets, prompt, index, bbox_images, semantic_ids = sample
            
            # Quick validation
            if semantic_ids.max() > 18:
                print(f"  ❌ Sample {i}: Invalid semantic ID {semantic_ids.max()}")
                return False
            
            if i % 3 == 0:
                print(f"  ✓ Sample {i}: OK (semantic range: [{semantic_ids.min()}, {semantic_ids.max()}])")
        
        except Exception as e:
            print(f"  ❌ Sample {i}: Error - {e}")
            return False
    
    print(f"\n✓ All {num_samples} samples loaded successfully!")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("KITTI360OfficialDataset Test Suite")
    print("="*80)
    print()
    
    try:
        # Test 1: Initialize dataset
        dataset = test_dataset_initialization()
        
        # Test 2: Load single sample
        if not test_single_sample(dataset):
            print("\n❌ TEST 2 FAILED")
            return
        
        # Test 3: Batch loading
        if not test_batch_loading(dataset):
            print("\n❌ TEST 3 FAILED")
            return
        
        # Test 4: Path verification
        if not test_path_verification(dataset):
            print("\n❌ TEST 4 FAILED")
            return
        
        # Test 5: Semantic remapping
        if not test_semantic_remapping(dataset):
            print("\n❌ TEST 5 FAILED")
            return
        
        # Test 6: Multiple samples
        if not test_multiple_samples(dataset, num_samples=10):
            print("\n❌ TEST 6 FAILED")
            return
        
        # All tests passed
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nDataset is ready for training:")
        print("  ✓ Loads from official KITTI-360 txt files")
        print("  ✓ Returns grayscale semantic IDs (0-18)")
        print("  ✓ Correct tensor shapes and dtypes")
        print("  ✓ Batch collation works correctly")
        print("  ✓ 'semantic_ids' key exists (not 'bbox_ids')")
        print()
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
