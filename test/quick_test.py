#!/usr/bin/env python
"""Quick test - Just verify dataset loads and one batch works"""

import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V/src')

print("Quick Dataset Test...")

# Test 1: Import
try:
    from ctrlv.datasets import Kitti360PreprocessedDataset
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create dataset
try:
    dataset = Kitti360PreprocessedDataset(
        root='/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/',
        train=True,
        data_type='clip',
        clip_length=8,
        train_H=320,
        train_W=512,
        if_return_bbox_im=True
    )
    print(f"✓ Dataset loaded: {len(dataset)} clips")
except Exception as e:
    print(f"✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load one sample
try:
    sample = dataset[0]
    # Dataset returns: (images, targets, prompt, index, bbox_images)
    images, targets, prompt, index, bbox_images = sample
    print(f"✓ Sample loaded: images {images.shape}, {len(targets)} frames")
    
    # Verify resolution
    assert images.shape == (8, 3, 320, 512), f"Wrong shape: {images.shape}"
    assert bbox_images.shape == (8, 3, 320, 512), f"Wrong bbox shape: {bbox_images.shape}"
    print(f"✓ Resolution correct: 320×512")
    
    # Check value range
    assert images.min() >= -1.01 and images.max() <= 1.01, f"Wrong range: [{images.min()}, {images.max()}]"
    print(f"✓ Values normalized: [{images.min():.2f}, {images.max():.2f}]")
    
    # Check annotations
    frame0 = targets[0]
    print(f"✓ Frame 0 has {len(frame0)} objects")
    if len(frame0) > 0:
        obj = frame0[0]
        print(f"  - First object: {obj['type']} at {obj['bbox']}")
    
except Exception as e:
    print(f"✗ Sample loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ ALL QUICK TESTS PASSED! ✓✓✓")
print("Run full test: python test_training_setup.py")
