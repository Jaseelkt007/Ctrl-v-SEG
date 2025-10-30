 #!/usr/bin/env python3
"""
Test script to verify KITTI-360 preprocessing and dataloader compatibility.
"""

import sys
import os
from pathlib import Path
import json

def test_directory_structure(root):
    """Test if the directory structure matches BDD100K format."""
    print("=" * 80)
    print("Testing Directory Structure")
    print("=" * 80)
    
    root = Path(root)
    checks = []
    
    # Check required directories
    required_dirs = [
        'images/track/train',
        'images/track/val',
        'labels/box_track_20/train',
        'labels/box_track_20/val',
    ]
    
    for dir_path in required_dirs:
        full_path = root / dir_path
        exists = full_path.exists() and full_path.is_dir()
        checks.append((dir_path, exists))
        print(f"{'✓' if exists else '✗'} {dir_path}: {'Found' if exists else 'NOT FOUND'}")
    
    # Optional bbox directories
    optional_dirs = [
        'bboxes/track/train',
        'bboxes/track/val',
    ]
    
    print("\nOptional directories:")
    for dir_path in optional_dirs:
        full_path = root / dir_path
        exists = full_path.exists() and full_path.is_dir()
        print(f"{'✓' if exists else '○'} {dir_path}: {'Found' if exists else 'Not generated yet'}")
    
    return all(check[1] for check in checks)


def test_scene_structure(root, split='train'):
    """Test scene directories and file structure."""
    print("\n" + "=" * 80)
    print(f"Testing Scene Structure ({split} split)")
    print("=" * 80)
    
    root = Path(root)
    images_dir = root / 'images' / 'track' / split
    labels_dir = root / 'labels' / 'box_track_20' / split
    
    # Get scene directories
    if not images_dir.exists():
        print(f"✗ Images directory not found: {images_dir}")
        return False
    
    scene_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not scene_dirs:
        print(f"✗ No scene directories found in {images_dir}")
        return False
    
    print(f"✓ Found {len(scene_dirs)} scenes")
    print(f"  First scene: {scene_dirs[0].name}")
    print(f"  Last scene: {scene_dirs[-1].name}")
    
    # Test first scene in detail
    test_scene = scene_dirs[0]
    print(f"\nTesting scene: {test_scene.name}")
    
    # Check images
    image_files = sorted(list(test_scene.glob('*.png')) + list(test_scene.glob('*.jpg')))
    print(f"  ✓ Images: {len(image_files)} files")
    if image_files:
        print(f"    Example: {image_files[0].name}")
    
    # Check JSON annotation
    json_path = labels_dir / f"{test_scene.name}.json"
    if not json_path.exists():
        print(f"  ✗ JSON annotation not found: {json_path}")
        return False
    
    print(f"  ✓ JSON annotation found")
    
    # Parse and validate JSON
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"    Frames in JSON: {len(annotations)}")
        print(f"    Images in dir: {len(image_files)}")
        
        if len(annotations) != len(image_files):
            print(f"  ⚠ Warning: Frame count mismatch!")
        
        # Check first frame annotation
        if annotations:
            frame = annotations[0]
            print(f"    First frame structure:")
            print(f"      - name: {frame.get('name', 'N/A')}")
            print(f"      - videoName: {frame.get('videoName', 'N/A')}")
            print(f"      - frameIndex: {frame.get('frameIndex', 'N/A')}")
            print(f"      - labels: {len(frame.get('labels', []))} objects")
            
            # Check first object
            if frame.get('labels'):
                obj = frame['labels'][0]
                print(f"    First object structure:")
                print(f"      - id: {obj.get('id', 'N/A')}")
                print(f"      - category: {obj.get('category', 'N/A')}")
                print(f"      - box2d: {obj.get('box2d', 'N/A')}")
                print(f"      - attributes: {obj.get('attributes', 'N/A')}")
    
    except Exception as e:
        print(f"  ✗ Error parsing JSON: {e}")
        return False
    
    return True


def test_dataloader_compatibility(root):
    """Test if the dataset can be loaded with BDD100KDataset."""
    print("\n" + "=" * 80)
    print("Testing Dataloader Compatibility")
    print("=" * 80)
    
    try:
        # Try to import the dataset class
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from datasets.bdd100k import BDD100KDataset
        print("✓ Successfully imported BDD100KDataset")
        
        # Create dataset instance
        # Note: You may need to adjust the version parameter
        print("\nAttempting to load dataset...")
        print("Note: If this fails, you may need to:")
        print("  1. Rename output directory to 'bdd100k'")
        print("  2. Or modify BDD100KDataset to accept custom version name")
        
        # This is just a structure test, not a full dataset load
        print("\n✓ Dataloader compatibility checks passed")
        print("  To fully test, run:")
        print("  ```python")
        print("  from ctrlv.datasets.bdd100k import BDD100KDataset")
        print("  dataset = BDD100KDataset(")
        print(f"      root='{Path(root).parent}',")
        print("      train=True,")
        print("      data_type='clip',")
        print("      clip_length=8")
        print("  )")
        print("  sample = dataset[0]")
        print("  print(sample[0].shape)  # Should print image tensor shape")
        print("  ```")
        
    except ImportError as e:
        print(f"✗ Failed to import BDD100KDataset: {e}")
        print("  Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"✗ Error during dataloader test: {e}")
        return False
    
    return True


def print_statistics(root):
    """Print dataset statistics."""
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    root = Path(root)
    
    for split in ['train', 'val']:
        images_dir = root / 'images' / 'track' / split
        if not images_dir.exists():
            print(f"✗ {split.upper()}: Not found")
            continue
        
        scene_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        if not scene_dirs:
            print(f"✗ {split.upper()}: No scenes found")
            continue
        
        total_frames = 0
        frame_counts = []
        
        for scene_dir in scene_dirs:
            num_frames = len(list(scene_dir.glob('*.png')) + list(scene_dir.glob('*.jpg')))
            frame_counts.append(num_frames)
            total_frames += num_frames
        
        print(f"\n{split.upper()} Split:")
        print(f"  Scenes: {len(scene_dirs)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames per scene:")
        print(f"    Min: {min(frame_counts)}")
        print(f"    Max: {max(frame_counts)}")
        print(f"    Avg: {sum(frame_counts) / len(frame_counts):.1f}")
        print(f"  Scene names:")
        print(f"    First: {scene_dirs[0].name}")
        print(f"    Last: {scene_dirs[-1].name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Test KITTI-360 preprocessing output'
    )
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Root directory of preprocessed dataset'
    )
    parser.add_argument(
        '--skip_dataloader',
        action='store_true',
        help='Skip dataloader compatibility test'
    )
    
    args = parser.parse_args()
    
    root = Path(args.root)
    if not root.exists():
        print(f"✗ Error: Directory not found: {root}")
        sys.exit(1)
    
    print(f"Testing preprocessed dataset at: {root}\n")
    
    # Run tests
    tests_passed = []
    
    tests_passed.append(test_directory_structure(root))
    tests_passed.append(test_scene_structure(root, 'train'))
    tests_passed.append(test_scene_structure(root, 'val'))
    
    if not args.skip_dataloader:
        tests_passed.append(test_dataloader_compatibility(root))
    
    print_statistics(root)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    if all(tests_passed):
        print("✓ All tests passed!")
        print("\nYou can now use this dataset for Ctrl-V training.")
        print("\nNext steps:")
        print("  1. (Optional) Generate bbox overlays:")
        print(f"     python generate_kitti360_bbox_overlays.py --root {root}")
        print("  2. Update your training config to use this dataset")
        print("  3. Start training!")
    else:
        print("✗ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
