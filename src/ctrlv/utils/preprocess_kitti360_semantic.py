#!/usr/bin/env python3
"""
KITTI-360 Semantic Image Preprocessing for Ctrl-V
Copies semantic RGB images into BDD100K-compatible format for training

This script reads the existing BDD-format KITTI-360 dataset and copies
the corresponding semantic RGB images into the same structure.

Usage:
    python preprocess_kitti360_semantic.py \
        --kitti360_src /data/public/kitti-360/KITTI-360 \
        --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
        --split train
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess KITTI-360 semantic images for Ctrl-V')
    parser.add_argument('--kitti360_src', type=str, 
                        default='/data/public/kitti-360/KITTI-360',
                        help='Path to KITTI-360 root directory')
    parser.add_argument('--ctrlv_dst', type=str,
                        default='/no_backups/s1492/kitti360_ctrlv',
                        help='Path to Ctrl-V KITTI-360 dataset root')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split to process')
    parser.add_argument('--camera', type=str, default='image_00',
                        choices=['image_00', 'image_01'],
                        help='Camera to use (image_00=left, image_01=right)')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying (faster, saves space)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without actually doing it')
    return parser.parse_args()


def get_semantic_image_path(kitti360_root, seq_name, frame_num, camera='image_00', split='train'):
    """
    Get the path to a semantic RGB image in the KITTI-360 dataset.
    
    Args:
        kitti360_root: Root directory of KITTI-360
        seq_name: Sequence name (e.g., '2013_05_28_drive_0000_sync')
        frame_num: Frame number (e.g., 250)
        camera: Camera ID ('image_00' or 'image_01')
        split: Dataset split ('train' or 'val') - NOTE: KITTI-360 stores all semantics in 'train' dir
    
    Returns:
        Path to semantic RGB image or None if not found
    """
    # KITTI-360 stores all semantic annotations in the 'train' directory
    # regardless of whether the frame is in train or val split
    semantic_path = Path(kitti360_root) / 'data_2d_semantics' / 'train' / seq_name / camera / 'semantic_rgb' / f'{frame_num:010d}.png'
    
    if semantic_path.exists():
        return semantic_path
    return None


def extract_frame_number_from_image(image_path):
    """
    Extract the original KITTI-360 frame number from an image file.
    If the image is a symlink, reads the target path to get the frame number.
    
    Args:
        image_path: Path to the BDD-format image file
    
    Returns:
        tuple of (sequence_name, frame_number) or (None, None) if extraction fails
    """
    image_path = Path(image_path)
    
    # If it's a symlink, resolve to get the original path
    if image_path.is_symlink():
        target_path = image_path.readlink()
    elif image_path.exists():
        # If it's a regular file, try to parse the path
        target_path = image_path
    else:
        return None, None
    
    # Parse the target path to extract sequence and frame number
    # Expected format: .../data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png
    path_str = str(target_path)
    parts = path_str.split('/')
    
    # Find the sequence name (e.g., '2013_05_28_drive_0000_sync')
    seq_name = None
    frame_name = None
    for i, part in enumerate(parts):
        if part.startswith('2013_05_28_drive_') and part.endswith('_sync'):
            seq_name = part
            # Frame name should be at the end
            frame_name = parts[-1]
            break
    
    if seq_name and frame_name:
        # Extract frame number from filename (e.g., '0000000250.png' -> 250)
        try:
            frame_num = int(frame_name.split('.')[0])
            return seq_name, frame_num
        except ValueError:
            pass
    
    return None, None


def process_scene(scene_name, json_path, images_dir, kitti360_root, output_dir, camera='image_00', 
                  split='train', symlink=False, dry_run=False):
    """
    Process a single scene: copy semantic images matching the frames in the JSON.
    
    Args:
        scene_name: Name of the scene (e.g., '2013_05_28_drive_0000_sync_0000')
        json_path: Path to the BDD-format JSON file
        images_dir: Path to the BDD-format images directory
        kitti360_root: Root directory of KITTI-360
        output_dir: Output directory for semantic images
        camera: Camera to use
        split: Dataset split
        symlink: Use symlinks instead of copying
        dry_run: Don't actually copy/symlink files
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_frames': 0,
        'found_frames': 0,
        'missing_frames': 0,
        'copied_frames': 0,
        'skipped_existing': 0
    }
    
    # Load JSON annotation
    with open(json_path, 'r') as f:
        annotations = json.load(f) # [{}, {},...]
    
    stats['total_frames'] = len(annotations) 
    
    # Create output directory for this scene
    scene_output_dir = Path(output_dir) / scene_name
    if not dry_run:
        scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scene images directory
    scene_images_dir = Path(images_dir) / scene_name
    
    # Process each frame
    for frame_anno in annotations:
        frame_name = frame_anno['name']  # e.g., '2013_05_28_drive_0000_sync_0000-0000001.png'
        
        # Get the corresponding image file to extract frame number
        image_file = scene_images_dir / frame_name
        
        # Extract sequence name and frame number from the image file
        seq_name, frame_num = extract_frame_number_from_image(image_file)
        
        if seq_name is None or frame_num is None:
            print(f"Warning: Cannot determine original frame number for {frame_name}")
            stats['missing_frames'] += 1
            continue
        
        # Get semantic image path
        semantic_src = get_semantic_image_path(kitti360_root, seq_name, frame_num, camera, split)
        
        if semantic_src is None:
            # Try without split subdirectory (some KITTI-360 versions may differ)
            print(f"Warning: Semantic image not found for {seq_name} frame {frame_num}")
            stats['missing_frames'] += 1
            continue
        
        stats['found_frames'] += 1
        
        # Destination path (same naming as BDD format)
        semantic_dst = scene_output_dir / frame_name
        
        # Check if already exists
        if semantic_dst.exists():
            stats['skipped_existing'] += 1
            continue
        
        # Copy or symlink
        if dry_run:
            print(f"Would {'symlink' if symlink else 'copy'}: {semantic_src} -> {semantic_dst}")
        else:
            try:
                if symlink:
                    semantic_dst.symlink_to(semantic_src.absolute())
                else:
                    shutil.copy2(semantic_src, semantic_dst)
                stats['copied_frames'] += 1
            except Exception as e:
                print(f"Error {'symlinking' if symlink else 'copying'} {semantic_src} to {semantic_dst}: {e}")
                stats['missing_frames'] += 1
    
    return stats


def main():
    args = parse_args()
    
    print("=" * 80)
    print("KITTI-360 Semantic Image Preprocessing")
    print("=" * 80)
    print(f"Source (KITTI-360):  {args.kitti360_src}")
    print(f"Destination (Ctrl-V): {args.ctrlv_dst}")
    print(f"Split:                {args.split}")
    print(f"Camera:               {args.camera}")
    print(f"Symlink:              {args.symlink}")
    print(f"Dry run:              {args.dry_run}")
    print("=" * 80)
    print()
    
    # Paths
    kitti360_root = Path(args.kitti360_src)
    ctrlv_root = Path(args.ctrlv_dst)
    
    # Input: BDD-format labels and images
    labels_dir = ctrlv_root / 'labels' / 'box_track_20' / args.split
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        print("Please run the BDD format preprocessing first!")
        return
    
    images_dir = ctrlv_root / 'images' / 'track' / args.split
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print("Please run the BDD format preprocessing first!")
        return
    
    # Output: Semantic images directory
    output_dir = ctrlv_root / 'semantics' / 'track' / args.split
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = sorted(labels_dir.glob('*.json'))
    print(f"Found {len(json_files)} scenes to process")
    print()
    
    # Process each scene
    total_stats = defaultdict(int)
    failed_scenes = []
    
    for json_path in tqdm(json_files, desc='Processing scenes'):
        scene_name = json_path.stem  # Remove .json extension
        
        try:
            stats = process_scene(
                scene_name=scene_name,
                json_path=json_path,
                images_dir=images_dir,
                kitti360_root=kitti360_root,
                output_dir=output_dir,
                camera=args.camera,
                split=args.split,
                symlink=args.symlink,
                dry_run=args.dry_run
            )
            
            # Accumulate statistics
            for key, value in stats.items():
                total_stats[key] += value
            
            # Report if there were issues
            if stats['missing_frames'] > 0:
                print(f"\n⚠️  {scene_name}: {stats['missing_frames']}/{stats['total_frames']} frames missing")
                
        except Exception as e:
            print(f"\n❌ Error processing {scene_name}: {e}")
            failed_scenes.append(scene_name)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total frames processed:     {total_stats['total_frames']}")
    print(f"Semantic images found:      {total_stats['found_frames']}")
    print(f"Semantic images copied:     {total_stats['copied_frames']}")
    print(f"Already existing (skipped): {total_stats['skipped_existing']}")
    print(f"Missing semantic images:    {total_stats['missing_frames']}")
    
    if total_stats['total_frames'] > 0:
        coverage = (total_stats['found_frames'] / total_stats['total_frames']) * 100
        print(f"\nSemantic coverage:          {coverage:.1f}%")
    
    if failed_scenes:
        print(f"\n⚠️  Failed scenes: {len(failed_scenes)}")
        for scene in failed_scenes[:10]:  # Show first 10
            print(f"   - {scene}")
        if len(failed_scenes) > 10:
            print(f"   ... and {len(failed_scenes) - 10} more")
    
    print("=" * 80)
    
    if args.dry_run:
        print("\n🔍 This was a DRY RUN - no files were actually copied/symlinked")
        print("Remove --dry_run flag to perform actual processing")
    else:
        print(f"\n✅ Processing complete! Semantic images saved to:")
        print(f"   {output_dir}")


if __name__ == '__main__':
    main()


# to run : 
# python src/ctrlv/utils/preprocess_kitti360_semantic.py \
#     --kitti360_src /data/public/kitti-360/KITTI-360 \
#     --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
#     --split train --camera image_00 --symlink

# for validation : 
# python src/ctrlv/utils/preprocess_kitti360_semantic.py \
#     --kitti360_src /data/public/kitti-360/KITTI-360 \
#     --ctrlv_dst /no_backups/s1492/kitti360_ctrlv \
#     --split val --camera image_00 --symlink