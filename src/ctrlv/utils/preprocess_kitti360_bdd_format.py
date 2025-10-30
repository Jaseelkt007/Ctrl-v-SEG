#!/usr/bin/env python3
"""
KITTI-360 Preprocessing Script for Ctrl-V Training
Converts KITTI-360 dataset to BDD100K-compatible format

Output Structure:
kitti360/
  ├── images/track/train/
  │     ├── scene_00000/
  │     │   ├── scene_00000-0000001.jpg
  │     │   └── ...
  │     └── scene_00001/
  ├── bboxes/track/train/
  │     ├── scene_00000/
  │     │   ├── scene_00000-0000001.jpg
  │     │   └── ...
  ├── labels/box_track_20/train/
  │     ├── scene_00000.json
  │     └── scene_00001.json
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse

VEHICLE_TYPES = {"car", "van", "truck", "bus", "motorcycle", "bicycle", "caravan", "trailer", "train"}

# Class mapping for KITTI-360 (compatible with BDD100K format)
CLASS_IDS_LOOKUP = {
    'car': 3,
    'van': 3,  # Map to car
    'truck': 4,
    'bus': 5,
    'motorcycle': 7,
    'bicycle': 8,
    'caravan': 4,  # Map to truck
    'trailer': 4,  # Map to truck
    'train': 6,
    'pedestrian': 1,
    'person': 1,
    'cyclist': 2,
}


def parse_kitti360_txt_annotation(txt_path):
    """
    Parse a single KITTI-360 text annotation file.
    
    Format: frame trackID type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z rot_y
    
    Returns:
        list of object dictionaries
    """
    objects = []
    if not os.path.exists(txt_path):
        return objects
    
    with open(txt_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 17:
                continue
            
            obj_type = vals[2].lower()
            # Filter to only keep relevant object types
            if obj_type not in VEHICLE_TYPES and obj_type not in ['pedestrian', 'person', 'cyclist']:
                continue
            
            objects.append({
                'id': int(vals[1]),  # trackID as integer (for BDD100K compatibility)
                'category': obj_type,
                'box2d': {
                    'x1': float(vals[6]),
                    'y1': float(vals[7]),
                    'x2': float(vals[8]),
                    'y2': float(vals[9])
                },
                'attributes': {
                    'truncated': float(vals[3]),  # 0-1 range, should be float
                    'occluded': int(vals[4])
                },
                'alpha': float(vals[5]),
                'dimensions': {
                    'height': float(vals[10]),
                    'width': float(vals[11]),
                    'length': float(vals[12])
                },
                'location': {
                    'x': float(vals[13]),
                    'y': float(vals[14]),
                    'z': float(vals[15])
                },
                'rotation_y': float(vals[16])
            })
    
    return objects


def detect_scene_boundaries(image_list, frame_jump_threshold=100):
    """
    Detect scene boundaries in the image list by finding large frame number jumps.
    
    Args:
        image_list: List of image paths
        frame_jump_threshold: Frame number difference to consider a new scene
    
    Returns:
        List of scene groups: [(scene_name, [image_paths])]
    """
    scenes = []
    current_scene = []
    prev_frame_num = None
    prev_seq_name = None
    scene_counter = 0
    
    for img_path in image_list:
        # Extract sequence name and frame number
        # Path format: data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png
        parts = img_path.strip().split('/')
        seq_name = parts[1]  # e.g., 2013_05_28_drive_0000_sync
        frame_name = parts[-1]  # e.g., 0000000250.png
        frame_num = int(frame_name.split('.')[0])
        
        # Detect scene boundary
        new_scene = False
        if prev_seq_name is not None and seq_name != prev_seq_name:
            # Sequence changed
            new_scene = True
        elif prev_frame_num is not None and (frame_num - prev_frame_num) > frame_jump_threshold:
            # Large frame jump within same sequence
            new_scene = True
        
        if new_scene and current_scene:
            # Save the current scene
            scene_name = f"{prev_seq_name}_{scene_counter:04d}"
            scenes.append((scene_name, current_scene))
            scene_counter += 1
            current_scene = []
        
        # Start new sequence counter
        if prev_seq_name is not None and seq_name != prev_seq_name:
            scene_counter = 0
        
        current_scene.append({
            'path': img_path.strip(),
            'seq_name': seq_name,
            'frame_num': frame_num,
            'frame_name': frame_name
        })
        
        prev_frame_num = frame_num
        prev_seq_name = seq_name
    
    # Don't forget the last scene
    if current_scene:
        scene_name = f"{prev_seq_name}_{scene_counter:04d}"
        scenes.append((scene_name, current_scene))
    
    return scenes


def create_bdd_format_structure(scenes, split, kitti_root, output_root, symlink=True):
    """
    Create BDD100K-compatible directory structure and copy/symlink images.
    
    Args:
        scenes: List of (scene_name, frames) tuples
        split: 'train' or 'val'
        kitti_root: Root directory of KITTI-360 dataset
        output_root: Output directory for processed dataset
        symlink: Use symlinks instead of copying (much faster)
    
    Returns:
        Dictionary mapping scene_name to list of processed frames
    """
    # Create directory structure
    images_dir = Path(output_root) / 'images' / 'track' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    
    scene_data = {}
    
    for scene_name, frames in tqdm(scenes, desc=f"Processing {split} scenes"):
        # Create scene directory
        scene_dir = images_dir / scene_name
        scene_dir.mkdir(exist_ok=True)
        
        scene_frames = []
        
        for idx, frame_info in enumerate(frames, start=1):
            # Source image path
            src_img = Path(kitti_root) / frame_info['path']
            
            # Destination image path (BDD100K format: scene_name-0000001.jpg)
            # Keep as .png for KITTI-360
            dst_name = f"{scene_name}-{idx:07d}.png"
            dst_img = scene_dir / dst_name
            
            # Copy or symlink image
            if not dst_img.exists():
                if symlink:
                    try:
                        dst_img.symlink_to(src_img.absolute())
                    except Exception as e:
                        print(f"Warning: Failed to create symlink for {src_img}, copying instead: {e}")
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            
            # Store frame info for JSON generation
            scene_frames.append({
                'name': dst_name,
                'frame_num': frame_info['frame_num'],
                'seq_name': frame_info['seq_name'],
                'src_path': frame_info['path']
            })
        
        scene_data[scene_name] = scene_frames
    
    return scene_data


def create_bbox_annotations(scene_data, split, kitti_root, output_root, skip_existing=True):
    """
    Create BDD100K-compatible JSON annotations for bounding boxes.
    
    Args:
        scene_data: Dictionary mapping scene_name to list of frames
        split: 'train' or 'val'
        kitti_root: Root directory of KITTI-360 dataset
        output_root: Output directory for processed dataset
        skip_existing: Skip scenes that already have valid JSON files
    """
    # Create labels directory
    labels_dir = Path(output_root) / 'labels' / 'box_track_20' / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for scene_name, frames in tqdm(scene_data.items(), desc=f"Creating {split} annotations"):
        # Skip if already processed and valid
        json_path = labels_dir / f"{scene_name}.json"
        if skip_existing and json_path.exists() and os.path.getsize(json_path) > 0:
            continue
        # BDD100K format: list of frame annotations
        scene_annotations = []
        
        for idx, frame in enumerate(frames):
            # Find corresponding annotation file
            # Path: data_2d_raw/<SEQ>/label_00/<FRAME>.txt
            seq_name = frame['seq_name']
            frame_num = frame['frame_num']
            txt_path = Path(kitti_root) / 'data_2d_raw' / seq_name / 'label_00' / f"{frame_num:010d}.txt"
            
            # Parse annotations
            objects = parse_kitti360_txt_annotation(txt_path)
            
            # Create frame annotation
            frame_annotation = {
                'name': frame['name'],
                'videoName': scene_name,
                'frameIndex': idx, 
                'labels': objects
            }
            
            scene_annotations.append(frame_annotation)
        
        # Save JSON file (already checked above if skip_existing)
        with open(json_path, 'w') as f:
            json.dump(scene_annotations, f, indent=2)


def print_statistics(scenes, split):
    """Print statistics about the processed scenes."""
    total_frames = sum(len(frames) for _, frames in scenes)
    frame_counts = [len(frames) for _, frames in scenes]
    
    print(f"\n{split.upper()} Split Statistics:")
    print(f"  Total scenes: {len(scenes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames per scene - Min: {min(frame_counts)}, Max: {max(frame_counts)}, "
          f"Avg: {sum(frame_counts) / len(frame_counts):.1f}")
    print(f"  Scene names: {scenes[0][0]} ... {scenes[-1][0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess KITTI-360 dataset to BDD100K-compatible format for Ctrl-V'
    )
    parser.add_argument(
        '--kitti_root',
        type=str,
        default='/data/public/kitti-360/KITTI-360',
        help='Root directory of KITTI-360 dataset'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='/no_backups/s1492/kitti360_ctrlv',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--frame_jump_threshold',
        type=int,
        default=100,
        help='Frame number difference to consider a new scene'
    )
    parser.add_argument(
        '--no_symlink',
        action='store_true',
        help='Copy images instead of creating symlinks (slower but more portable)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing of existing files (skip existing files by default)'
    )
    
    args = parser.parse_args()
    
    kitti_root = Path(args.kitti_root)
    output_root = Path(args.output_root)
    
    # Paths to image lists
    train_list_path = kitti_root / 'train_images.txt'
    val_list_path = kitti_root / 'val_images.txt'
    
    # Verify paths exist
    if not train_list_path.exists():
        raise FileNotFoundError(f"Train image list not found: {train_list_path}")
    if not val_list_path.exists():
        raise FileNotFoundError(f"Val image list not found: {val_list_path}")
    
    print("=" * 80)
    print("KITTI-360 to BDD100K Format Conversion for Ctrl-V")
    print("=" * 80)
    print(f"KITTI-360 root: {kitti_root}")
    print(f"Output root: {output_root}")
    print(f"Frame jump threshold: {args.frame_jump_threshold}")
    print(f"Use symlinks: {not args.no_symlink}")
    print()
    
    # Process training set
    print("Processing TRAINING set...")
    with open(train_list_path, 'r') as f:
        train_images = f.readlines()
    
    train_scenes = detect_scene_boundaries(train_images, args.frame_jump_threshold)
    print_statistics(train_scenes, 'train')
    
    train_scene_data = create_bdd_format_structure(
        train_scenes, 'train', kitti_root, output_root, symlink=not args.no_symlink
    )
    create_bbox_annotations(train_scene_data, 'train', kitti_root, output_root, 
                           skip_existing=not args.force)
    
    # Process validation set
    print("\nProcessing VALIDATION set...")
    with open(val_list_path, 'r') as f:
        val_images = f.readlines()
    
    val_scenes = detect_scene_boundaries(val_images, args.frame_jump_threshold)
    print_statistics(val_scenes, 'val')
    
    val_scene_data = create_bdd_format_structure(
        val_scenes, 'val', kitti_root, output_root, symlink=not args.no_symlink
    )
    create_bbox_annotations(val_scene_data, 'val', kitti_root, output_root,
                           skip_existing=not args.force)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nOutput structure:")
    print(f"  {output_root}/")
    print(f"    ├── images/track/train/scene_xxxxx/")
    print(f"    ├── images/track/val/scene_xxxxx/")
    print(f"    └── labels/box_track_20/train/ & val/")
    print(f"\nTo generate bbox overlays, run:")
    print(f"  python generate_kitti360_bbox_overlays.py --root {output_root} --mode track --workers 8")


if __name__ == '__main__':
    main()
