#!/usr/bin/env python3
"""
Generate Bounding Box Overlays for KITTI-360 Dataset (BDD100K format)

Creates bbox overlay images in bboxes/track/{train,val}/ directories
to match the BDD100K dataset structure for Ctrl-V training.

Default mode: 'track' - Solid colored filled boxes on black background,
where each track ID gets a deterministic color (matches training requirements).
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import argparse


# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    'car': (255, 0, 0),          # Blue
    'van': (255, 0, 0),          # Blue
    'truck': (0, 255, 255),      # Yellow
    'bus': (255, 0, 255),        # Magenta
    'train': (0, 0, 255),        # Red
    'motorcycle': (0, 255, 0),   # Green
    'bicycle': (0, 255, 255),    # Yellow
    'caravan': (0, 128, 255),    # Orange
    'trailer': (128, 0, 255),    # Purple
    'pedestrian': (255, 255, 0), # Cyan
    'person': (255, 255, 0),     # Cyan
    'cyclist': (255, 128, 0),    # Light blue
}


def track_color(track_id):
    """Generate deterministic color for track ID."""
    rng = np.random.default_rng(int(track_id))
    return tuple(map(int, rng.integers(50, 256, size=3)))


def draw_bbox_binary(objects, H, W):
    """
    Draw binary bbox mask (white boxes on black background).
    
    Args:
        objects: List of object annotations
        H, W: Image height and width
    
    Returns:
        Binary mask image (H, W) uint8
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    
    for obj in objects:
        if 'box2d' not in obj:
            continue
        
        x1 = int(obj['box2d']['x1'])
        y1 = int(obj['box2d']['y1'])
        x2 = int(obj['box2d']['x2'])
        y2 = int(obj['box2d']['y2'])
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw filled rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
    
    return canvas


def draw_bbox_rgb(objects, H, W, mode='track', alpha=1.0, border_thickness=0):
    """
    Draw colored bbox overlays on black background.
    
    Args:
        objects: List of object annotations
        H, W: Image height and width
        mode: 'class' (color by class) or 'track' (color by track ID)
        alpha: Alpha value (use 1.0 for solid colors, matching screenshot style)
        border_thickness: Thickness of box borders (0 for filled only)
    
    Returns:
        RGB image (H, W, 3) uint8
    """
    # Create black canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    for obj in objects:
        if 'box2d' not in obj:
            continue
        
        x1 = int(obj['box2d']['x1'])
        y1 = int(obj['box2d']['y1'])
        x2 = int(obj['box2d']['x2'])
        y2 = int(obj['box2d']['y2'])
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Choose fill color
        if mode == 'track':
            # Deterministic color per track ID (matches screenshot style)
            fill_color = track_color(obj.get('id', -1))
        else:  # mode == 'class'
            obj_type = obj.get('category', 'car')
            fill_color = CLASS_COLORS.get(obj_type, (255, 255, 255))
        
        # Draw solid filled rectangle on black background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, thickness=cv2.FILLED)
        
        # Optionally draw border (if border_thickness > 0)
        if border_thickness > 0:
            border_color = CLASS_COLORS.get(obj.get('category', 'car'), (255, 255, 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, thickness=border_thickness)
    
    return canvas


def process_scene(scene_dir, labels_dir, output_dir, mode='track', alpha=1.0, 
                  border=0, png_comp=1):
    """
    Process a single scene: generate bbox overlays for all frames.
    
    Args:
        scene_dir: Path to scene image directory
        labels_dir: Path to labels directory
        output_dir: Path to output bbox directory
        mode: 'binary', 'class', or 'track' (default: 'track' for colored boxes)
        alpha: Alpha value (1.0 = solid colors, matching screenshot style)
        border: Border thickness (0 = no border, just filled boxes)
        png_comp: PNG compression level (0-9)
    """
    scene_name = scene_dir.name
    
    # Load scene annotations
    json_path = labels_dir / f"{scene_name}.json"
    if not json_path.exists():
        print(f"Warning: No annotations found for {scene_name}")
        return
    
    with open(json_path, 'r') as f:
        scene_annotations = json.load(f)
    
    if not scene_annotations:
        return
    
    # Get image dimensions from first frame
    first_frame_path = scene_dir / scene_annotations[0]['name']
    img = cv2.imread(str(first_frame_path))
    if img is None:
        print(f"Warning: Cannot read {first_frame_path}")
        return
    H, W = img.shape[:2]
    
    # Create output directory
    output_scene_dir = output_dir / scene_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG compression settings
    imwrite_flags = [cv2.IMWRITE_PNG_COMPRESSION, png_comp]
    
    # Process each frame
    for frame_ann in scene_annotations:
        frame_name = frame_ann['name']
        objects = frame_ann.get('labels', [])
        
        # Draw bbox overlay
        if mode == 'binary':
            canvas = draw_bbox_binary(objects, H, W)
        else:
            canvas = draw_bbox_rgb(objects, H, W, mode=mode, alpha=alpha, border_thickness=border)
        
        # Save bbox image
        output_path = output_scene_dir / frame_name
        cv2.imwrite(str(output_path), canvas, imwrite_flags)


def generate_all_overlays(root, mode='track', alpha=1.0, border=0, workers=None, png_comp=1):
    """
    Generate bbox overlays for all scenes in train and val splits.
    
    Args:
        root: Root directory of processed dataset
        mode: 'binary', 'class', or 'track' (default: 'track' for colored boxes)
        alpha: Alpha value (1.0 = solid colors)
        border: Border thickness (0 = no border)
        workers: Number of parallel workers (None = auto)
        png_comp: PNG compression level (0-9)
    """
    root = Path(root)
    
    for split in ['train', 'val']:
        images_dir = root / 'images' / 'track' / split
        labels_dir = root / 'labels' / 'box_track_20' / split
        output_dir = root / 'bboxes' / 'track' / split
        
        if not images_dir.exists():
            print(f"Skipping {split}: images directory not found")
            continue
        
        if not labels_dir.exists():
            print(f"Skipping {split}: labels directory not found")
            continue
        
        # Get all scene directories
        scene_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        
        if not scene_dirs:
            print(f"No scenes found in {split}")
            continue
        
        print(f"\nGenerating bbox overlays for {split.upper()} ({len(scene_dirs)} scenes)...")
        
        # Create processing function with fixed parameters
        fn = partial(
            process_scene,
            labels_dir=labels_dir,
            output_dir=output_dir,
            mode=mode,
            alpha=alpha,
            border=border,
            png_comp=png_comp
        )
        
        # Process in parallel if workers > 0
        if (workers or 0) != 0:
            n = workers if workers else max(1, cpu_count() - 1)
            with Pool(processes=n) as pool:
                list(tqdm(pool.imap_unordered(fn, scene_dirs), total=len(scene_dirs)))
        else:
            for scene_dir in tqdm(scene_dirs):
                fn(scene_dir)
        
        print(f"✓ {split.upper()} bbox overlays saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate bbox overlays for KITTI-360 dataset in BDD100K format'
    )
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Root directory of processed dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['binary', 'class', 'track'],
        default='track',
        help='Bbox overlay mode: binary (white boxes), class (colored by class), track (colored by ID, default)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Alpha value for RGB modes (1.0 = solid colors, 0.0-1.0)'
    )
    parser.add_argument(
        '--border',
        type=int,
        default=0,
        help='Border thickness for RGB modes (0 = no border, just filled boxes)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel workers (0=sequential, >0=parallel, None=auto)'
    )
    parser.add_argument(
        '--png_comp',
        type=int,
        default=1,
        help='PNG compression level (0=fastest, 9=smallest)'
    )
    
    args = parser.parse_args()
    
    # Disable OpenCV threading (let multiprocessing manage CPU)
    cv2.setNumThreads(0)
    
    print("=" * 80)
    print("KITTI-360 Bbox Overlay Generation")
    print("=" * 80)
    print(f"Dataset root: {args.root}")
    print(f"Mode: {args.mode}")
    if args.mode != 'binary':
        print(f"Alpha: {args.alpha}")
        print(f"Border: {args.border}")
    print(f"Workers: {args.workers if args.workers > 0 else 'sequential'}")
    print(f"PNG compression: {args.png_comp}")
    print()
    
    generate_all_overlays(
        args.root,
        mode=args.mode,
        alpha=args.alpha,
        border=args.border,
        workers=args.workers,
        png_comp=args.png_comp
    )
    
    print("\n" + "=" * 80)
    print("BBOX OVERLAY GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
