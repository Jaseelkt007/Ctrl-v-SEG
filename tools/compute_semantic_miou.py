#!/usr/bin/env python3
"""
Semantic Segmentation mIoU Evaluation for Generated Videos

This script evaluates semantic segmentation quality by:
1. Running a pretrained segmentation model (DRN-D-105) on generated frames
2. Comparing predicted segmentation to ground truth KITTI-360 semantic labels
3. Computing mIoU across all Cityscapes classes (19 classes)

Requirements:
    pip install torch torchvision pillow opencv-python
    git clone https://github.com/fyu/drn.git (in home directory)

Usage:
    python tools/compute_semantic_miou.py \
        --generated_frames_dir /path/to/generated/frames/ \
        --gt_semantic_dir /data/public/kitti-360/KITTI-360/data_2d_semantics/ \
        --val_split_file /data/public/kitti-360/KITTI-360/val_images.txt \
        --drn_path /usrhomes/s1492/drn \
        --output_file semantic_miou_results.txt
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# Cityscapes 19-class labels (used by KITTI-360)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], 
    [0, 0, 230], [119, 11, 32]
]

# KITTI-360 to Cityscapes trainId mapping
# KITTI-360 uses Cityscapes labels directly
# trainId is 0-18 for 19 classes, 255 for ignore
KITTI360_TO_TRAINID = {
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,  # parking
    10: 255, # rail track
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, # guard rail
    15: 255, # bridge
    16: 255, # tunnel
    17: 5,   # pole
    18: 255, # polegroup
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 16,  # caravan
    30: 16,  # trailer
    31: 17,  # train
    32: 18,  # motorcycle
    33: 19,  # bicycle
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute semantic segmentation mIoU for generated videos'
    )
    parser.add_argument(
        '--generated_frames_dir',
        type=str,
        required=True,
        help='Directory containing generated frames (e.g., media/images/)'
    )
    parser.add_argument(
        '--gt_semantic_dir',
        type=str,
        default='/data/public/kitti-360/KITTI-360/data_2d_semantics/',
        help='Directory containing KITTI-360 semantic labels'
    )
    parser.add_argument(
        '--val_split_file',
        type=str,
        default='/data/public/kitti-360/KITTI-360/val_images.txt',
        help='Validation split file with image paths'
    )
    parser.add_argument(
        '--drn_path',
        type=str,
        default='/usrhomes/s1492/drn',
        help='Path to DRN repository'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='drn_d_105',
        choices=['drn_d_105', 'drn_d_54', 'drn_d_38', 'drn_c_26'],
        help='DRN model variant to use'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file for results (default: generated_frames_dir/semantic_miou_results.txt)'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save predicted segmentation maps'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device (cuda:0 or cpu)'
    )
    return parser.parse_args()


def get_segmentation_model(drn_path, model_name='drn_d_105', num_classes=19, device='cuda:0'):
    """
    Load pretrained DRN segmentation model.
    
    Args:
        drn_path: Path to DRN repository
        model_name: DRN model variant (e.g., 'drn_d_105')
        num_classes: Number of segmentation classes (19 for Cityscapes)
        device: Device to load model on
    
    Returns:
        model: Initialized DRN segmentation model
    """
    print(f"Loading {model_name} model pretrained on Cityscapes...")
    
    # Add DRN path to Python path
    if drn_path not in sys.path:
        sys.path.insert(0, drn_path)
    
    # Import DRN modules
    try:
        import drn
        from segment import DRNSeg
    except ImportError as e:
        print(f"Error: Could not import DRN modules from {drn_path}")
        print(f"Make sure you've cloned: git clone https://github.com/fyu/drn.git")
        print(f"Error details: {e}")
        sys.exit(1)
    
    # Load pretrained model
    print(f"Downloading pretrained weights for {model_name}...")
    # Note: DRNSeg uses 'classes' parameter, not 'num_classes'
    model = DRNSeg(model_name, classes=num_classes, pretrained_model=None, pretrained=False)
    
    # Load pretrained weights from model zoo
    pretrained_dict = torch.hub.load_state_dict_from_url(
        drn.model_urls[model_name.replace('_', '-')],
        progress=True
    )
    
    # Load weights for segmentation model
    model_dict = model.state_dict()
    # Filter out unnecessary keys (classification head)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model


def load_kitti360_semantic_label(label_path):
    """
    Load KITTI-360 semantic label and convert to trainId format.
    
    Args:
        label_path: Path to KITTI-360 semantic PNG (grayscale with class IDs)
    
    Returns:
        label: numpy array [H, W] with trainId values (0-18 for classes, 255 for ignore)
    """
    # Load grayscale image
    label = np.array(Image.open(label_path))
    
    # Convert KITTI-360 labelIds to trainIds
    label_trainid = np.ones_like(label) * 255  # Start with ignore
    for labelid, trainid in KITTI360_TO_TRAINID.items():
        label_trainid[label == labelid] = trainid
    
    return label_trainid.astype(np.uint8)


def parse_val_split(val_split_file):
    """
    Parse validation split file to get list of scenes and frame numbers.
    
    Returns:
        val_frames: List of (scene, frame_number) tuples
    """
    val_frames = []
    with open(val_split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png
            parts = line.split('/')
            scene = parts[1]  # e.g., 2013_05_28_drive_0000_sync
            frame_num = parts[-1].replace('.png', '')  # e.g., 0000000250
            
            val_frames.append((scene, frame_num))
    
    return val_frames


def find_generated_frames(generated_frames_dir):
    """
    Find all generated frames in the directory.
    
    Supports multiple naming patterns:
    - frames_with_{sample_idx}_{step}_{hash}.png
    - frames_with_bboxes_{sample_idx}_{step}_{hash}.png
    
    Returns:
        frames: Sorted list of (frame_path, sample_idx) tuples
    """
    frames = []
    
    # Look for frames_with_*.png pattern (but not frames_with_bboxes)
    image_files = [f for f in os.listdir(generated_frames_dir) 
                   if f.startswith('frames_with_') and f.endswith('.png') 
                   and not f.startswith('frames_with_bboxes_')]
    
    if len(image_files) == 0:
        # Fallback: try frames_with_bboxes pattern
        print("No 'frames_with_*.png' found, trying 'frames_with_bboxes_*.png' pattern...")
        image_files = [f for f in os.listdir(generated_frames_dir) 
                       if f.startswith('frames_with_bboxes_') and f.endswith('.png')]
        is_bbox_pattern = True
    else:
        is_bbox_pattern = False
    
    # Sort by sample index
    # Format: frames_with_{sample_idx}_{step}_{hash}.png
    # or:     frames_with_bboxes_{sample_idx}_{step}_{hash}.png
    for fname in sorted(image_files):
        # Extract sample index from filename
        try:
            parts = fname.split('_')
            if is_bbox_pattern:
                # frames_with_bboxes_{idx}_{step}_{hash}.png
                sample_idx = int(parts[3])  # Skip 'frames', 'with', 'bboxes'
            else:
                # frames_with_{idx}_{step}_{hash}.png
                sample_idx = int(parts[2])  # Skip 'frames', 'with'
            
            frame_path = os.path.join(generated_frames_dir, fname)
            frames.append((frame_path, sample_idx))
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse filename {fname}, skipping (error: {e})")
            continue
    
    # Sort by sample index
    frames.sort(key=lambda x: x[1])
    
    return frames


def compute_miou(pred, gt, num_classes=19, ignore_index=255):
    """
    Compute mIoU between predicted and ground truth segmentation.
    
    Args:
        pred: Predicted segmentation [H, W] with class IDs
        gt: Ground truth segmentation [H, W] with class IDs
        num_classes: Number of classes (19 for Cityscapes)
        ignore_index: Index to ignore (255)
    
    Returns:
        iou_per_class: IoU for each class
        miou: Mean IoU across all classes
    """
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        
        # Ignore pixels
        valid_mask = (gt != ignore_index)
        pred_mask = pred_mask & valid_mask
        gt_mask = gt_mask & valid_mask
        
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        
        if union == 0:
            iou = float('nan')  # No pixels for this class
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    # Compute mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return iou_per_class, miou


def main():
    args = parse_args()
    
    # Validate directories
    if not os.path.exists(args.generated_frames_dir):
        print(f"Error: Generated frames directory not found: {args.generated_frames_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.gt_semantic_dir):
        print(f"Error: GT semantic directory not found: {args.gt_semantic_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.val_split_file):
        print(f"Error: Validation split file not found: {args.val_split_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Semantic Segmentation mIoU Evaluation")
    print("=" * 60)
    print()
    print(f"Generated frames: {args.generated_frames_dir}")
    print(f"GT semantic dir:  {args.gt_semantic_dir}")
    print(f"Val split file:   {args.val_split_file}")
    print(f"DRN path:         {args.drn_path}")
    print(f"Model:            {args.model_name}")
    print(f"Device:           {args.device}")
    print()
    
    # Validate DRN path
    if not os.path.exists(args.drn_path):
        print(f"Error: DRN repository not found at: {args.drn_path}")
        print("Please clone it with: git clone https://github.com/fyu/drn.git")
        sys.exit(1)
    
    # Load segmentation model
    model = get_segmentation_model(args.drn_path, args.model_name, num_classes=19, device=args.device)
    
    # Parse validation split
    print("Parsing validation split...")
    val_frames = parse_val_split(args.val_split_file)
    print(f"✓ Found {len(val_frames)} validation frames")
    print()
    
    # Find generated frames
    print("Finding generated frames...")
    generated_frames = find_generated_frames(args.generated_frames_dir)
    print(f"✓ Found {len(generated_frames)} generated frames")
    print()
    
    if len(generated_frames) == 0:
        print("Error: No generated frames found!")
        sys.exit(1)
    
    # Limit samples if specified
    if args.max_samples is not None:
        generated_frames = generated_frames[:args.max_samples]
        print(f"Limiting to {len(generated_frames)} samples")
        print()
    
    # Evaluate each frame
    print("=" * 60)
    print("Running semantic segmentation evaluation...")
    print("=" * 60)
    print()
    
    all_ious = []
    class_ious = [[] for _ in range(19)]
    
    # Track resolution info
    gen_resolution = None
    gt_resolution = None
    
    for frame_idx, (gen_frame_path, sample_idx) in enumerate(tqdm(generated_frames, desc="Evaluating")):
        # Map to corresponding GT frame
        if sample_idx >= len(val_frames):
            print(f"Warning: Sample {sample_idx} exceeds validation set size, skipping")
            continue
        
        scene, frame_num = val_frames[sample_idx]
        
        # Construct GT semantic path
        # /data/public/kitti-360/KITTI-360/data_2d_semantics/train/{scene}/image_00/semantic/{frame_num}.png
        gt_semantic_path = os.path.join(
            args.gt_semantic_dir,
            'train',
            scene,
            'image_00',
            'semantic',
            f'{frame_num}.png'
        )
        
        if not os.path.exists(gt_semantic_path):
            print(f"Warning: GT semantic not found: {gt_semantic_path}, skipping")
            continue
        
        # Load generated frame
        gen_frame = Image.open(gen_frame_path).convert('RGB')
        orig_size = gen_frame.size  # (W, H)
        
        # Preprocess image for DRN model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.290101, 0.328081, 0.286964],
                               std=[0.182954, 0.186566, 0.184475])
        ])
        
        img_tensor = transform(gen_frame).unsqueeze(0).to(args.device)
        
        # Run segmentation
        with torch.no_grad():
            # DRNSeg returns (log_softmax_output, raw_features)
            output, _ = model(img_tensor)
            # Get class predictions from log probabilities
            pred_seg = output[0].max(0)[1].cpu().numpy()  # [H, W] with class IDs
        
        # Load GT semantic
        gt_seg = load_kitti360_semantic_label(gt_semantic_path)
        
        # Track resolutions (first frame only)
        if gen_resolution is None:
            gen_resolution = f"{pred_seg.shape[1]}×{pred_seg.shape[0]}"
            gt_resolution = f"{gt_seg.shape[1]}×{gt_seg.shape[0]}"
        
        # Resize GT to match prediction resolution (evaluate at generation resolution)
        # This is more fair than upscaling low-res predictions
        if pred_seg.shape != gt_seg.shape:
            gt_seg = cv2.resize(gt_seg, (pred_seg.shape[1], pred_seg.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Compute IoU
        iou_per_class, miou = compute_miou(pred_seg, gt_seg)
        all_ious.append(miou)
        
        for cls_idx, iou in enumerate(iou_per_class):
            if not np.isnan(iou):
                class_ious[cls_idx].append(iou)
        
        # Save predictions if requested
        if args.save_predictions:
            pred_dir = os.path.join(args.generated_frames_dir, 'predicted_segmentation')
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f'pred_{sample_idx:04d}.png')
            # Convert to uint8 for PIL
            pred_seg_uint8 = pred_seg.astype(np.uint8)
            Image.fromarray(pred_seg_uint8).save(pred_path)
    
    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print()
    
    # Compute overall mIoU
    overall_miou = np.mean(all_ious)
    std_miou = np.std(all_ious)
    
    # Compute per-class mIoU
    class_miou = []
    for cls_idx in range(19):
        if class_ious[cls_idx]:
            cls_miou = np.mean(class_ious[cls_idx])
            class_miou.append(cls_miou)
        else:
            class_miou.append(float('nan'))
    
    # Prepare output
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("Semantic Segmentation mIoU Results")
    output_lines.append("=" * 60)
    output_lines.append("")
    output_lines.append(f"Model:            {args.model_name}")
    output_lines.append(f"Samples evaluated: {len(all_ious)}")
    output_lines.append(f"Classes:          19 (Cityscapes)")
    output_lines.append("")
    output_lines.append(f"Generated resolution: {gen_resolution}")
    output_lines.append(f"GT resolution:        {gt_resolution}")
    output_lines.append(f"Evaluation at:        {gen_resolution} (GT downsampled)")
    output_lines.append("")
    output_lines.append(f"Overall mIoU:     {overall_miou:.4f} ± {std_miou:.4f}")
    output_lines.append("")
    output_lines.append("=" * 60)
    output_lines.append("Per-Class mIoU:")
    output_lines.append("=" * 60)
    
    for cls_idx, cls_name in enumerate(CITYSCAPES_CLASSES):
        if not np.isnan(class_miou[cls_idx]):
            output_lines.append(f"  {cls_name:20s}: {class_miou[cls_idx]:.4f}")
        else:
            output_lines.append(f"  {cls_name:20s}: N/A (no pixels)")
    
    output_lines.append("")
    output_lines.append("=" * 60)
    
    # Print output
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file
    if args.output_file is None:
        output_file = os.path.join(args.generated_frames_dir, 'semantic_miou_results.txt')
    else:
        output_file = args.output_file
    
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print()
    print(f"Results saved to: {output_file}")
    print()


if __name__ == '__main__':
    main()