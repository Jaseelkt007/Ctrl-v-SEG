"""
Semantic preprocessing utilities for Ctrl-V with Semantic VAE

This module provides the preprocessing pipeline for KITTI-360 grayscale semantic labels.
Uses the same preprocessing as the semantic VAE training.

Pipeline:
1. Load grayscale semantic PNG (raw KITTI-360 labels)
2. Remap to continuous trainIds (0-18) using KITTI360_LABEL_MAPPING
3. Set invalid labels to ignore_index (255)
4. Convert to one-hot encoding for semantic VAE

NOTE: This is NOT RGB semantic. These are real grayscale semantic labels.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Optional
from pathlib import Path

# Import KITTI-360 label mapping
try:
    from kitti360scripts.helpers.labels import id2label
    
    def build_kitti360_mapping():
        """Build mapping from KITTI-360 label IDs to continuous trainIds."""
        mapping = {}
        for label_id, label_obj in id2label.items():
            if label_obj.trainId != 255 and label_obj.trainId >= 0:
                mapping[label_id] = label_obj.trainId
        return mapping
    
    KITTI360_LABEL_MAPPING = build_kitti360_mapping()
    HAS_KITTI360_SCRIPTS = True
except ImportError:
    print("Warning: kitti360scripts not available. Using fallback mapping.")
    HAS_KITTI360_SCRIPTS = False
    # Fallback mapping (19 classes)
    KITTI360_LABEL_MAPPING = {
        7: 0,   # road
        8: 1,   # sidewalk
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        17: 5,  # pole
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        31: 16, # train
        32: 17, # motorcycle
        33: 18, # bicycle
    }


def load_and_remap_semantic(semantic_path: Union[str, Path], ignore_index: int = 255) -> np.ndarray:
    """
    Load a grayscale semantic PNG and remap KITTI-360 IDs to continuous trainIDs.
    
    Args:
        semantic_path: Path to grayscale semantic PNG
        ignore_index: Value to use for ignore class (default 255)
    
    Returns:
        numpy array [H, W] with trainIDs (0-18) or ignore_index
    """
    # Load grayscale semantic image - force to grayscale mode
    semantic_img = Image.open(semantic_path).convert('L')  # Force grayscale
    semantic = np.array(semantic_img, dtype=np.int64)  # [H, W] with KITTI-360 IDs
    
    # Initialize with ignore_index
    remapped = np.full_like(semantic, ignore_index, dtype=np.int64)
    
    # Remap valid labels to trainIds
    for kitti_id, train_id in KITTI360_LABEL_MAPPING.items():
        remapped[semantic == kitti_id] = train_id
    
    return remapped


def semantic_ids_to_onehot(
    semantic_ids: torch.Tensor,
    num_classes: int = 19,
    ignore_index: int = 255
) -> torch.Tensor:
    """
    Convert semantic IDs to one-hot encoding for semantic VAE.
    
    Args:
        semantic_ids: [B, H, W] or [H, W] tensor with trainIds (0-18) or ignore_index
        num_classes: Number of classes (19 for KITTI-360)
        ignore_index: Value to ignore (default: 255)
    
    Returns:
        onehot: [B, num_classes, H, W] or [num_classes, H, W] one-hot tensor
    """
    # Add batch dim if needed
    if semantic_ids.ndim == 2:
        semantic_ids = semantic_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W = semantic_ids.shape
    device = semantic_ids.device
    
    # Clamp to valid range [0, num_classes-1]
    # ignore_index (255) will be mapped to 0, but masked in loss
    semantic_clamped = torch.clamp(semantic_ids, 0, num_classes - 1)
    
    # One-hot encoding
    onehot = F.one_hot(semantic_clamped.long(), num_classes=num_classes)  # [B, H, W, num_classes]
    onehot = onehot.permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]
    
    # Mask ignore pixels (set all channels to 0)
    ignore_mask = (semantic_ids == ignore_index).unsqueeze(1)  # [B, 1, H, W]
    onehot = onehot * (~ignore_mask).float()
    
    if squeeze_output:
        onehot = onehot.squeeze(0)
    
    return onehot


def compute_boundary_mask(semantic_ids: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Compute boundary mask from semantic IDs.
    
    A pixel is a boundary if at least one neighbor has a different label.
    
    Args:
        semantic_ids: [B, H, W] tensor with class IDs
        ignore_index: Label to ignore
    
    Returns:
        boundary_mask: [B, H, W] binary mask (1 at boundaries, 0 elsewhere)
    """
    B, H, W = semantic_ids.shape
    device = semantic_ids.device
    
    # Valid mask
    valid_mask = (semantic_ids != ignore_index).float()
    
    # Convert to float for pooling
    semantic_float = semantic_ids.float().unsqueeze(1)  # [B, 1, H, W]
    
    # 3x3 max and min pooling to detect label changes
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    min_vals = -max_pool(-semantic_float)
    max_vals = max_pool(semantic_float)
    
    # Boundary where max != min
    boundary = (max_vals != min_vals).float().squeeze(1)  # [B, H, W]
    
    # Only at valid pixels
    boundary_mask = boundary * valid_mask
    
    return boundary_mask


def resize_semantic(semantic_ids: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize semantic map using nearest neighbor interpolation.
    
    Args:
        semantic_ids: [H, W] numpy array with class IDs
        target_size: (H, W) target size
    
    Returns:
        resized: [H', W'] resized semantic map
    """
    # Convert to tensor
    semantic_tensor = torch.from_numpy(semantic_ids).unsqueeze(0).unsqueeze(0).float()
    
    # Resize with nearest neighbor
    resized = F.interpolate(
        semantic_tensor,
        size=target_size,
        mode='nearest'
    )
    
    # Back to numpy
    return resized.squeeze(0).squeeze(0).numpy().astype(np.int64)


# KITTI-360 19-class names (trainIds 0-18)
KITTI360_CLASS_NAMES = [
    'road',        # 0
    'sidewalk',    # 1
    'building',    # 2
    'wall',        # 3
    'fence',       # 4
    'pole',        # 5
    'traffic light', # 6
    'traffic sign',  # 7
    'vegetation',  # 8
    'terrain',     # 9
    'sky',         # 10
    'person',      # 11
    'rider',       # 12
    'car',         # 13
    'truck',       # 14
    'bus',         # 15
    'train',       # 16
    'motorcycle',  # 17
    'bicycle',     # 18
]


# Visualization colormap (optional, for debugging)
KITTI360_VIZ_COLORS = {
    0: (128, 64, 128),    # road - purple
    1: (244, 35, 232),    # sidewalk - pink
    2: (70, 70, 70),      # building - gray
    3: (102, 102, 156),   # wall - light purple
    4: (190, 153, 153),   # fence - light pink
    5: (153, 153, 153),   # pole - gray
    6: (250, 170, 30),    # traffic light - orange
    7: (220, 220, 0),     # traffic sign - yellow
    8: (107, 142, 35),    # vegetation - olive
    9: (152, 251, 152),   # terrain - light green
    10: (70, 130, 180),   # sky - blue
    11: (220, 20, 60),    # person - red
    12: (255, 0, 0),      # rider - bright red
    13: (0, 0, 142),      # car - dark blue
    14: (0, 0, 70),       # truck - darker blue
    15: (0, 60, 100),     # bus - teal
    16: (0, 80, 100),     # train - cyan
    17: (0, 0, 230),      # motorcycle - blue
    18: (119, 11, 32),    # bicycle - maroon
}


def semantic_ids_to_viz_rgb(semantic_ids: np.ndarray, ignore_color=(0, 0, 0)) -> np.ndarray:
    """
    Convert semantic IDs to RGB for visualization.
    
    Args:
        semantic_ids: [H, W] array with trainIds (0-18) or ignore_index
        ignore_color: RGB color for ignored pixels
    
    Returns:
        rgb: [H, W, 3] RGB image
    """
    H, W = semantic_ids.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id, color in KITTI360_VIZ_COLORS.items():
        mask = (semantic_ids == class_id)
        rgb[mask] = color
    
    # Ignore pixels
    ignore_mask = (semantic_ids == 255)
    rgb[ignore_mask] = ignore_color
    
    return rgb


def get_class_weights(inverse_freq: bool = True) -> np.ndarray:
    """
    Get class weights for KITTI-360 (based on typical class frequencies).
    
    Args:
        inverse_freq: If True, return inverse frequency weights
    
    Returns:
        weights: [num_classes] array
    """
    # Approximate class frequencies from KITTI-360
    frequencies = np.array([
        0.35,  # road
        0.08,  # sidewalk
        0.15,  # building
        0.01,  # wall
        0.01,  # fence
        0.02,  # pole
        0.01,  # traffic light
        0.02,  # traffic sign
        0.15,  # vegetation
        0.03,  # terrain
        0.10,  # sky
        0.01,  # person
        0.005, # rider
        0.05,  # car
        0.01,  # truck
        0.005, # bus
        0.001, # train
        0.005, # motorcycle
        0.01,  # bicycle
    ])
    
    if inverse_freq:
        weights = 1.0 / (frequencies + 1e-6)
        weights = weights / weights.sum() * len(weights)  # Normalize
        return weights
    else:
        return frequencies
