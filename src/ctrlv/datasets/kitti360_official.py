"""
KITTI-360 Dataset using Official TXT Files

This dataset loads RGB images and grayscale semantic IDs directly from
the official KITTI-360 directory structure using the train/val split
txt files provided by KITTI-360.

- Uses official KITTI-360 txt files for train/val split
- Loads RGB from: /misc/data/public/kitti-360/KITTI-360/data_2d_raw/
- Loads semantics from: /misc/data/public/kitti-360/KITTI-360/data_2d_semantics/train/

TXT file format (2013_05_28_drive_train_frames.txt):
    data_2d_raw/SEQUENCE/image_00/data_rect/FRAME.png data_2d_semantics/train/SEQUENCE/image_00/semantic/FRAME.png

Example:
    data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic/0000000250.png
"""

import os
from PIL import Image
import torch
import numpy as np
from typing import Tuple, List, Optional
import sys
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V-seg/src')

from .kitti_abstract import KittiAbstract
from ctrlv.utils.semantic_preprocessing import load_and_remap_semantic


class KITTI360OfficialDataset(KittiAbstract):
    """
    KITTI-360 dataset using official directory structure and txt files.
    
    This dataset reads frame pairs from official KITTI-360 txt files which
    specify both RGB image paths and corresponding semantic label paths.
    
    Args:
        root: Path to KITTI-360 root (/misc/data/public/kitti-360/KITTI-360/)
        train: If True, use train split, else use val split
        data_type: 'image' or 'clip'
        clip_length: Number of frames per clip (for video training)
        train_H, train_W: Target training resolution (images will be resized)
        use_segmentation: If True, return semantic images
        return_semantic_ids: If True, return grayscale semantic IDs (not RGB)
    """
    
    # Official KITTI-360 paths
    KITTI360_ROOT = "/misc/data/public/kitti-360/KITTI-360"
    TRAIN_TXT = "/misc/data/public/kitti-360/KITTI-360/data_2d_semantics/train/2013_05_28_drive_train_frames.txt"
    VAL_TXT = "/misc/data/public/kitti-360/KITTI-360/data_2d_semantics/train/2013_05_28_drive_val_frames.txt"
    
    def __init__(
        self,
        root: str = None,  # Override with KITTI360_ROOT if not provided
        train: bool = True,
        target_transform=None,
        data_type: str = 'image',
        clip_length: int = None,
        if_return_prompt: bool = True,
        if_return_index: bool = True,
        if_return_calib: bool = False,
        if_return_bbox_im: bool = False,
        H: int = None,
        W: int = None,
        train_H: int = None,
        train_W: int = None,
        use_segmentation: bool = False,
        use_preplotted_bbox: bool = True,
        return_semantic_ids: bool = False,
        non_overlapping_clips: bool = False
    ):
        # Use official KITTI-360 root
        if root is None:
            root = self.KITTI360_ROOT
        
        self.kitti360_root = root
        self.use_segmentation = use_segmentation
        self.return_semantic_ids = return_semantic_ids
        self.use_preplotted_bbox = use_preplotted_bbox
        
        # Load frame pairs from txt file
        txt_file = self.TRAIN_TXT if train else self.VAL_TXT
        self.frame_pairs = self._load_frame_pairs(txt_file)
        
        print(f"✓ Loaded {len(self.frame_pairs)} frame pairs from {os.path.basename(txt_file)}")
        
        # Initialize parent class
        # Note: KittiAbstract expects certain directory structures, but we override
        # the key methods to use our txt-based loading
        super().__init__(
            root=root,
            train=train,
            target_transform=target_transform,
            data_type=data_type,
            clip_length=clip_length,
            if_return_prompt=if_return_prompt,
            if_return_index=if_return_index,
            if_return_calib=if_return_calib,
            if_return_bbox_im=if_return_bbox_im,
            H=H, W=W,
            train_H=train_H,
            train_W=train_W
        )
        
        # Override image_paths with our frame pairs
        self.image_paths = [pair['rgb_path'] for pair in self.frame_pairs]
        
        # Set up clip loading if needed
        if data_type == 'clip':
            self.non_overlapping_clips = non_overlapping_clips
            self._setup_clips()
    
    def _load_frame_pairs(self, txt_file: str) -> List[dict]:
        """
        Load RGB-semantic frame pairs from official KITTI-360 txt file.
        
        Returns:
            List of dicts with 'rgb_path', 'semantic_path', 'rgb_rel', 'semantic_rel'
        """
        frame_pairs = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse: "data_2d_raw/.../frame.png data_2d_semantics/.../frame.png"
                parts = line.split()
                if len(parts) != 2:
                    continue
                
                rgb_rel, semantic_rel = parts
                
                # Construct absolute paths
                rgb_path = os.path.join(self.kitti360_root, rgb_rel)
                semantic_path = os.path.join(self.kitti360_root, semantic_rel)
                
                # Verify files exist (at least check a few)
                if len(frame_pairs) < 10:  # Only check first 10 to save time
                    if not os.path.exists(rgb_path):
                        print(f"Warning: RGB image not found: {rgb_path}")
                        continue
                    if not os.path.exists(semantic_path):
                        print(f"Warning: Semantic image not found: {semantic_path}")
                        continue
                
                frame_pairs.append({
                    'rgb_path': rgb_path,
                    'semantic_path': semantic_path,
                    'rgb_rel': rgb_rel,
                    'semantic_rel': semantic_rel
                })
        
        return frame_pairs
    
    def _setup_clips(self):
        """
        Group frames into clips for video training.
        
        Since KITTI-360 has continuous sequences, we can create clips by
        grouping consecutive frames from the same sequence.
        """
        # Group frames by sequence
        from collections import defaultdict
        sequences = defaultdict(list)
        
        for idx, pair in enumerate(self.frame_pairs):
            # Extract sequence name from path
            # e.g., "data_2d_raw/2013_05_28_drive_0000_sync/..."
            seq_name = pair['rgb_rel'].split('/')[1]  # "2013_05_28_drive_0000_sync"
            sequences[seq_name].append(idx)
        
        # Create clips from consecutive frames
        self.clips = []
        for seq_name, indices in sequences.items():
            # Sort indices to ensure temporal order
            indices.sort(key=lambda i: self.frame_pairs[i]['rgb_path'])
            
            if self.non_overlapping_clips:
                # Non-overlapping clips
                for i in range(0, len(indices) - self.clip_length + 1, self.clip_length):
                    self.clips.append(indices[i:i+self.clip_length])
            else:
                # Overlapping clips (sliding window)
                for i in range(len(indices) - self.clip_length + 1):
                    self.clips.append(indices[i:i+self.clip_length])
        
        print(f"✓ Created {len(self.clips)} clips from {len(sequences)} sequences")
        
        # Compatibility with base class methods that expect clip_list and image_list
        self.clip_list = self.clips  # Alias for get_frame_file_by_index()
        self.image_list = [pair['rgb_path'] for pair in self.frame_pairs]  # For base class compatibility
    
    def get_bbox_image_file_by_index(self, index, image_file=None):
        """Get semantic visualization image path for compatibility with base class."""
        # Return the semantic RGB visualization path
        pair = self.frame_pairs[index] if index < len(self.frame_pairs) else self.frame_pairs[self.clips[index][0]]
        return pair['semantic_path']
    
    def __len__(self):
        """Return number of samples (frames or clips)."""
        if self.data_type == 'clip':
            return len(self.clips)
        else:
            return len(self.frame_pairs)
    
    def _getimageitem(self, index, return_prompt=True, return_calib=False, 
                      return_index=False, return_bbox_im=False):
        """
        Load a single frame (RGB + optional semantic).
        
        Overrides parent method to load from our txt-based frame pairs.
        """
        pair = self.frame_pairs[index]
        
        # Load RGB image
        rgb_image = Image.open(pair['rgb_path']).convert('RGB')
        
        # Resize if needed
        if self.train_H is not None and self.train_W is not None:
            rgb_image = rgb_image.resize((self.train_W, self.train_H), Image.LANCZOS)
        
        # Convert to tensor [C, H, W] in range [-1, 1]
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        # Prepare return tuple
        result = [rgb_tensor]
        
        # Load semantic image if requested
        if return_bbox_im and self.use_segmentation:
            if self.return_semantic_ids:
                # Load grayscale semantic IDs (returns numpy array)
                semantic_ids = load_and_remap_semantic(
                    pair['semantic_path'], 
                    ignore_index=255
                )
                
                # Convert to tensor if numpy
                if isinstance(semantic_ids, np.ndarray):
                    semantic_ids = torch.from_numpy(semantic_ids).long()
                
                # Resize if needed
                if self.train_H is not None and self.train_W is not None:
                    semantic_ids = torch.nn.functional.interpolate(
                        semantic_ids.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.train_H, self.train_W),
                        mode='nearest'
                    ).squeeze().long()
                
                # For visualization, create RGB semantic image (for bbox_images)
                # This is just for logging, not used in training
                semantic_rgb = self._semantic_ids_to_rgb(semantic_ids)
                result.append(semantic_rgb)
                
                # Add semantic_ids as separate item
                result.append(semantic_ids)
            else:
                # Load RGB semantic visualization (not used in training)
                semantic_rgb = Image.open(pair['semantic_path']).convert('RGB')
                if self.train_H is not None and self.train_W is not None:
                    semantic_rgb = semantic_rgb.resize((self.train_W, self.train_H), Image.LANCZOS)
                semantic_tensor = torch.from_numpy(np.array(semantic_rgb)).permute(2, 0, 1).float() / 127.5 - 1.0
                result.append(semantic_tensor)
        
        # Add index if requested
        if return_index:
            result.append(index)
        
        return tuple(result) if len(result) > 1 else result[0]
    
    def __getitem__(self, index, return_prompt=None, return_calib=None, return_index=None, return_bbox_im=None):
        """
        Override parent __getitem__ to properly handle instance variables.
        """
        # Use instance variables if parameters not explicitly provided
        if return_prompt is None:
            return_prompt = self.if_return_prompt
        if return_calib is None:
            return_calib = self.if_return_calib
        if return_index is None:
            return_index = self.if_return_index
        if return_bbox_im is None:
            return_bbox_im = self.if_return_bbox_im
        
        if self.data_type == "image":
            return self._getimageitem(index, return_prompt=return_prompt, return_calib=return_calib,
                                      return_index=return_index, return_bbox_im=return_bbox_im)
        elif self.data_type == "clip":
            return self._getclipitem(index, return_prompt=return_prompt, return_calib=return_calib,
                                      return_index=return_index, return_bbox_im=return_bbox_im)
    
    def _semantic_ids_to_rgb(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert semantic IDs (trainIDs 0-18) to RGB visualization for logging.
        Uses official KITTI-360 colors from kitti360scripts.
        
        Args:
            semantic_ids: [H, W] with trainIDs 0-18
        
        Returns:
            rgb_tensor: [3, H, W] in range [-1, 1]
        """
        # Build colormap from official kitti360scripts
        try:
            from kitti360scripts.helpers.labels import labels
            
            # Create mapping from trainId to color
            colormap = torch.zeros(19, 3, dtype=torch.float32)
            for label in labels:
                if hasattr(label, 'trainId') and 0 <= label.trainId < 19:
                    colormap[label.trainId] = torch.tensor(label.color, dtype=torch.float32)
        except ImportError:
            # Fallback to hardcoded colors if kitti360scripts not available
            colormap = torch.tensor([
                [128, 64, 128],   # 0: road
                [244, 35, 232],   # 1: sidewalk
                [70, 70, 70],     # 2: building
                [102, 102, 156],  # 3: wall
                [190, 153, 153],  # 4: fence
                [153, 153, 153],  # 5: pole
                [250, 170, 30],   # 6: traffic light
                [220, 220, 0],    # 7: traffic sign
                [107, 142, 35],   # 8: vegetation
                [152, 251, 152],  # 9: terrain
                [70, 130, 180],   # 10: sky
                [220, 20, 60],    # 11: person
                [255, 0, 0],      # 12: rider
                [0, 0, 142],      # 13: car
                [0, 0, 70],       # 14: truck
                [0, 60, 100],     # 15: bus
                [0, 80, 100],     # 16: train
                [0, 0, 230],      # 17: motorcycle
                [119, 11, 32],    # 18: bicycle
            ], dtype=torch.float32)
        
        H, W = semantic_ids.shape
        rgb = torch.zeros(3, H, W, dtype=torch.float32)
        
        for i in range(19):
            mask = (semantic_ids == i)
            rgb[:, mask] = colormap[i].unsqueeze(1)
        
        # Normalize to [-1, 1]
        rgb = rgb / 127.5 - 1.0
        
        return rgb
    
    def _getclipitem(self, index, return_prompt=True, return_calib=False,
                     return_index=False, return_bbox_im=False):
        """
        Load a clip of frames.
        
        Returns tuple matching base class format:
            clips: [T, C, H, W] tensor of RGB frames
            targets: dummy list
            prompt: string (if return_prompt=True)
            calib: dummy tensor (if return_calib=True)  
            index: int (if return_index=True)
            bbox_images: [T, C, H, W] tensor of semantic RGB (if return_bbox_im=True)
            semantic_ids: [T, H, W] tensor of trainIDs 0-18 (if return_bbox_im=True and return_semantic_ids=True)
        """
        clip_indices = self.clips[index]
        
        # Load all frames in clip
        clips = []
        bbox_images = []
        semantic_ids_list = []
        
        for frame_idx in clip_indices:
            result = self._getimageitem(
                frame_idx,
                return_prompt=False,
                return_calib=False,
                return_index=False,
                return_bbox_im=return_bbox_im
            )
            
            # Parse result tuple based on what was returned
            if return_bbox_im and self.use_segmentation and self.return_semantic_ids:
                # Returns: (rgb_frame, bbox_frame, semantic_ids)
                rgb_frame, bbox_frame, semantic_frame = result
                clips.append(rgb_frame)
                bbox_images.append(bbox_frame)
                semantic_ids_list.append(semantic_frame)
            elif return_bbox_im and self.use_segmentation:
                # Returns: (rgb_frame, bbox_frame)
                rgb_frame, bbox_frame = result
                clips.append(rgb_frame)
                bbox_images.append(bbox_frame)
            else:
                # Returns: rgb_frame only
                clips.append(result if isinstance(result, torch.Tensor) else result[0])
        
        # Stack frames
        clips = torch.stack(clips, dim=0)  # [T, C, H, W]
        
        # Build return tuple matching base class format with conditional items
        targets = [None] * len(clip_indices)  # Dummy list
        
        # Build return tuple conditionally based on flags (matching base class behavior)
        ret = (clips, targets)
        
        if return_prompt or self.if_return_prompt:
            prompt = "KITTI-360 sequence"
            ret += (prompt,)
        
        if return_calib or self.if_return_calib:
            # Dummy calib - KITTI-360 doesn't have calibration in this format
            calib = torch.eye(3, 4)  # Dummy 3x4 projection matrix
            ret += (calib,)
        
        if return_index or self.if_return_index:
            ret += (index,)
        
        if return_bbox_im or self.if_return_bbox_im:
            # Stack bbox_images and semantic_ids
            if bbox_images:
                bbox_images = torch.stack(bbox_images, dim=0)  # [T, C, H, W]
            else:
                bbox_images = torch.zeros_like(clips)  # Dummy
            ret += (bbox_images,)
            
            if self.return_semantic_ids:
                if semantic_ids_list:
                    semantic_ids = torch.stack(semantic_ids_list, dim=0)  # [T, H, W]
                else:
                    semantic_ids = torch.zeros(clips.shape[0], clips.shape[2], clips.shape[3], dtype=torch.long)
                ret += (semantic_ids,)
        
        return ret
