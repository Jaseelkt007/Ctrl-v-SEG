from .kitti_abstract import KittiAbstract
from PIL import Image
import torch
import os
import json


class Kitti360PreprocessedDataset(KittiAbstract):
    """
    KITTI360 Preprocessed Dataset for Ctrl-V bbox generation training.
    
    Dataset structure:
    kitti_preprocessed/
    ├── train/
    │   ├── clip_00000/
    │   │   ├── frames/
    │   │   │   ├── 0000000250.png
    │   │   │   └── ...
    │   │   ├── bboxes/
    │   │   │   ├── 0000000250.png
    │   │   │   └── ...
    │   │   └── annotations.json
    │   ├── clip_00001/
    │   └── ...
    └── val/
        └── ...
    """
    
    # Class mappings for KITTI360
    CLASS_IDS_LOOKUP = {
        'car': 1,
        'van': 2,
        'truck': 3,
        'pedestrian': 4,
        'person': 5,
        'cyclist': 6,
        'tram': 7,
        'bicycle': 8,
        'misc': 8,
    }
    
    IDS_CLASS_LOOKUP = {
        1: 'car',
        2: 'van',
        3: 'truck',
        4: 'pedestrian',
        5: 'person',
        6: 'cyclist',
        7: 'tram',
        8: 'bicycle',
    }

    def __init__(self,
                 root='./datasets',
                 train=True,
                 target_transform=None,
                 data_type='image',
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 H=None, W=None,
                 train_H=None, train_W=None,
                 use_preplotted_bbox=True,
                 non_overlapping_clips=False):

        # KITTI360 default resolution (can be adjusted based on preprocessed images)
        super(Kitti360PreprocessedDataset, self).__init__(
            root=root,
            train=train,
            target_transform=target_transform,
            data_type=data_type,
            clip_length=clip_length,
            if_return_prompt=if_return_prompt,
            if_return_index=if_return_index,
            if_return_calib=if_return_calib,
            if_return_bbox_im=if_return_bbox_im,
            H=376 if H is None else H,
            W=1408 if W is None else W,
            train_H=train_H, train_W=train_W,
            use_preplotted_bbox=use_preplotted_bbox
        )

        self.MAX_BOXES_PER_DATA = 15
        self._location = 'train' if self.train else 'val'
        self.version = 'kitti360_preprocessed'
        self.non_overlapping_clips = non_overlapping_clips

        # Path to preprocessed data
        self.data_dir = os.path.join(self.root, self._location)    
        
        # Get all clip folders
        self.clip_folders = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('clip_')
        ]) 
        
        # Load and parse all annotations
        self.clip_annotations = {}
        self.clip_folder_lengths = {}
        
        for clip_folder in self.clip_folders:
            annotation_path = os.path.join(self.data_dir, clip_folder, 'annotations.json')
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    self.clip_annotations[clip_folder] = json.load(f)
                    self.clip_folder_lengths[clip_folder] = self.clip_annotations[clip_folder]['num_frames']
            else:
                print(f"[Warning] No annotations.json found for {clip_folder}")
        
        # Build clip list for clip-based data type
        if self.data_type == 'clip':
            self.clip_list = []
            for clip_folder in self.clip_folders:
                num_frames = self.clip_folder_lengths[clip_folder]
                
                if not non_overlapping_clips:
                    # Overlapping clips (sliding window)
                    for start_idx in range(num_frames - self.clip_length + 1):
                        # Store (clip_folder, start_frame_idx) pairs
                        self.clip_list.append((clip_folder, start_idx))
                else:
                    # Non-overlapping clips
                    for start_idx in range(0, num_frames, self.clip_length):
                        if start_idx + self.clip_length <= num_frames:
                            self.clip_list.append((clip_folder, start_idx))
        
        print(f"[Kitti360PreprocessedDataset] Loaded {len(self.clip_folders)} clips "
              f"with {sum(self.clip_folder_lengths.values())} total frames")
        if self.data_type == 'clip':
            print(f"[Kitti360PreprocessedDataset] Created {len(self.clip_list)} clip samples")

    def _getimageitem(self, index, return_prompt=False, return_calib=False, 
                      return_index=False, return_bbox_im=False):
        """
        Get a single image and its annotations.
        
        Args:
            index: Global frame index across all clips
        
        Returns:
            image, target, [prompt], [calib], [index], [bbox_im]
        """
        # Convert global index to (clip_folder, frame_idx)
        clip_folder, frame_idx = self._get_clip_and_frame_index(index)
        
        # Get frame annotation
        clip_data = self.clip_annotations[clip_folder]
        frame_data = clip_data['frames'][frame_idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, clip_folder, frame_data['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Parse annotations
        target = self._parse_frame_objects(frame_data['objects'], frame_data['frame_id'])
        
        # Apply transforms
        if not self.transforms is None:
            image, target = self.transforms(image, target)
        
        # Build return tuple
        ret = (image, target,)
        
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer()
            ret += (prompt,)
        
        if return_calib or self.if_return_calib:
            ret += (None,)  # KITTI360 preprocessed doesn't have per-frame calib
        
        if return_index or self.if_return_index:
            ret += (index,)
        
        if return_bbox_im or self.if_return_bbox_im:
            if self.use_preplotted_bbox:
                # Load preplotted bbox image
                bbox_path = image_path.replace('frames/', 'bboxes/')
                bbox_im = Image.open(bbox_path).convert('RGB')
                if not self.transform is None:
                    bbox_im = self.transform(bbox_im)
            else:
                # Draw bbox on the fly
                bbox_im = self._draw_bbox(target, None)
            ret += (bbox_im,)
        
        return ret

    def _getclipitem(self, index, return_prompt=False, return_calib=False,
                     return_index=False, return_bbox_im=False):
        """
        Get a clip (sequence of frames) and their annotations.
        
        Args:
            index: Clip index
        
        Returns:
            images, targets, [prompt], [calib], [index], [bboxes]
        """
        clip_folder, start_frame_idx = self.clip_list[index]
        clip_data = self.clip_annotations[clip_folder]
        
        images = []
        targets = []
        bboxes = []
        
        if_return_bbox_im_cp = self.if_return_bbox_im
        self.set_if_return_bbox_im(False)
        
        if not self.if_return_bbox_im:
            prompt = self.prompt_engineer()
        
        self.disable_all_settings()
        
        for frame_offset in range(self.clip_length):
            frame_idx = start_frame_idx + frame_offset
            frame_data = clip_data['frames'][frame_idx] # from annotation.json
            
            # Load image
            image_path = os.path.join(self.data_dir, clip_folder, frame_data['image_path'])
            image = Image.open(image_path).convert('RGB')
            
            # Parse annotations
            target = self._parse_frame_objects(frame_data['objects'], frame_data['frame_id'])
            
            # Apply transforms
            if not self.transforms is None:
                image, target = self.transforms(image, target)
            
            images.append(image)
            targets.append(target)
            
            # Load bbox image if needed
            if if_return_bbox_im_cp or return_bbox_im:
                if self.use_preplotted_bbox:
                    bbox_path = image_path.replace('frames/', 'bboxes/')
                    bbox_im = Image.open(bbox_path).convert('RGB')
                    if not self.transform is None:
                        bbox_im = self.transform(bbox_im)
                else:
                    bbox_im = self._draw_bbox(target, None)
                bboxes.append(bbox_im)
        
        self.revert_setting()
        self.set_if_return_bbox_im(if_return_bbox_im_cp)
        
        images = torch.stack(images)
        ret = (images, targets,)
        
        if return_prompt or self.if_return_prompt:
            ret += (prompt,)
        
        if return_calib or self.if_return_calib:
            ret += (None,)
        
        if return_index or self.if_return_index:
            ret += (index,)
        
        if return_bbox_im or self.if_return_bbox_im:
            bboxes = torch.stack(bboxes)
            ret += (bboxes,)
        
        return ret

    def _get_clip_and_frame_index(self, global_index):
        """
        Convert global frame index to (clip_folder, frame_idx) pair.
        """
        current_count = 0
        for clip_folder in self.clip_folders:
            num_frames = self.clip_folder_lengths[clip_folder]
            if current_count + num_frames > global_index:
                frame_idx = global_index - current_count
                return clip_folder, frame_idx
            current_count += num_frames
        raise IndexError(f"Global index {global_index} out of range")

    def _parse_frame_objects(self, objects, frame_id):
        """
        Parse frame objects into the expected target format.
        
        Args:
            objects: List of object dictionaries from annotations.json
            frame_id: Frame identifier
        
        Returns:
            List of target dictionaries
        """
        target = []
        bbox_count = 0
        
        for obj in objects:
            # Map object type to class ID
            obj_type = obj['type'].lower()
            if obj_type not in self.CLASS_IDS_LOOKUP:
                continue
            
            target.append({
                'frame': frame_id,
                'trackID': obj['id'],
                'type': obj['type'],
                'truncated': float(obj.get('truncation', 0.0)),
                'occluded': int(obj.get('occlusion', 0)),
                'alpha': float(obj.get('alpha', 0.0)),
                'bbox': obj['bbox_2d'],  # [x1, y1, x2, y2]
                'dimensions': obj['dimensions'],  # [h, w, l]
                'location': obj['location'],  # [x, y, z]
                'rotation_y': float(obj.get('rotation_y', 0.0)),
                'id_type': self.CLASS_IDS_LOOKUP[obj_type]
            })
            
            bbox_count += 1
            if bbox_count >= self.MAX_BOXES_PER_DATA:
                break
        
        return target

    def prompt_engineer(self, *args):
        return "This is a real-world driving scene from the KITTI-360 dataset."

    def get_frame_file_by_index(self, index, timestep=0):
        """
        Get frame file path by clip index and timestep.
        
        Args:
            index: Clip index in clip_list
            timestep: Frame index within the clip
            
        Returns:
            str: Path to the frame file
        """
        if self.data_type == 'clip':
            clip_folder, start_frame_idx = self.clip_list[index]
            frame_idx = start_frame_idx + timestep
            
            # Get frame filename from annotations
            clip_data = self.clip_annotations[clip_folder]
            if frame_idx < len(clip_data['frames']):
                frame_path = clip_data['frames'][frame_idx]['image_path']
                return os.path.join(self.data_dir, clip_folder, frame_path)
            else:
                # Fallback to last frame if timestep out of bounds
                frame_path = clip_data['frames'][-1]['image_path']
                return os.path.join(self.data_dir, clip_folder, frame_path)
        else:
            raise NotImplementedError("get_frame_file_by_index not implemented for image data_type")

    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        """
        Get bbox image file path from frame image path.
        
        Args:
            index: Clip index (unused if image_file provided)
            image_file: Frame image file path
            
        Returns:
            str: Path to the bbox image file
        """
        if image_file is None:
            image_file = self.get_frame_file_by_index(index, 0)
        
        # Bbox images are stored in 'bboxes/' instead of 'frames/'
        return image_file.replace('frames/', 'bboxes/')

    def __len__(self):
        if self.data_type == 'image':
            return sum(self.clip_folder_lengths.values())
        else:  # data_type == 'clip'
            return len(self.clip_list)


if __name__ == "__main__":
    # Test dataset loading
    dataset = Kitti360PreprocessedDataset(
        root='/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/',
        train=True,
        data_type='clip',
        clip_length=8
    )
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a clip
    sample = dataset[0]
    print(f"Sample structure: images shape={sample[0].shape}, num targets={len(sample[1])}")
