"""
KITTI-360 Dataset in BDD100K Format for Ctrl-V

This is a thin wrapper around BDD100KDataset that makes it easy to use
KITTI-360 data preprocessed in BDD100K format.

Usage:
    from ctrlv.datasets.kitti360_bdd_format import KITTI360BDDDataset
    
    dataset = KITTI360BDDDataset(
        root='/no_backups/s1492/',  # Parent directory of kitti360_ctrlv/
        train=True,
        data_type='clip',
        clip_length=8,
        if_return_bbox_im=True,
        use_preplotted_bbox=True
    )
"""

from .bdd100k import BDD100KDataset


class KITTI360BDDDataset(BDD100KDataset):
    """
    KITTI-360 dataset using BDD100K-compatible format.
    
    This class inherits all functionality from BDD100KDataset and only
    changes the version name to 'kitti360_ctrlv' (or customizable).
    
    The dataset should be preprocessed using preprocess_kitti360_bdd_format.py
    to create the following structure:
    
    root/
      └── kitti360_ctrlv/  (or custom name)
          ├── images/track/train/ & val/
          ├── labels/box_track_20/train/ & val/
          └── bboxes/track/train/ & val/ (optional)
    """
    
    # KITTI-360 specific class IDs (compatible with BDD100K)
    IDS_CLASS_LOOKUP = {
        1: 'pedestrian',
        2: 'cyclist',
        3: 'car',
        4: 'truck',
        5: 'bus',
        6: 'train',
        7: 'motorcycle',
        8: 'bicycle',
    }

    CLASS_IDS_LOOKUP = {
        'pedestrian': 1,
        'person': 1,
        'cyclist': 2,
        'car': 3,
        'van': 3,
        'truck': 4,
        'caravan': 4,
        'trailer': 4,
        'bus': 5,
        'train': 6,
        'motorcycle': 7,
        'bicycle': 8,
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
                 use_segmentation=False,
                 use_preplotted_bbox=True,
                 version='kitti360_ctrlv'):
        """
        Initialize KITTI-360 dataset in BDD100K format.
        
        Args:
            root: Parent directory containing the preprocessed dataset
            train: If True, load training set; otherwise validation set
            data_type: 'image' or 'clip'
            clip_length: Number of frames per clip (required if data_type='clip')
            if_return_bbox_im: Return bbox overlay images
            use_preplotted_bbox: Use pregenerated bbox images (faster)
            version: Name of the dataset directory (default: 'kitti360_ctrlv')
            H, W: Image height and width (default: 376x1408 for KITTI-360)
            Other args: Same as BDD100KDataset
        """
        # Initialize parent class with all parameters
        super(BDD100KDataset, self).__init__(
            root=root,
            train=train,
            target_transform=target_transform,
            data_type=data_type,
            clip_length=clip_length,
            if_return_prompt=if_return_prompt,
            if_return_index=if_return_index,
            if_return_calib=if_return_calib,
            if_return_bbox_im=if_return_bbox_im,
            H=376 if H is None else H,  # KITTI-360 default height
            W=1408 if W is None else W,  # KITTI-360 default width
            train_H=train_H,
            train_W=train_W,
            use_preplotted_bbox=use_preplotted_bbox
        )
        
        # Override version to use KITTI-360 directory
        self.version = version
        self.MAX_BOXES_PER_DATA = 30
        self._location = 'train' if self.train else 'val'
        self.use_segmentation = use_segmentation

        # Set up paths (same structure as BDD100K)
        import os
        self.image_dir = os.path.join(self.root, self.version, 
                                      BDD100KDataset.TO_IMAGE_DIR, self._location)
        self.bbox_label_dir = os.path.join(self.root, self.version, 
                                           BDD100KDataset.TO_BBOX_LABELS, self._location)

        # Load clip folders (always use image directory structure)
        listed_image_dir = os.listdir(self.image_dir)
        try:
            listed_image_dir.remove('pred')
        except:
            pass
        self.clip_folders = sorted(listed_image_dir)
        self.clip_folder_lengths = {
            k: len(os.listdir(os.path.join(self.image_dir, k))) 
            for k in self.clip_folders
        }

        if self.data_type == 'clip':
            for l in self.clip_folder_lengths.values():
                assert l >= self.clip_length, \
                    f'clip length {self.clip_length} is too long for clip folder length {l}'
    
    def get_image_file_by_index(self, index):
        """
        Override to use .png extension for KITTI-360 images.
        
        This method is called by _getimageitem() to get the image path.
        BDD100K hardcodes .jpg on line 237, but KITTI-360 uses .png.
        """
        import os
        index += 1
        image_counter = 0
        clip_folder_counter = 0
        while image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]] < index:
            image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
            clip_folder_counter += 1
        return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                            f'{self.clip_folders[clip_folder_counter]}-{index - image_counter:07d}.png')
    
    def get_frame_file_by_index(self, index, timestep=None):
        """
        Override to use .png extension for KITTI-360 images.
        
        This method is called by get_first_training_sample() for demo samples.
        BDD100K hardcodes .jpg on lines 277 and 282, but KITTI-360 uses .png.
        """
        import os
        if self.train:
            index += 1
            clip_folder_counter = 0
            clip_counter = 0
            while clip_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1) < index:
                clip_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1
                clip_folder_counter += 1
            
            start_frame = index - clip_counter
            if timestep is None:
                ret = []
                for i in range(self.clip_length):
                    ret.append(os.path.join(self.image_dir, self.clip_folders[clip_folder_counter], f'{start_frame+i:07d}.png'))
                return ret
            assert timestep < self.clip_length
            curr_frame = start_frame+timestep
            return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                f'{self.clip_folders[clip_folder_counter]}-{curr_frame:07d}.png')
        else:
            indices = self.get_clip_frame_indices(index)
            if timestep is None:
                ret = []
                for i in range(self.clip_length):
                    ret.append(self.get_image_file_by_index(indices[i]))
                return ret
            return self.get_image_file_by_index(indices[timestep])
    
    def get_frame_file(self, index, timestep=None):
        """
        Override to use .png extension for KITTI-360 images.
        
        BDD100K uses .jpg, but KITTI-360 uses .png files.
        """
        import os
        
        if self.data_type == 'image':
            image_counter = 0
            clip_folder_counter = 0
            while index >= image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]:
                image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
                clip_folder_counter += 1
            return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                f'{self.clip_folders[clip_folder_counter]}-{index - image_counter:07d}.png')
        
        else:  # data_type == 'clip'
            # For non-overlapping clips
            if self.non_overlapping_clips:
                clip_folder_counter = 0
                image_counter = 0
                while index >= image_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]] // self.clip_length):
                    image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]] // self.clip_length
                    clip_folder_counter += 1
                start_frame = (index - image_counter) * self.clip_length + 1
                
                if timestep is None:
                    ret = []
                    for i in range(self.clip_length):
                        ret.append(os.path.join(self.image_dir, self.clip_folders[clip_folder_counter], f'{start_frame+i:07d}.png'))
                    return ret
                assert timestep < self.clip_length
                curr_frame = start_frame + timestep
                return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                    f'{self.clip_folders[clip_folder_counter]}-{curr_frame:07d}.png')
            else:
                # For overlapping clips (default)
                indices = self.get_clip_frame_indices(index)
                if timestep is None:
                    image_list = []
                    for idx in indices:
                        image_counter = 0
                        clip_folder_counter = 0
                        while idx >= image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]:
                            image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
                            clip_folder_counter += 1
                        image_file = os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                                  f'{self.clip_folders[clip_folder_counter]}-{idx - image_counter + 1:07d}.png')
                        image_list.append(image_file)
                    return image_list
                else:
                    idx = indices[timestep]
                    image_counter = 0
                    clip_folder_counter = 0
                    while idx >= image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]:
                        image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
                        clip_folder_counter += 1
                    return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                        f'{self.clip_folders[clip_folder_counter]}-{idx - image_counter + 1:07d}.png')
    
    def prompt_engineer(self, *args):
        """Return KITTI-360 specific prompt."""
        return 'This is a real-world driving scene from the KITTI-360 dataset.'


# Convenience function for quick dataset creation
def create_kitti360_dataset(root='/no_backups/s1492/',
                           train=True,
                           clip_length=8,
                           version='kitti360_ctrlv',
                           **kwargs):
    """
    Convenience function to create KITTI-360 dataset.
    
    Args:
        root: Parent directory of preprocessed dataset
        train: Training or validation split
        clip_length: Frames per clip
        version: Dataset directory name
        **kwargs: Additional arguments passed to KITTI360BDDDataset
    
    Returns:
        KITTI360BDDDataset instance
    
    Example:
        >>> train_dataset = create_kitti360_dataset(
        ...     root='/no_backups/s1492/',
        ...     train=True,
        ...     clip_length=8,
        ...     if_return_bbox_im=True
        ... )
    """
    return KITTI360BDDDataset(
        root=root,
        train=train,
        data_type='clip',
        clip_length=clip_length,
        version=version,
        **kwargs
    )


if __name__ == '__main__':
    # Test dataset loading
    print("Testing KITTI360BDDDataset...")
    
    try:
        dataset = KITTI360BDDDataset(
            root='/no_backups/s1492/',
            train=True,
            data_type='clip',
            clip_length=8,
            if_return_bbox_im=False
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Length: {len(dataset)}")
        print(f"  Clip folders: {len(dataset.clip_folders)}")
        print(f"  First clip: {dataset.clip_folders[0]}")
        
        # Try to load first sample
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"✓ Sample loaded")
        print(f"  Images shape: {sample[0].shape}")
        print(f"  Num targets: {len(sample[1])}")
        if len(sample[1]) > 0:
            print(f"  First target keys: {list(sample[1][0].keys())}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
