from .kitti import KittiDataset
from .vkitti import VKittiDataset
from .mkitti import MergedKittiDataset
from .bdd100k import BDD100KDataset
from .davis import DAVISDataset
from .kitti360_preprocessed import Kitti360PreprocessedDataset
from .kitti360_official import KITTI360OfficialDataset

MAX_BOXES_PER_DATA = 30

def init_objects(len_target=0, device=None):
    import torch
    objects = dict( type         =   [None]*MAX_BOXES_PER_DATA,
                        truncated    =   torch.zeros(MAX_BOXES_PER_DATA, device=device),
                        occluded     =   torch.zeros(MAX_BOXES_PER_DATA, dtype=torch.long, device=device),
                        alpha        =   torch.zeros(MAX_BOXES_PER_DATA, device=device),
                        bbox         =   torch.zeros(MAX_BOXES_PER_DATA, 4, device=device),
                        dimensions   =   torch.zeros(MAX_BOXES_PER_DATA, 3, device=device),
                        locations    =   torch.zeros(MAX_BOXES_PER_DATA, 3, device=device),
                        rotation_y   =   torch.zeros(MAX_BOXES_PER_DATA, device=device),
                        id_type      =   torch.zeros(MAX_BOXES_PER_DATA, dtype=torch.long, device=device),
                        track_id     =   torch.zeros(MAX_BOXES_PER_DATA, dtype=torch.long, device=device),
                        num_objects  =   min(len_target, MAX_BOXES_PER_DATA) )
    return objects

def kitti_collate_fn(batch, tokenize_fn):
    import torch
    from collections import defaultdict
    collated_batch = {'pixel_values': [], 'objects': defaultdict(list), 'prompts': [], 'indices': [], 'bbox_images': []}
    for batch_i, frame, prompt_i, index_i, bbox_i in batch:
        collated_batch['pixel_values'].append(batch_i)
        collated_batch['prompts'].append(prompt_i)
        collated_batch['indices'].append(index_i)
        collated_batch['bbox_images'].append(bbox_i)

        # make sure that the number of objects is the same for all images
        len_target = len(frame) if not frame is None else 0
        objects = init_objects(len_target)
                        
        for object_i in range(objects['num_objects']):
            objects['type'][object_i] = frame[object_i]['type']
            objects['truncated'][object_i] = frame[object_i]['truncated']
            objects['occluded'][object_i] = frame[object_i]['occluded']
            objects['alpha'][object_i] = frame[object_i]['alpha']
            objects['bbox'][object_i] = torch.tensor(frame[object_i]['bbox'])
            objects['dimensions'][object_i] = torch.tensor(frame[object_i]['dimensions'])
            objects['locations'][object_i] = torch.tensor(frame[object_i]['location'])
            objects['rotation_y'][object_i] = frame[object_i]['rotation_y']
            objects['id_type'][object_i] = frame[object_i]['id_type']
        
        for key in objects:
            collated_batch['objects'][key].append(objects[key])
    
    collated_batch['pixel_values'] = torch.stack(collated_batch['pixel_values'])
    collated_batch['bbox_images'] = torch.stack(collated_batch['bbox_images'])
    for key in collated_batch['objects']:
        if not key == 'type' and not key == 'num_objects':
            collated_batch['objects'][key] = torch.stack(collated_batch['objects'][key])
    collated_batch['prompts'] = tokenize_fn(collated_batch['prompts']) if not tokenize_fn is None else collated_batch['prompts']
    return collated_batch

def kitti_clip_collate_fn(batch, tokenize_fn):
    import torch
    from collections import defaultdict
    collated_batch = {'clips': [], 'objects': defaultdict(list), 'prompts': [], 'indices': []}
    for batch_i, target_i, prompt_i, index_i in batch:
        collated_batch['clips'].append(batch_i)
        collated_batch['prompts'].append(prompt_i)
        collated_batch['indices'].append(index_i)
        frame_objects = defaultdict(list)

        for frame in target_i:
            
            # make sure that the number of objects is the same for all images
            len_target = len(frame) if not frame is None else 0
            objects = init_objects(len_target)
                        
            for object_i in range(objects['num_objects']):
                objects['type'][object_i] = frame[object_i]['type']
                objects['truncated'][object_i] = frame[object_i]['truncated']
                objects['occluded'][object_i] = frame[object_i]['occluded']
                objects['alpha'][object_i] = frame[object_i]['alpha']
                objects['bbox'][object_i] = torch.tensor(frame[object_i]['bbox'])
                objects['dimensions'][object_i] = torch.tensor(frame[object_i]['dimensions'])
                objects['locations'][object_i] = torch.tensor(frame[object_i]['location'])
                objects['rotation_y'][object_i] = frame[object_i]['rotation_y']
                objects['id_type'][object_i] = frame[object_i]['id_type']
                objects['track_id'][object_i] = frame[object_i]['trackID']
        
            for key in objects:
                frame_objects[key].append(objects[key])
        for key in frame_objects:
            if not key == 'type' and not key == 'num_objects':
                frame_objects[key] = torch.stack(frame_objects[key])
            collated_batch['objects'][key].append(frame_objects[key])
    
    if collated_batch['clips'][0] is not None:
        collated_batch['clips'] = torch.stack(collated_batch['clips'])
        
    for key in collated_batch['objects']:
        if not key == 'type' and not key == 'num_objects':
            collated_batch['objects'][key] = torch.stack(collated_batch['objects'][key])
    collated_batch['objects'] = dict(collated_batch['objects'])
    collated_batch['prompts'] = tokenize_fn(collated_batch['prompts']) if not tokenize_fn is None else collated_batch['prompts']
    return collated_batch

def kitti_clip_with_bbox_collate_fn(batch, tokenize_fn):
    import torch
    from collections import defaultdict
    collated_batch = {'clips': [], 'objects': defaultdict(list), 'prompts': [], 'indices': [], 'bbox_images': [], 'semantic_ids': []}
    for item in batch:
        collated_batch['clips'].append(item[0])
        objects_list = item[1]
        prompt = item[2]
        index = item[3]
        bbox_images = item[4]
        
        collated_batch['prompts'].append(prompt)
        collated_batch['indices'].append(index)
        collated_batch['bbox_images'].append(bbox_images)
        
        # Handle semantic IDs if present (when using semantic VAE)
        # Dataset can return:
        #   With calib: (clips, targets, prompt, calib, index, bbox_images, semantic_ids) - 7 items
        #   No calib:   (clips, targets, prompt, index, bbox_images, semantic_ids) - 6 items
        # semantic_ids is always the last item when present
        if len(item) > 5:
            semantic_ids = item[-1]  # Last item
            # Verify it's actually semantic_ids (should be int64 tensor)
            if hasattr(semantic_ids, 'dtype') and semantic_ids.dtype == torch.int64:
                collated_batch['semantic_ids'].append(semantic_ids)
        frame_objects = defaultdict(list)

        for frame in objects_list:
            
            # make sure that the number of objects is the same for all images
            len_target = len(frame) if not frame is None else 0
            objects = init_objects(len_target)
                        
            for object_i in range(objects['num_objects']):
                objects['type'][object_i] = frame[object_i]['type']
                objects['truncated'][object_i] = frame[object_i]['truncated']
                objects['occluded'][object_i] = frame[object_i]['occluded']
                objects['alpha'][object_i] = frame[object_i]['alpha']
                objects['bbox'][object_i] = torch.tensor(frame[object_i]['bbox'])
                objects['dimensions'][object_i] = torch.tensor(frame[object_i]['dimensions'])
                objects['locations'][object_i] = torch.tensor(frame[object_i]['location'])
                objects['rotation_y'][object_i] = frame[object_i]['rotation_y']
                objects['id_type'][object_i] = frame[object_i]['id_type']
                objects['track_id'][object_i] = frame[object_i]['trackID']
        
            for key in objects:
                frame_objects[key].append(objects[key])
        for key in frame_objects:
            if not key == 'type' and not key == 'num_objects':
                frame_objects[key] = torch.stack(frame_objects[key])
            collated_batch['objects'][key].append(frame_objects[key])
    
    if collated_batch['clips'][0] is not None:
        collated_batch['clips'] = torch.stack(collated_batch['clips'])

    collated_batch['bbox_images'] = torch.stack(collated_batch['bbox_images'])
    if collated_batch['semantic_ids']:
        collated_batch['semantic_ids'] = torch.stack(collated_batch['semantic_ids'])
    else:
        collated_batch['semantic_ids'] = None
    for key in collated_batch['objects']:
        if not key == 'type' and not key == 'num_objects':
            collated_batch['objects'][key] = torch.stack(collated_batch['objects'][key])
    collated_batch['objects'] = dict(collated_batch['objects'])
    collated_batch['prompts'] = tokenize_fn(collated_batch['prompts']) if not tokenize_fn is None else collated_batch['prompts']
    return collated_batch

COCO_LABELS_LOOKUP = {
    0   :   'person',
    1   :   'bicycle',
    2   :   'car',
    3   :   'motorcycle',
    4   :   'airplane',
    5   :   'bus',
    6   :   'train',
    7   :   'truck',
    8   :   'boat',
    9   :   'traffic light',
    10  :   'fire hydrant',
    11  :   'street sign',
    12  :   'stop sign',
    13  :   'parking meter',
    14  :   'bench',
}

__all__ = ['KittiDataset', 'VKittiDataset', 'MergedKittiDataset', 'BDD100KDataset', 'DAVISDataset',
           'Kitti360PreprocessedDataset',  # NuScenesDataset 
           'kitti_collate_fn', 'kitti_clip_collate_fn', 'kitti_clip_with_bbox_collate_fn', 'init_objects',
           'MAX_BOXES_PER_DATA', 'COCO_LABELS_LOOKUP']