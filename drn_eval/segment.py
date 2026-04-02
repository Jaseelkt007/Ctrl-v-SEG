#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms.functional as TR

import drn
import data_transforms as transforms

try:
    from modules import batchnormsync
except ImportError:
    pass

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
Label_kitti = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )

labels_kitti = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label_kitti(  'unlabeled'            ,  0 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label_kitti(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label_kitti(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label_kitti(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label_kitti(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label_kitti(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label_kitti(  'ground'               ,  6 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label_kitti(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label_kitti(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label_kitti(  'parking'              ,  9 ,        2 ,       255 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label_kitti(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label_kitti(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label_kitti(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label_kitti(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label_kitti(  'guard rail'           , 14 ,        30,       255 , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label_kitti(  'bridge'               , 15 ,        31,       255 , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label_kitti(  'tunnel'               , 16 ,        32,       255 , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label_kitti(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label_kitti(  'polegroup'            , 18 ,       -1 ,       255 , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label_kitti(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label_kitti(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label_kitti(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label_kitti(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label_kitti(  'sky'                  , 23 ,        9 ,        10 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label_kitti(  'person'               , 24 ,        19,        11 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label_kitti(  'rider'                , 25 ,        20,        12 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label_kitti(  'car'                  , 26 ,        13,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label_kitti(  'truck'                , 27 ,        14,        14 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label_kitti(  'bus'                  , 28 ,        34,        15 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label_kitti(  'caravan'              , 29 ,        16,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label_kitti(  'trailer'              , 30 ,        15,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label_kitti(  'train'                , 31 ,        33,        16 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label_kitti(  'motorcycle'           , 32 ,        17,        17 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label_kitti(  'bicycle'              , 33 ,        18,        18 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label_kitti(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label_kitti(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label_kitti(  'stop'                 , 36 ,        29,       255 , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label_kitti(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label_kitti(  'lamp'                 , 38 ,        25,       255 , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label_kitti(  'trash bin'            , 39 ,        26,       255 , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label_kitti(  'vending machine'      , 40 ,        27,       255 , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label_kitti(  'box'                  , 41 ,        28,       255 , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label_kitti(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label_kitti(  'unknown vehicle'      , 43 ,        36,       255 , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label_kitti(  'unknown object'       , 44 ,        37,       255 , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label_kitti(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


def id2label_kitti(image):
    array = np.array(image)
    out_array = np.full(array.shape, -1, dtype=np.int32)  # Initialize with a default value, e.g., -1
    for l in labels_kitti:
        out_array[array == l.id] = l.trainId
    return Image.fromarray(out_array)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.confidence_list = None
        self.read_lists()

    def __getitem__(self, index):
        # Read the image
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        size = data[0].size
        # Read the label if available
        if self.label_list is not None:
            label_image = Image.open(join(self.data_dir, self.label_list[index]))
            if data[0].size != label_image.size:
                # Resize the label to match the image size using nearest neighbor interpolation
                label_image = label_image.resize(data[0].size, Image.NEAREST)
            data.append(id2label_kitti(label_image))

            # data.append(id2label_kitti(Image.open(
            #     join(self.data_dir, self.label_list[index]))))
        data = list(self.transforms(*data))
        
        
        # Read the confidence file if available
        if self.confidence_list is not None:
            
            class ToTensor16Bit(transforms.ToTensor):
                def __call__(self, pic):
                    return (torch.from_numpy(np.array(pic, np.int16, copy=False)).view(1, pic.size[1], pic.size[0]).float() / 65535) + 0.5


            conf_image = Image.open(join(self.data_dir, self.confidence_list[index]))
            
            # Convert I;16 mode to numpy array for resizing (PIL can't resize I;16 directly)
            if conf_image.mode == 'I;16':
                import cv2
                conf_array = np.array(conf_image, dtype=np.uint16)
                if size != conf_image.size:
                    # Resize using cv2 (supports uint16)
                    conf_array = cv2.resize(conf_array, size, interpolation=cv2.INTER_LINEAR)
                # Convert back to PIL Image for ToTensor16Bit
                conf_image = Image.fromarray(conf_array, mode='I;16')
            else:
                if size != conf_image.size:
                    # Resize the label to match the image size using nearest neighbor interpolation
                    conf_image = conf_image.resize(size, Image.BICUBIC)
                
                
            confidence_image = ToTensor16Bit()(conf_image)


            # print(np.array(Image.open(join(self.data_dir, self.confidence_list[index]))).shape)
            # confidence_image = transforms.ToTensor()(Image.open(join(self.data_dir, self.confidence_list[index])).convert('L'))
            data.append(confidence_image)
        
        # Apply transformations to the data
        
        # Handle the output name if specified
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        
        return tuple(data)


    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
            
            # Generate the confidence list by replacing "semantic" with "confidence" in each label path
            # Only set confidence_list if at least one confidence file exists
            potential_conf_list = [line.replace("semantic", "confidence") for line in self.label_list]
            if potential_conf_list and exists(join(self.data_dir, potential_conf_list[0])):
                self.confidence_list = potential_conf_list
            else:
                self.confidence_list = None


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.confidence_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        size = data[0].size
        w, h = data[0].size

        if self.label_list is not None:
            label_image = Image.open(join(self.data_dir, self.label_list[index]))
            if data[0].size != label_image.size:
                # Resize the label to match the image size using nearest neighbor interpolation
                label_image = label_image.resize(data[0].size, Image.NEAREST)
            data.append(id2label_kitti(label_image))

            # data.append(id2label_kitti(Image.open(
            #     join(self.data_dir, self.label_list[index]))))
        # data = list(self.transforms(*data))

        out_data = list(self.transforms(*data))
        
         # Read the confidence file if available
        if self.confidence_list is not None:
            
            class ToTensor16Bit(transforms.ToTensor):
                def __call__(self, pic):
                    return (torch.from_numpy(np.array(pic, np.int16, copy=False)).view(1, pic.size[1], pic.size[0]).float() / 65535) + 0.5


            conf_image = Image.open(join(self.data_dir, self.confidence_list[index]))
            
            # Convert I;16 mode to numpy array for resizing (PIL can't resize I;16 directly)
            if conf_image.mode == 'I;16':
                import cv2
                conf_array = np.array(conf_image, dtype=np.uint16)
                if size != conf_image.size:
                    # Resize using cv2 (supports uint16)
                    conf_array = cv2.resize(conf_array, size, interpolation=cv2.INTER_LINEAR)
                # Convert back to PIL Image for ToTensor16Bit
                conf_image = Image.fromarray(conf_array, mode='I;16')
            else:
                if size != conf_image.size:
                    # Resize the label to match the image size using nearest neighbor interpolation
                    conf_image = conf_image.resize(size, Image.BICUBIC)
                
                
            confidence_image = ToTensor16Bit()(conf_image)




            # print(np.array(Image.open(join(self.data_dir, self.confidence_list[index]))).shape)
            # confidence_image = transforms.ToTensor()(Image.open(join(self.data_dir, self.confidence_list[index])).convert('L'))
            out_data.append(confidence_image)
            
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
            # Generate the confidence list by replacing "semantic" with "confidence" in each label path
            # Only set confidence_list if at least one confidence file exists
            potential_conf_list = [line.replace("semantic", "confidence") for line in self.label_list]
            if potential_conf_list and exists(join(self.data_dir, potential_conf_list[0])):
                self.confidence_list = potential_conf_list
            else:
                self.confidence_list = None



def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.item()


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(non_blocking=True)


        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose(t),
                list_dir=args.list_dir),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

                
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              eval_score=accuracy)

        with torch.no_grad():
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, eval_score=accuracy)

        checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth.tar')

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.save_iter == 0:
            history_path = os.path.join(args.save_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_weighted(pred, label, confidence, n):
    k = (label >= 0) & (label < n)
    # Incorporate confidence into the bincount calculation
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], weights=confidence[k], minlength=n ** 2
    )
    return bin_count.reshape(n, n)

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        
        # Detect if confidence is available
        # Structure: [image, label, (confidence), name]
        # With confidence: length = 4
        # Without confidence: length = 3
        if len(input_data) == 4:
            image, label, confidence, name = input_data
            has_confidence = True
            if iter == 0:
                logger.info(f"test(): Using confidence mode, len={len(input_data)}")
        elif len(input_data) == 3:
            image, label, name = input_data
            has_confidence = False
            confidence = None
            if iter == 0:
                logger.info(f"test(): Using standard mode, len={len(input_data)}")
        else:
            logger.error(f"Unexpected data structure: len={len(input_data)}")
            raise ValueError(f"Unexpected input_data length: {len(input_data)}")
        
        with torch.no_grad():
            image_var = Variable(image, requires_grad=False)
            final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        # if save_vis:
        #     save_output_images(pred, name, output_dir)
        #     save_colorful_images(
        #         pred, name, output_dir + '_color',
        #         TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            
            if has_confidence:
                # Use weighted histogram if confidence is available
                confidence_np = confidence[0].numpy() / 255.0
                hist += fast_hist_weighted(pred.flatten(), label.flatten(), confidence_np.flatten(), num_classes)
            else:
                # Use standard unweighted histogram (standard mIoU calculation)
                hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False, save_hist_path=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            # Detect if confidence is available
            # Structure: [image, label, (confidence), name, ...ms_images (num_scales)]
            # Without confidence: [image, label, name, ms1, ms2, ..., ms5] = 3 + num_scales
            # With confidence:    [image, label, confidence, name, ms1, ms2, ..., ms5] = 4 + num_scales
            expected_len_with_conf = 4 + num_scales
            expected_len_without_conf = 3 + num_scales
            
            # Debug on first iteration
            if iter == 0:
                logger.info(f"Data structure: len={len(input_data)}, expected_with_conf={expected_len_with_conf}, expected_without_conf={expected_len_without_conf}")
                logger.info(f"  input_data[2] type: {type(input_data[2])}")
            
            # Simpler detection: check if length matches
            has_confidence = len(input_data) == expected_len_with_conf
            
            if has_confidence:
                label = input_data[1]
                confidence = input_data[2]
                name = input_data[3]
                if iter == 0:
                    logger.info(f"Using confidence mode - name type: {type(name)}, value: {str(name)[:100]}")
            else:
                label = input_data[1]
                confidence = None
                name = input_data[2]
                if iter == 0:
                    logger.info(f"Using standard mode - name type: {type(name)}, value: {str(name)[:100]}")
        else:
            name = input_data[1]
            has_confidence = False
        
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        with torch.no_grad():
            for image in images:
                image_var = Variable(image, requires_grad=False)
                final = model(image_var)[0]
                outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        # if save_vis:
        #     save_output_images(pred, name, output_dir)
        #     save_colorful_images(pred, name, output_dir + '_color',
        #                          CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            
            if has_confidence:
                # Use weighted histogram if confidence is available
                confidence_np = confidence[0].numpy() / 255.0
                hist += fast_hist_weighted(pred.flatten(), label.flatten(), confidence_np.flatten(), num_classes)
            else:
                # Use standard unweighted histogram (standard mIoU calculation)
                hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        if save_hist_path:
            np.save(save_hist_path, hist)
            logger.info('Confusion matrix saved to %s', save_hist_path)
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        # Load the original state dict with 'module.' prefix
        state_dict = torch.load(args.pretrained)["state_dict"]

        # Create a new state dict without the 'module.' prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the new state dict into your model
        single_model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), scales, list_dir=args.list_dir)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    hist_save_path = join(data_dir, 'confusion_matrix.npy') if args.ms else None
    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales,
                      save_hist_path=hist_save_path)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    logger.info('mAP: %f', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                        help='output path for training checkpoints')
    parser.add_argument('--save_iter', default=1, type=int,
                        help='number of training iterations between'
                             'checkpoint history saves')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        with torch.no_grad():
            test_seg(args)


if __name__ == '__main__':
    main()
