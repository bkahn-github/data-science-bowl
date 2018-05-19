import os
import random
import glob
import numpy as np

import cv2
import skimage

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms, utils

from config import config
from utils import get_path

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]

        h, w = img.shape[:2]

        if h != self.size:
            top = np.random.randint(0, h - self.size)
            left = np.random.randint(0, w - self.size)
        else:
            top = 0
            left = 0

        img = img[top: top + self.size, left: left + self.size]
        mask = mask[top: top + self.size, left: left + self.size]

        return img, mask

class FlipLR(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        if random.random() < self.p:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            return img, mask
        return img, mask

class FlipUD(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        if random.random() < self.p:
            img = img[::-1].copy()
            mask = mask[::-1].copy()
            return img, mask
        return img, mask

class Rotate(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, sample):
        img, mask = sample[0], sample[1]

        angle = random.randint(0, self.max_angle)
        
        img = skimage.transform.rotate(img, angle, preserve_range=True)
        mask = skimage.transform.rotate(mask, angle, preserve_range=True)

        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        return img, mask

class CLAHE(object):
    def __init__(self, cliplimit, gridSize):
        self.cliplimit = cliplimit
        self.gridSize = gridSize

    def __call__(self, sample):
        img, mask = sample[0], sample[1]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit, tileGridSize=self.gridSize)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
        return img, mask

class InvertImages(object):
    def __init__(self, invert):
        self.invert = invert

    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        img_gray = img[:,:,0]
        if np.mean(img_gray) > self.invert:
            img = 255 - img
    
        return img, mask

class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        
        return img, mask
    
def augmentation(img, mask):
    flipLR = FlipLR(p=config.FLIP_LR)
    flipUD = FlipUD(p=config.FLIP_UD)
    rotate = Rotate(max_angle=config.ROTATE)

    img, mask = flipLR([img, mask])
    img, mask = flipUD([img, mask])
    img, mask = rotate([img, mask])

    return img, mask

class TrainDataset(Dataset):
    def __init__(self, ids, augmentation=None):
        
        self.ids = ids
        self.augmentation = augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        clahe = CLAHE(cliplimit=config.CLIP_LIMIT, gridSize=(config.GRID_SIZE, config.GRID_SIZE))
        invertImages = InvertImages(invert=config.INVERT)
        randomCrop = RandomCrop(size=config.RANDOM_CROP)
        toTensor = ToTensor()

        img_path, masks_path = get_path(id)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(masks_path, cv2.IMREAD_COLOR)
        
        img, mask = clahe([img, mask])
        img, mask = invertImages([img, mask])
        img, mask = randomCrop([img, mask])

        if config.AUGMENT:
            img, mask = self.augmentation(img, mask)

        img, mask = toTensor([img, mask])

        return {'img': img, 'mask': mask}
