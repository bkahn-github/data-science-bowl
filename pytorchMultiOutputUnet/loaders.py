import os
import glob
import cv2
import random
import numpy as np

import torch
import torchvision

from utils import get_path
from config import config

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import skimage
from skimage import io, transform
from PIL import Image

class RandomCrop(object):
    def __call__(self, sample, size):
        img = sample[0]
        mask = sample[1]

        h, w = img.shape[:2]

        if h != size:
            top = np.random.randint(0, h - size)
            left = np.random.randint(0, w - size)
        else:
            top = 0
            left = 0

        img = img[top: top + size, left: left + size]
        mask = mask[top: top + size, left: left + size]

        return img, mask

class FlipLR(object):
    def __call__(self, sample, p):
        img, mask = sample[0], sample[1]
        
        if random.random() < p:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            return img, mask
        return img, mask

class FlipUD(object):
    def __call__(self, sample, p):
        img, mask = sample[0], sample[1]
        
        if random.random() < p:
            img = img[::-1].copy()
            mask = mask[::-1].copy()
            return img, mask
        return img, mask

class Rotate(object):
    def __call__(self, sample, max_angle):
        img, mask = sample[0], sample[1]

        angle = random.randint(0, max_angle)
        
        img = skimage.transform.rotate(img, angle, preserve_range=True)
        mask = skimage.transform.rotate(mask, angle, preserve_range=True)

        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        return img, mask

class CLAHE(object):
    def __call__(self, sample, cliplimit, gridSize):
        img, mask = sample[0], sample[1]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=gridSize)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
        return img, mask

class InvertImages(object):
    def __call__(self, sample, invert):
        img, mask = sample[0], sample[1]
        
        img_gray = img[:,:,0]
        if np.mean(img_gray) > invert:
            img = 255 - img
    
        return img, mask

class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        
        return img, mask
    
def augmentation(img, mask):
    flipLR = FlipLR()
    flipUD = FlipUD()
    rotate = Rotate()

    img, mask = flipLR([img, mask], config.FLIPLR)
    img, mask = flipUD([img, mask], config.FLIPUD)
    img, mask = rotate([img, mask], config.ROTATE)

    return img, mask

class TrainDataset(Dataset):
    def __init__(self, ids, augmentation=None):
        
        self.ids = ids
        self.augmentation = augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        clahe = CLAHE()
        invertImages = InvertImages()
        randomCrop = RandomCrop()
        toTensor = ToTensor()

        img_path, masks_path = get_path(id)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(masks_path, cv2.IMREAD_COLOR)
        
        img, mask = clahe([img, mask], config.CLIPLIMIT, config.GRIDSIZE)
        img, mask = invertImages([img, mask], config.INVERT)
        img, mask = randomCrop([img, mask], config.RANDOMCROP)

        if config.AUGMENT:
            img, mask = self.augmentation(img, mask)

        img, mask = toTensor([img, mask])

        return {'img': img, 'mask': mask}