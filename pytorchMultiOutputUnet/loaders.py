import os
import glob
import cv2
import random
import numpy as np

import torch
import torchvision

from utils import get_path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
from PIL import Image

class RandomCrop(object):
    def __call__(self, sample, size=0.5):
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

class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample[0], sample[1]
        
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        
        return img, mask
    
def augmentation(img, mask):
    randomCrop = RandomCrop()
    toTensor = ToTensor()
    flipLR = FlipLR()
    flipUD = FlipUD()
    
    img, mask = randomCrop([img, mask], 256)
    img, mask = flipLR([img, mask], 0.5)
    img, mask = flipUD([img, mask], 0.5)
    img, mask = toTensor([img, mask])    

    return img, mask

class TrainDataset(Dataset):
    def __init__(self, ids, augmentation=None):
        
        self.ids = ids
        self.augmentation = augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        img_path, targets_path = get_path(id)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.imread(targets_path, cv2.IMREAD_COLOR)
        
        img, target = self.augmentation(img, target)

        return {'img': img, 'target': target}