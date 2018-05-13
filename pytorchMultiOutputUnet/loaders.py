import os
import glob
import cv2
import numpy as np

import torch
import torchvision

import imgaug
from imgaug import augmenters as iaa

from utils import get_path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
from PIL import Image

seq = iaa.Sequential([iaa.Noop()])

def randomCrop(image, mask, output_size):
    h, w = image.shape[:2]
    new_h, new_w = output_size, output_size
    
    if h != new_h:
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
    else:
        top = 0
        left = 0
        
    image = image[top: top + new_h, left: left + new_w]
    mask = mask[top: top + new_h, left: left + new_w]

    return image, mask

class TrainDataset(Dataset):
    def __init__(self, ids, x_transform=None, target_transforms=None):
        
        self.ids = ids

        self.x_transform = x_transform
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        img_path, targets_path = get_path(id)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.imread(targets_path, cv2.IMREAD_COLOR)
        
        seq_det = seq.to_deterministic()
        img = seq_det.augment_image(img)
        target = seq_det.augment_image(target)

        img, target = randomCrop(img, target, 256)

        img = self.x_transform(img)
        target = self.target_transforms(target)

        return {'img': img, 'target': target}


x_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

target_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])