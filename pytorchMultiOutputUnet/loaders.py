import os
import glob
import numpy as np

import torch
import torchvision

from utils import get_ids, get_path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, ids, x_transform=None, y_transform=None):
        
        self.ids = ids

        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        img_path, mask_path, contour_path, center_path = get_path(id)
        
        img = Image.open(img_path)
        img.load()

        RGBimg = Image.new("RGB", img.size, (255, 255, 255))
        RGBimg.paste(img, mask=img.split()[3])
        
        img = np.array(RGBimg)
        
        mask = io.imread(mask_path)
        contour = io.imread(contour_path)
        center = io.imread(center_path)

        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        contour = contour.reshape(contour.shape[0], contour.shape[1], 1)
        center = center.reshape(center.shape[0], center.shape[1], 1)

        img = self.x_transform(img)
        mask = self.y_transform(mask)
        contour = self.y_transform(contour)
        center = self.y_transform(center)
        
        return {'img': img, 'mask': mask, 'contour': contour, 'center': center}

x_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
])

y_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])