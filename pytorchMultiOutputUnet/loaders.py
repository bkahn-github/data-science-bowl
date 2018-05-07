import os
import glob
import numpy as np

import torch
import torchvision

from utils import get_path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
from PIL import Image

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
        
        img = Image.open(img_path)
        img.load()

        RGBimg = Image.new("RGB", img.size, (255, 255, 255))
        RGBimg.paste(img, mask=img.split()[3])
        
        img = np.array(RGBimg)

        target = io.imread(targets_path)
        target = target.reshape(target.shape[0], target.shape[1], 3)

        img = self.x_transform(img)
        target = self.target_transforms(target)

        return {'img': img, 'target': target}


x_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

target_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])