import os
import glob
import numpy as np

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class TrainDataset(Dataset):
    def __init__(self, stage, root_folder, imgs_folder, mask_folder, contours_folder, centers_folder, subset=False, transform=None):
        self.stage = stage
        
        self.root_folder = root_folder

        self.imgs_folder = imgs_folder
        self.mask_folder = mask_folder
        self.contours_folder = contours_folder
        self.centers_folder = centers_folder

        self.subset = subset
        self.transform = transform

    def __len__(self):
        if self.subset:
            return 10
        else:
            return len(glob.glob(os.path.join(self.root_folder, 'stage' + self.stage + '_train_masks', '*')))

    def __getitem__(self, idx):
        return None