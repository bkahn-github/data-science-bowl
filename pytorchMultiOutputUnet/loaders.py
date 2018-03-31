import os
import glob
import numpy as np

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
from PIL import Image

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

        if self.subset:
            ids = glob.glob(os.path.join(self.root_folder, 'stage' + self.stage + '_train', '*'))[:10]
        else:
            ids = glob.glob(os.path.join(self.root_folder, 'stage' + self.stage + '_train', '*'))

        self.ids = [id.split('/')[-1] for id in ids]

    def __len__(self):
        if self.subset:
            return len(glob.glob(os.path.join(self.root_folder, 'stage' + self.stage + '_train_masks', '*'))[:10])
        else:
            return len(glob.glob(os.path.join(self.root_folder, 'stage' + self.stage + '_train_masks', '*')))

    def __getitem__(self, idx):
        id = self.ids[idx]

        img_path = os.path.join(self.root_folder, 'stage' + self.stage + '_train', id, 'images', id + '.png')

        mask_path = os.path.join(self.root_folder, 'stage' + self.stage + '_train_masks', id + '.png')
        contour_path = os.path.join(self.root_folder, 'stage' + self.stage + '_train_contours', id + '.png')
        center_path = os.path.join(self.root_folder, 'stage' + self.stage + '_train_centers', id + '.png')
        
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

        img = self.transform(img)
        mask = self.transform(mask)
        contour = self.transform(contour)
        center = self.transform(center)
        
        return {'img': img, 'mask': mask, 'contour': contour, 'center': center}

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
])