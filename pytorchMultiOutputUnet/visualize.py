import os
import logging
import numpy as np

import torch
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from config import config
from loaders import TrainDataset, augmentation
from model import Unet
from utils import get_kfolds

def show_images(weights):
    logging.info('Visualizing model')
    model = Unet()
    model.load_state_dict(torch.load(weights))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
  
    kfolds = get_kfolds(2)
  
    dataset = TrainDataset(kfolds[0][0], augmentation=augmentation)
    dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    with torch.no_grad():
        for data in dataLoader:
            img, target = data['img'], data['target']

            x = img.to(device)
            y = target.to(device)

            outs = model(x)
            break

    x = x.detach().cpu().numpy()    
    y = y.detach().cpu().numpy()
    outs = outs.detach().cpu().numpy()

    x = np.concatenate((x[0][0:1].reshape(256, 256, 1), x[0][1:2].reshape(256, 256, 1), (x[0][2] > 0).astype(np.uint8).reshape(256, 256, 1)), axis=-1).reshape(256, 256, 3)
    y = np.concatenate((y[0][0:1].reshape(256, 256, 1), y[0][1:2].reshape(256, 256, 1), (y[0][2] > 0).astype(np.uint8).reshape(256, 256, 1)), axis=-1).reshape(256, 256, 3)
    outs = np.concatenate((outs[0][0:1].reshape(256, 256, 1), outs[0][1:2].reshape(256, 256, 1), (outs[0][2]).reshape(256, 256, 1)), axis=-1).reshape(256, 256, 3)

    fig = plt.figure(figsize=(20, 20))

    ax = plt.subplot(2, 3, 1)
    ax.set_title('Image')
    ax.imshow(x)

    ax = plt.subplot(2, 3, 2)
    ax.set_title('Ground truth')
    ax.imshow(y)

    ax = plt.subplot(2, 3, 3)
    ax.set_title('Predicted mask')
    ax.imshow(outs[:,:,:])

    ax = plt.subplot(2, 3, 4)
    ax.set_title('Predicted edges')
    ax.imshow(outs[:,:,1])

    ax = plt.subplot(2, 3, 5)
    ax.set_title('Predicted mask - edges')
    ax.imshow(outs[:,:,0] - outs[:,:,1])
