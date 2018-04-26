import os
import logging

import torch
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from config import config
from loaders import TrainDataset, x_transforms, y_transforms
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
  
    dataset = TrainDataset(kfolds[0][0], x_transform=x_transforms, y_transform=y_transforms)
    dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    with torch.no_grad():
        for data in dataLoader:
            img, target = data['img'], data['target']

            x = Variable(img).to(device)
            y = Variable(target).to(device)

            outs = model(x)
            break

    y = y.detach().cpu().numpy()
    outs = outs.detach().cpu().numpy()

    fig = plt.figure(figsize=(30, 20))

    ax = plt.subplot(4,4, 1)
    ax.set_title('Ground truth mask')
    ax.imshow(y[1][0].reshape(256, 256))

    ax = plt.subplot(4, 4, 2)
    ax.set_title('Ground truth contours')
    ax.imshow(y[1][1].reshape(256, 256))

    ax = plt.subplot(4, 4, 3)
    ax.set_title('Ground truth mask - contours')
    ax.imshow((((y[1][0] - y[1][1])) > threshold_otsu((y[1][0] - y[1][1]))).reshape(256, 256))

    ax = plt.subplot(4, 4, 4)
    ax.set_title('Predicted mask')
    ax.imshow((outs[1][0]).reshape(256, 256))

    ax = plt.subplot(4, 4, 5)
    ax.set_title('Predicted mask with Otsu thresholding')
    ax.imshow((outs[1][0] > threshold_otsu(outs[1][0])).reshape(256, 256))

    ax = plt.subplot(4, 4, 6)
    ax.set_title('Predicted contours')
    ax.imshow((outs[1][1]).reshape(256, 256))

    ax = plt.subplot(4, 4, 7)
    ax.set_title('Predicted contours with Otsu thresholding')
    ax.imshow((outs[1][1] > threshold_otsu(outs[1][1])).reshape(256, 256))

    ax = plt.subplot(4, 4, 8)
    ax.set_title('Predicted mask - contours')
    ax.imshow((outs[1][0]) - (outs[1][1]).reshape(256, 256))  

    ax = plt.subplot(4,4, 9)
    ax.set_title('Predicted mask - contours with Otsu thresholding')
    ax.imshow(((outs[1][0] - outs[1][1]) > threshold_otsu((outs[1][0]) - (outs[1][1])) ).reshape(256, 256))