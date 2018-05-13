import os
import logging

import torch
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from config import config
from loaders import TrainDataset, x_transforms, target_transforms
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
  
    dataset = TrainDataset(kfolds[0][0], x_transform=x_transforms, target_transforms=target_transforms)
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

    img = np.concatenate((outs[0][0:1].reshape(256, 256, 1), outs[0][1:2].reshape(256, 256, 1), (outs[0][2]).reshape(256, 256, 1)), axis=-1).reshape(256, 256, 3)
    truth = np.concatenate((y[0][0:1].reshape(256, 256, 1), y[0][1:2].reshape(256, 256, 1), (y[0][2] > 0).astype(np.uint8).reshape(256, 256, 1)), axis=-1).reshape(256, 256, 3)

    fig = plt.figure(figsize=(20, 20))

    ax = plt.subplot(2, 2, 1)
    ax.set_title('Ground truth')
    ax.imshow(truth)

    ax = plt.subplot(2, 2, 2)
    ax.set_title('Predicted mask')
    ax.imshow(img[:,:,:])

    ax = plt.subplot(2, 2, 3)
    ax.set_title('Predicted edges')
    ax.imshow(img[:,:,1])

    ax = plt.subplot(2, 2, 4)
    ax.set_title('Predicted mask - edges')
    ax.imshow(img[:,:,0] - img[:,:,1])
