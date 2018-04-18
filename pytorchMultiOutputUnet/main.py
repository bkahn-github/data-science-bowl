import os
import click
import logging
from tqdm import tqdm

import torch
import torchvision

from torch.utils.data import DataLoader

from config import config
from create_masks import create_masks
from loaders import TrainDataset, x_transforms, y_transforms
from model import Unet
from metrics import dice_loss
from utils import get_ids

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
    
@click.group(chain=True)
def action():
    pass

@action.command()
@click.option('--subset', default=False, help='Use a subset of the data')
def subset(subset):
    if subset == 'True':
        logging.info('Using a subset')
        config.SUBSET = True
    else:
        logging.info('Using the full dataset')
        config.SUBSET = False

@action.command()
def preprocess():
    logging.info('Starting Preprocessing')
    logging.info('Creating masks')
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.MASKS_OUTPUT_FOLDER, 'masks', config.SUBSET)
    logging.info('Creating contours')    
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.CONTOURS_OUTPUT_FOLDER, 'contours', config.SUBSET)
    logging.info('Creating centers')
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.CENTERS_OUTPUT_FOLDER, 'centers', config.SUBSET)

@action.command()
@click.option('--epochs', default=10, help='Number of epochs')
def train(epochs):
    logging.info('Starting Training')
    logging.info('Training for ' + str(epochs) + ' epochs')

    logging.info('Getting Ids')
    ids = get_ids()

    logging.info('Creating Training Dataset')
    train = TrainDataset(ids, x_transform=x_transforms, y_transform=y_transforms)
    trainDataloader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    model = Unet()

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(epochs):
        logging.info('Epoch # ' + str(epoch))
        for data in tqdm(trainDataloader):
            img, mask, contour, center = data['img'], data['mask'], data['contour'], data['center']

            if torch.cuda.is_available():         
                x = Variable(img).cuda()
                # y = [Variable(mask).cuda(), Variable(contour).cuda(), Variable(center).cuda()]
                y = Variable(mask).cuda()
            else:
                x = Variable(img)
                # y = [Variable(mask), Variable(contour), Variable(center)]
                y = Variable(mask)                

            optimizer.zero_grad()

            outs = model(x)

            total_loss = dice_loss(outs, y)
            # losses = [
            #     ('mask_loss', criterion(outs[0], y[0]) * 0.3),
            #     ('contour_loss', criterion(outs[1], y[1]) * 0.6),
            #     ('center_loss', criterion(outs[2], y[2]) * 0.1)]

            # total_loss = 0
            # for loss in losses:
            #     total_loss += loss[1]

            total_loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), './model-' + str(epoch) + '.pt')
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()