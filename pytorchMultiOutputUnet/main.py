import os
import click
import logging
from tqdm import tqdm

import torch
import torchvision

from torch.utils.data import DataLoader

from config import config
from create_masks import create_masks
from loaders import TrainDataset, train_transforms
from model import Unet
from utils import get_ids

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
    
@click.group()
def action():
    pass

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
def train():
    logging.info('Starting Training')

    logging.info('Getting Ids')
    ids = get_ids()

    logging.info('Creating Training Dataset')
    train = TrainDataset(ids, transform=train_transforms)
    trainDataloader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    model = Unet()

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for data in tqdm(trainDataloader):
        img, mask, contour, center = data['img'], data['mask'], data['contour'], data['center']

        if torch.cuda.is_available():         
            x = Variable(img).cuda()
            y = [Variable(mask).cuda(), Variable(contour).cuda(), Variable(center).cuda()]
        else:
            x = Variable(img)
            y = [Variable(mask), Variable(contour), Variable(center)]


        optimizer.zero_grad()

        outs = model(x)

        losses = []
        for i, out in enumerate(outs):
            loss = criterion(out, y[i])
            losses.append(loss)

        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()