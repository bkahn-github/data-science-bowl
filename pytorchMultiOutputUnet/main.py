import os
import click
import logging
from tqdm import tqdm
from glob import glob

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
@click.option('--weights', default='', help='Path to weights')
def train(epochs, weights):
    logging.info('Starting Training')
    logging.info('Training for ' + str(epochs) + ' epochs')

    logging.info('Getting Ids')
    train_ids, val_ids = get_ids()

    logging.info('Creating Training Dataset')
    train = TrainDataset(train_ids, x_transform=x_transforms, y_transform=y_transforms)
    trainDataloader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    logging.info('Creating Validation Dataset')
    val = TrainDataset(val_ids, x_transform=x_transforms, y_transform=y_transforms)
    valDataloader = DataLoader(val, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

    model = Unet()

    if torch.cuda.is_available():
        model.cuda()

    startingEpoch = 0
    if weights != '':
        startingEpoch = weights.split('-')[-1].split('.')[0]
        logging.info('Starting from epoch ' + startingEpoch)
        model.load_state_dict(torch.load(weights))

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(epochs):
        epoch = epoch + int(startingEpoch) + 1

        logging.info('Epoch # ' + str(epoch))
        total_train_loss = 0
        for data in tqdm(trainDataloader):
            img, target = data['img'], data['target']

            if torch.cuda.is_available():         
                x = Variable(img).cuda()
                y = Variable(target).cuda()
            else:
                x = Variable(img)
                y = Variable(target)                

            optimizer.zero_grad()

            outs = model(x)

            train_loss = dice_loss(outs, y)
            total_train_loss += train_loss.data.cpu().numpy()[0]
            train_loss.backward()

            optimizer.step()

        avg_train_loss = total_train_loss.data.cpu().numpy()[0] / len(train_ids)

        print('\nTraining Loss: ' + str(avg_train_loss))
        torch.save(model.state_dict(), './model-' + str(epoch) + '.pt')
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()