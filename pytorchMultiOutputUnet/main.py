import os
import logging
from tqdm import tqdm
from glob import glob
import argparse

import torch
import torchvision

from torch.utils.data import DataLoader

from config import config
from create_masks import create_masks
from loaders import TrainDataset, x_transforms, y_transforms
from model import Unet
from train import model_loop
from visualize import show_images
from metrics import dice_loss
from utils import get_ids, print_losses, save_model, load_model, EarlyStopping

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
    
def subset(subset):
    if subset == 'True':
        logging.info('Using a subset')
        config.SUBSET = True
    else:
        logging.info('Using the full dataset')
        config.SUBSET = False

def preprocess():
    logging.info('Starting Preprocessing')
    logging.info('Creating masks')
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.MASKS_OUTPUT_FOLDER, 'masks', config.SUBSET)
    logging.info('Creating contours')    
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.CONTOURS_OUTPUT_FOLDER, 'contours', config.SUBSET)
    logging.info('Creating centers')
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.CENTERS_OUTPUT_FOLDER, 'centers', config.SUBSET)

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

    if weights != '':
        model, startingEpoch = load_model(model, weights)
    else:
        startingEpoch = 0

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        epoch = epoch + int(startingEpoch) + 1
        logging.info('Epoch # ' + str(epoch))
        
        model, train_loss = model_loop(trainDataloader, model, optimizer)
        model, val_loss = model_loop(valDataloader, model, optimizer, is_training=False)

        print_losses(train_loss, val_loss, epoch)
        action = early_stopping.evaluate(model, val_loss, epoch, config.PATIENCE)

        if action == 'save':
            save_model(model)
        elif action == 'stop':
            break
        else:
            continue

def visualize(weights, subset):
    if subset == 'True':
        config.SUBSET = True
    else:
        config.SUBSET = False

    show_images(weights)
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')

    parser = argparse.ArgumentParser()

    parser.add_argument("mode")

    parser.add_argument("--subset")

    parser.add_argument('--epochs', type=int)
    parser.add_argument("--weights")

    args = parser.parse_args()

    if args.subset:
        subset(args.subset)

    if args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'train':
        if args.epochs:
            if args.weights:
                train(args.epochs, args.weights)
            else:
                train(args.epochs, '')
        else:
            logging.info('You must give a number of epochs')
    elif args.mode == 'visualize':
        if args.weights:
            visualize(args.weights)
        else:
            logging.info('You must give model file')
    else:
        logging.info('You must provide an argument')