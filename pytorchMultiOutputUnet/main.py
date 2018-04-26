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
# from visualize import show_images
from metrics import dice_loss
from utils import get_splits, calculate_losses, save_model, load_model, EarlyStopping

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

def train(epochs, weights, splits):
    logging.info('Starting Training')
    logging.info('Training for ' + str(epochs) + ' epochs')

    splits = get_splits(splits)
    logging.info(str(len(splits)) + ' splits in cross validation')

    if weights != '':
        model = Unet()
        model = load_model(model, weights)

    splits_train_losses = 0
    splits_val_losses = 0

    for i, split in enumerate(splits):
        print('\n')
        logging.info('=' * 50)
        logging.info('Split # ' + str(i + 1))

        train_ids, val_ids = split[0], split[1]

        logging.info('Creating Dataset')
        train = TrainDataset(train_ids, x_transform=x_transforms, y_transform=y_transforms)
        trainDataloader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)
        val = TrainDataset(val_ids, x_transform=x_transforms, y_transform=y_transforms)
        valDataloader = DataLoader(val, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

        if weights != '' and i == 0:
            model = model
            weights = ''
        else:
            model = Unet()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        early_stopping = EarlyStopping()

        for epoch in range(epochs):
            epoch += 1
            print('\n')
            logging.info('-' * 50)
            logging.info('Epoch # ' + str(epoch))
            
            total_train_loss = 0
            for data in tqdm(trainDataloader):
                img, target = data['img'], data['target']

                x = Variable(img).to(device)            
                y = Variable(target).to(device)

                optimizer.zero_grad()

                outs = model(x)
                train_loss = dice_loss(outs, y)
                total_train_loss += train_loss.item()

                train_loss.backward()
                optimizer.step()

            total_val_loss = 0
            with torch.no_grad():
                for data in tqdm(valDataloader):
                    img, target = data['img'], data['target']

                    x = Variable(img).to(device)
                    y = Variable(target).to(device)

                    optimizer.zero_grad()

                    outs = model(x)
                    val_loss = dice_loss(outs, y)
                    total_val_loss += val_loss.item()

            message, train_loss, val_loss = calculate_losses(total_train_loss, total_val_loss, train_ids, val_ids, epoch)
            splits_train_losses += train_loss
            splits_val_losses += val_loss

            print(message)

            action = early_stopping.evaluate(model, val_loss, epoch, config.PATIENCE)

            if action == 'save':
                save_model(model, i)
            elif action == 'stop':
                break
            else:
                continue
    
    message, train_loss, val_loss = calculate_losses(splits_train_losses, splits_val_losses, train_ids, val_ids, epoch)
    print(message)

def visualize(weights):
    return ''
    # show_images(weights)
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')

    parser = argparse.ArgumentParser()

    parser.add_argument("mode")

    parser.add_argument("--rootFolder")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--imgsFolder")
    parser.add_argument("--masksOutputFolder")
    parser.add_argument("--contoursOutputFolder")
    parser.add_argument("--centersOutputFolder")
    parser.add_argument("--subset")
    parser.add_argument("--shuffle")
    parser.add_argument("--batchSize", type=int)
    parser.add_argument("--numWorkers", type=int)

    parser.add_argument('--splits', type=int)    
    parser.add_argument("--patience", type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument("--weights")

    args = parser.parse_args()

    if args.rootFolder:
        config.ROOT_FOLDER = args.rootFolder
        logging.info('Root folder has been changed to ' + config.ROOT_FOLDER)

    if args.stage:
        config.STAGE = args.stage
        logging.info('Stage has been changed to ' + config.STAGE)

    if args.imgsFolder:
        config.IMGS_FOLDER = args.imgsFolder
        logging.info('Imgs folder has been changed to ' + config.IMGS_FOLDER)

    if args.masksOutputFolder:
        config.MASKS_OUTPUT_FOLDER = args.masksOutputFolder
        logging.info('Masks output folder has been changed to ' + config.MASKS_OUTPUT_FOLDER)

    if args.contoursOutputFolder:
        config.CONTOURS_OUTPUT_FOLDER = args.contoursOutputFolder
        logging.info('Contours output folder has been changed to ' + config.CONTOURS_OUTPUT_FOLDER)        

    if args.centersOutputFolder:
        config.CENTERS_OUTPUT_FOLDER = args.centersOutputFolder
        logging.info('Centers output folder has been changed to ' + config.CENTERS_OUTPUT_FOLDER)

    if args.subset:
        subset(args.subset)
    
    if args.shuffle:
        config.SHUFFLE = args.shuffle
        logging.info('Shuffle has been changed to ' + config.SHUFFLE)

    if args.batchSize:
        config.BATCH_SIZE = args.batchSize
        logging.info('Batch size has been changed to ' + str(config.BATCH_SIZE))

    if args.numWorkers:
        config.NUM_WORKERS = args.numWorkers
        logging.info('Num workers has been changed to ' + str(config.NUM_WORKERS))

    if args.splits:
        config.SPLITS = args.splits
        logging.info('Splits has been changed to ' + str(config.SPLITS))

    if args.patience:
        config.PATIENCE = args.patience
        logging.info('Patience has been changed to ' + str(config.PATIENCE))

    if args.epochs:
        config.EPOCHS = args.epochs
        logging.info('Epochs has been changed to ' + str(config.EPOCHS))

    if args.weights:
        config.WEIGHTS = args.weights
        logging.info('Weights has been changed to ' + config.WEIGHTS)


    if args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'train':
        train(config.EPOCHS, config.WEIGHTS, config.SPLITS)
    elif args.mode == 'visualize':
        visualize(config.WEIGHTS)
    else:
        logging.info('You must provide an argument')