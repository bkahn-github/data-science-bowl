import os
import argparse
import logging
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision

from config import config
from create_masks import create_masks
from loaders import TrainDataset, augmentation
from metrics import loss
from model import Unet
from utils import EarlyStopping, calculate_kfolds_losses, calculate_losses, get_kfolds, load_model, save_model
from visualize import show_images

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
    create_masks(config.ROOT_FOLDER, config.STAGE, 'train', config.TARGETS_FOLDER, 'masks', config.SUBSET)

def train(epochs, weights, kfolds):
    logging.info('Starting Training')
    logging.info('Training for ' + str(epochs) + ' epochs')

    kfolds = get_kfolds(kfolds)
    logging.info(str(len(kfolds)) + ' kfolds in cross validation')

    if weights != '':
        model = Unet()
        model = load_model(model, weights)

    total_kfolds_train_loss = 0
    total_kfolds_val_loss = 0

    for i, kfold in enumerate(kfolds):
        print('\n')
        logging.info('=' * 50)
        logging.info('Kfold # ' + str(i + 1))

        train_ids, val_ids = kfold[0], kfold[1]

        logging.info('Creating Dataset')
        train = TrainDataset(train_ids, augmentation=augmentation)
        trainDataloader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)
        val = TrainDataset(val_ids, augmentation=augmentation)
        valDataloader = DataLoader(val, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS)

        if weights != '' and i == 0:
            model = model
            weights = ''
        else:
            model = Unet()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.LR)

        early_stopping = EarlyStopping()

        for epoch in range(epochs):
            epoch += 1
            print('\n')
            logging.info('-' * 50)
            logging.info('Epoch # ' + str(epoch))
            
            total_train_loss = 0
            for data in tqdm(trainDataloader):
                img, mask = data['img'], data['mask']

                x = img.requires_grad_().to(device)
                y = mask.requires_grad_().to(device)

                optimizer.zero_grad()

                outs = model(x)
                train_loss = loss(outs, y)
                total_train_loss += (torch.sum(train_loss.view(-1)) / len(train_loss.view(-1))).item()

                train_loss.backward(gradient=train_loss)
                optimizer.step()

            total_val_loss = 0
            with torch.no_grad():
                for data in tqdm(valDataloader):
                    img, mask = data['img'], data['mask']

                    x = img.to(device)
                    y = mask.to(device)

                    optimizer.zero_grad()

                    outs = model(x)
                    val_loss = loss(outs, y)
                    total_val_loss += (torch.sum(val_loss.view(-1)) / len(val_loss.view(-1))).item()

            message, train_loss, val_loss = calculate_losses(total_train_loss, total_val_loss, train_ids, val_ids, epoch)
            print(message)

            total_kfolds_train_loss += train_loss
            total_kfolds_val_loss += val_loss

            action = early_stopping.evaluate(model, val_loss, epoch, config.PATIENCE)

            if action == 'save':
                save_model(model, i)
            elif action == 'stop':
                break
            else:
                continue
    
    message = calculate_kfolds_losses(total_kfolds_train_loss, total_kfolds_val_loss, config.KFOLDS, config.EPOCHS)
    print(message)

def visualize(weights):
    show_images(weights)
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')

    parser = argparse.ArgumentParser()

    parser.add_argument("mode")

    parser.add_argument("--rootFolder")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--imgsFolder")
    parser.add_argument("--masksOutputFolder")
    parser.add_argument("--edgesOutputFolder")
    parser.add_argument("--subset")
    parser.add_argument("--subsetSize", type=int)    
    parser.add_argument("--shuffle")
    parser.add_argument("--batchSize", type=int)
    parser.add_argument("--numWorkers", type=int)

    parser.add_argument('--kfolds', type=int)    
    parser.add_argument("--patience", type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument("--weights")

    parser.add_argument("--augment")
    parser.add_argument("--clipLimit", type=int)
    parser.add_argument("--gridSize")
    parser.add_argument("--invert", type=int)
    parser.add_argument("--randomCrop", type=int)
    parser.add_argument("--flipLR", type=float)
    parser.add_argument("--flipUD", type=float)
    parser.add_argument("--rotate", type=int)

    args = parser.parse_args()

    if args.rootFolder:
        config.ROOT_FOLDER = args.rootFolder
        logging.info('Root folder has been changed to ' + config.ROOT_FOLDER)

    if args.stage:
        config.STAGE = args.stage
        logging.info('Stage has been changed to ' + str(config.STAGE))

    if args.imgsFolder:
        config.IMGS_FOLDER = args.imgsFolder
        logging.info('Imgs folder has been changed to ' + config.IMGS_FOLDER)

    if args.masksOutputFolder:
        config.MASKS_OUTPUT_FOLDER = args.masksOutputFolder
        logging.info('Masks output folder has been changed to ' + config.MASKS_OUTPUT_FOLDER)

    if args.edgesOutputFolder:
        config.EDGES_OUTPUT_FOLDER = args.edgesOutputFolder
        logging.info('Edges output folder has been changed to ' + config.EDGES_OUTPUT_FOLDER)        

    if args.subset:
        subset(args.subset)

    if args.subsetSize:
        config.SUBSET_SIZE = args.subsetSize
        logging.info('Subset size has been changed to ' + str(config.SUBSET_SIZE))

    if args.shuffle:
        config.SHUFFLE = args.shuffle
        logging.info('Shuffle has been changed to ' + config.SHUFFLE)

    if args.batchSize:
        config.BATCH_SIZE = args.batchSize
        logging.info('Batch size has been changed to ' + str(config.BATCH_SIZE))

    if args.numWorkers:
        config.NUM_WORKERS = args.numWorkers
        logging.info('Num workers has been changed to ' + str(config.NUM_WORKERS))

    if args.kfolds:
        config.KFOLDS = args.kfolds
        logging.info('Kfolds has been changed to ' + str(config.KFOLDS))

    if args.patience:
        config.PATIENCE = args.patience
        logging.info('Patience has been changed to ' + str(config.PATIENCE))

    if args.epochs:
        config.EPOCHS = args.epochs
        logging.info('Epochs has been changed to ' + str(config.EPOCHS))

    if args.lr:
        config.LR = args.lr
        logging.info('lr has been changed to ' + str(config.LR))

    if args.weights:
        config.WEIGHTS = args.weights
        logging.info('Weights has been changed to ' + config.WEIGHTS)

    if args.augment:
        config.AUGMENT = args.augment
        logging.info('Augment has been changed to ' + config.AUGMENT)

    if args.clipLimit:
        config.CLIP_LIMIT = args.clipLimit
        logging.info('Clip limit has been changed to ' + str(config.CLIP_LIMIT))

    if args.gridSize:
        config.GRID_SIZE = args.gridSize
        logging.info('Grid size has been changed to ' + str(config.GRID_SIZE))

    if args.invert:
        config.INVERT = args.invert
        logging.info('Invert has been changed to ' + str(config.INVERT))

    if args.randomCrop:
        config.RANDOM_CROP = args.randomCrop
        logging.info('Random crop has been changed to ' + str(config.RANDOM_CROP))

    if args.flipLR:
        config.FLIP_LR = args.flipLR
        logging.info('Flip LR has been changed to ' + str(config.FLIP_LR))

    if args.flipUD:
        config.FLIP_UD = args.flipUD
        logging.info('Flip UD has been changed to ' + str(config.FLIP_UD))

    if args.rotate:
        config.ROTATE = args.rotate
        logging.info('Rotate has been changed to ' + str(config.ROTATE))

    if args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'train':
        train(config.EPOCHS, config.WEIGHTS, config.KFOLDS)
    elif args.mode == 'visualize':
        visualize(config.WEIGHTS)
    else:
        logging.info('You must provide an argument')
