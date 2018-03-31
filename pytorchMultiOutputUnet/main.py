import os
import click
import logging

import torch
import torchvision

from torch.utils.data import DataLoader

from config import root_folder, stage, imgs_folder, masks_output_folder, contours_output_folder, centers_output_folder, subset
from create_masks import create_masks
from loaders import TrainDataset, train_transforms
from utils import get_ids
    
@click.group()
def action():
    pass

@action.command()
def preprocess():
    logging.info('Starting Preprocessing')
    logging.info('Creating masks')
    create_masks(root_folder, stage, 'train', masks_output_folder, 'masks', subset)
    logging.info('Creating contours')    
    create_masks(root_folder, stage, 'train', contours_output_folder, 'contours', subset)
    logging.info('Creating centers')
    create_masks(root_folder, stage, 'train', centers_output_folder, 'centers', subset)

@action.command()
def train():
    logging.info('Starting Training')

    logging.info('Getting Ids')
    ids = get_ids()

    logging.info('Creating Training Dataset')
    train = TrainDataset(ids, transform=train_transforms)
    trainDataloader = DataLoader(train, batch_size=4, shuffle=True, num_workers=4)

    print(train[0])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()