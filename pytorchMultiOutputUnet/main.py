import os
import click
import logging

from create_masks import create_masks
from loaders import TrainDataset, train_transforms

# root_folder = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/'
root_folder = '~/.kaggle/competitions/data-science-bowl-2018/'

imgs_folder = 'stage1_train'

masks_output_folder = 'stage1_masks'
contours_output_folder = 'stage1_contours'
centers_output_folder = 'stage1_centers'

subset = True
    
@click.group()
def action():
    pass

@action.command()
def preprocess():
    logging.info('Starting Preprocessing')
    logging.info('Creating masks')
    create_masks(root_folder, '1', 'train', masks_output_folder, 'masks', subset)
    logging.info('Creating contours')    
    create_masks(root_folder, '1', 'train', contours_output_folder, 'contours', subset)
    logging.info('Creating centers')
    create_masks(root_folder, '1', 'train', centers_output_folder, 'centers', subset)

@action.command()
def train():
    logging.info('Starting Training')
    train = TrainDataset('1', root_folder, imgs_folder, masks_output_folder, contours_output_folder, centers_output_folder, subset=True, transform=train_transforms)
    train[0]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()