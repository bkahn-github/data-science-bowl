import os
import click
import logging

from create_masks import create_masks#, create_contours, create_centers

root_folder = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/'
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
    create_masks(root_folder, '1', 'train', masks_output_folder, subset)
    # create_contours(root_folder, train_folder, contours_output_folder)
    # create_centers(root_folder, train_folder, centers_output_folder)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s >>> %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
    logging.info('Started the program')
    action()