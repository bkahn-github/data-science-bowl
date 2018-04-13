import os
import glob

from config import config

def get_ids():
    if config.SUBSET:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))[:10]
    else:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))

    ids = [id.split('/')[-1] for id in ids]
    return ids

def get_path(id):
    img_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', id, 'images', id + '.png')

    mask_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_masks', id + '.png')
    contour_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_contours', id + '.png')
    center_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_centers', id + '.png')

    return img_path, mask_path, contour_path, center_path
