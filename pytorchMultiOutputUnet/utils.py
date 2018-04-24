import os
import glob

from config import config
from sklearn.model_selection import train_test_split

class EarlyStopping():
    def __init__(self):
        
        self.best_score = 1
        self.best_epoch = 0 

    @classmethod
    def check(self, loss, epoch):
        if loss < self.best_score:
            self.best_score = loss
            self.best_epoch = epoch
            print('OK')
        elif epoch - self.best_epoch > 0:
            print('Stopped')

def get_ids():
    if config.SUBSET:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))[:10]
    else:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))

    ids = [id.split('/')[-1] for id in ids]

    train_ids, val_ids = train_test_split(ids, test_size=config.TEST_SIZE)

    return train_ids, val_ids

def get_path(id):
    img_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', id, 'images', id + '.png')

    mask_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_masks', id + '.png')
    contour_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_contours', id + '.png')
    center_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_centers', id + '.png')

    return img_path, mask_path, contour_path, center_path
