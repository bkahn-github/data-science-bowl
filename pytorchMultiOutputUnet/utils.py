import os
import glob
import logging

import torch

from config import config

from sklearn.model_selection import KFold

class EarlyStopping:
    def __init__(self):
        
        self.best_score = 1
        self.best_epoch = 0 

    def evaluate(self, model, loss, epoch, patience=0):
        
        if loss < self.best_score:
            logging.info('Val score has improved, saving model')
            self.best_score = loss
            self.best_epoch = epoch
            return 'save'

        elif epoch - self.best_epoch > patience:
            logging.info('Val score hasn\'t improved for ' + str(epoch - self.best_epoch) + ' epochs, stopping training')
            return 'stop'
        else:
            logging.info('Val score hasn\'t improved for ' + str(epoch - self.best_epoch) + ' epochs, not saving model')
            return 'continue'
    
def calculate_losses(total_train_loss, total_val_loss, train_ids, val_ids, epoch):
    train_loss = total_train_loss / (len(train_ids) / config.BATCH_SIZE)
    val_loss = total_val_loss / (len(val_ids) / config.BATCH_SIZE)

    message = 'Epoch # ' + str(epoch) + ' | Training Loss: ' + str(round(train_loss, 4)) + ' | Validation Loss: ' + str(round(val_loss, 4))
    
    return message, train_loss, val_loss

def save_model(model, split, epoch):
    torch.save(model.state_dict(), './model-split-' + str(split) + '-epoch-' + str(epoch) + '.pt')

def load_model(model, path):
    startingEpoch = path.split('-')[-1].split('.')[0]
    logging.info('Starting from epoch ' + startingEpoch)
    model.load_state_dict(torch.load(path))

    return model, startingEpoch

def get_splits(splits):
    if config.SUBSET:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))[:20]
    else:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))

    ids = [id.split('/')[-1] for id in ids]

    kf = KFold(n_splits=splits)

    splits = []
    for x, y in kf.split(ids):
        x = ids[x[0]: x[-1]]
        y = ids[y[0]: y[-1]]

        splits.append([x, y])

    return splits

def get_path(id):
    img_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', id, 'images', id + '.png')

    mask_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_masks', id + '.png')
    contour_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_contours', id + '.png')
    center_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_centers', id + '.png')

    return img_path, mask_path, contour_path, center_path
