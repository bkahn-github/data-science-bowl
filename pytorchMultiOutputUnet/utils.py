import os
import glob
import logging

import torch

from config import config
from sklearn.model_selection import train_test_split

class EarlyStopping:
    def __init__(self):
        
        self.best_score = 1
        self.best_epoch = 0 

    def evaluate(self, model, loss, epoch, patience=0):

        if torch.cuda.is_available():         
            loss = loss.data.cpu().numpy()[0]
        else:
            loss = loss.data.numpy()[0]            
        
        if loss < self.best_score:
            logging.info('Val score has improved, saving model\n')
            self.best_score = loss
            self.best_epoch = epoch
            save_model(model)

        elif epoch - self.best_epoch > patience:
            logging.info('Val score hasn\'t improved for more than ' + str(patience) + 'epochs, stopping training\n')
        else:
            logging.info('Val score hasn\'t improved for more than ' + str(epoch - self.best_epoch) + 'epochs, not saving model\n')

def print_losses(train_loss, val_loss, epoch):

    if torch.cuda.is_available():         
        train_loss = train_loss.data.cpu().numpy()[0]
        val_loss = val_loss.data.cpu().numpy()[0]
    else:
        train_loss = train_loss.data.numpy()[0]
        val_loss = val_loss.data.numpy()[0]

    message = '\nEpoch # ' + str(epoch) + ' | Training Loss: ' + str(round(train_loss, 4)) + ' | Validation Loss: ' + str(round(val_loss, 4))
    
    print(message)

def save_model(model):
    torch.save(model.state_dict(), './model-best.pt')

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
